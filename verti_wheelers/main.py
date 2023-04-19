import gc
from pathlib import Path
from datetime import datetime
import sys
import json
import argparse

try:
    sys.path.append(str(Path(".").resolve()))
except:
    raise RuntimeError(
        "Can't append root directory of the project to the path")

import comet_ml
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR

from model.net import VWBehaviorCloning
from model.data_loader import VWDataset
from utils.nn import check_grad_norm, init_weights, EarlyStopping, op_counter
from utils.io import save_checkpoint, load_checkpoint
from utils.helpers import get_conf, timeit
from utils.vision import draw_on_image


class Learner:
    def __init__(self, cfg_dir: str):
        # load config file and initialize the logger and the device
        self.cfg = get_conf(cfg_dir)
        self.logger = self.init_logger(self.cfg.logger)
        self.device = self.init_device()
        # creating dataset interface and dataloader for trained data
        self.data, self.val_data = self.init_dataloader()
        # create model and initialize its weights and move them to the device
        self.model = self.init_model()
        # initialize the optimizer
        self.optimizer, self.lr_scheduler = self.init_optimizer()
        # define loss function
        self.criterion = torch.nn.MSELoss()
        # if resuming, load the checkpoint
        self.if_resume()

        # initialize the early_stopping object
        self.early_stopping = EarlyStopping(
            patience=self.cfg.train_params.patience,
            verbose=True,
            delta=self.cfg.train_params.early_stopping_delta,
        )

        # stochastic weight averaging
        if self.cfg.train_params.epochs > self.cfg.train_params.swa_start:
            self.swa_model = AveragedModel(self.model)
            self.swa_scheduler = SWALR(self.optimizer, **self.cfg.SWA)

    def train(self):
        """Trains the model"""
        # a variable to print the start of SWA
        print_swa_start = True

        while self.epoch <= self.cfg.train_params.epochs:
            running_loss = []

            bar = tqdm(
                self.data,
                desc=f"Epoch {self.epoch:03}/{self.cfg.train_params.epochs:03}, training: ",
            )
            for data in bar:
                self.iteration += 1
                (loss, grad_norm), t_train = self.forward_batch(data)
                t_train /= self.data.batch_size
                running_loss.append(loss)

                bar.set_postfix(loss=loss, Grad_Norm=grad_norm, Time=t_train)

                self.logger.log_metrics(
                    {
                        "batch_loss": loss,
                        "grad_norm": grad_norm,
                    },
                    epoch=self.epoch,
                    step=self.iteration,
                )

            bar.close()
            # update SWA model parameters
            if self.epoch > self.cfg.train_params.swa_start:
                if print_swa_start:
                    print(
                        f"Epoch {self.epoch:03}, step {self.iteration:05}, starting SWA!")
                    # print only once
                    print_swa_start = False

                self.swa_model.update_parameters(self.model)
                self.swa_scheduler.step()
            else:
                self.lr_scheduler.step()

            # validate on val set
            val_loss, t = self.validate()
            t /= len(self.val_data.dataset)

            # average loss for an epoch
            self.e_loss.append(np.mean(running_loss))  # epoch loss
            print(
                f"{datetime.now():%Y-%m-%d %H:%M:%S} Epoch {self.epoch:03}, " +
                f"Iteration {self.iteration:05} summary: train Loss: " +
                f"{self.e_loss[-1]:.2f} \t| Val loss: {val_loss:.2f}" +
                f"\t| time: {t:.3f} seconds\n"
            )

            self.logger.log_metrics(
                {
                    "train_loss": self.e_loss[-1],
                    "val_loss": val_loss,
                    "time": t,
                },
                epoch=self.epoch,
                step=self.iteration,
            )

            # early_stopping needs the validation loss to check if it has decreased,
            # and if it has, it will make a checkpoint of the current model
            self.early_stopping(val_loss, self.model)

            if self.early_stopping.early_stop and self.cfg.train_params.early_stopping:
                print(
                    f"{datetime.now():%Y-%m-%d %H:%M:%S} - Epoch {self.epoch:03}, Early stopping")
                self.save()
                break

            if self.epoch % self.cfg.train_params.save_every == 0 or (
                self.e_loss[-1] < self.best
                and self.epoch % self.cfg.train_params.start_saving_best == 0
            ):
                self.save()

            gc.collect()
            self.epoch += 1

        # Update bn statistics for the swa_model at the end
        if self.epoch >= self.cfg.train_params.swa_start:
            # if the first element of sample is the tensor that network should be applied to
            # otherwise, comment the line below, and uncomment the for loop
            # torch.optim.swa_utils.update_bn(self.data, self.swa_model)
            # otherwise, just run a forward pass of every sample in dataset through swa model
            # uncomment the for loop below
            for uid, x, y in self.data:
                x = x.to(device=self.device)
                self.swa_model(x)

            self.save(
                name=self.cfg.directory.model_name +
                "-final" + str(self.epoch) + "-swa"
            )
        x, _ = next(iter(self.data))
        macs, params = op_counter(self.model, sample=x.to(device=self.device))
        print("macs = ", macs, " | params = ", params)
        self.logger.log_metrics({"GFLOPS": macs[:-1], "#Params": params[:-1]})
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Training is DONE!")

    @timeit
    def forward_batch(self, data):
        """Forward pass of a batch"""
        self.model.train()
        # move data to device
        img, action = data
        img = img.to(device=self.device)
        action = action.to(device=self.device)

        # forward, backward
        out = self.model(img)
        loss = self.criterion(out, action)
        self.optimizer.zero_grad()
        loss.backward()
        # gradient clipping
        if self.cfg.train_params.grad_clipping > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.train_params.grad_clipping
            )
        # check grad norm for debugging
        grad_norm = check_grad_norm(self.model)
        # update
        self.optimizer.step()

        return loss.item(), grad_norm

    @timeit
    @torch.no_grad()
    def validate(self):

        self.model.eval()

        running_loss = []
        bar = tqdm(
            self.val_data, desc=f"Epoch {self.epoch:03}/{self.cfg.train_params.epochs:03}, validating")
        for img, action in bar:
            # move data to device
            img = img.to(device=self.device)
            action = action.to(device=self.device)

            # forward
            if self.epoch > self.cfg.train_params.swa_start:
                out = self.swa_model(img)
            else:
                out = self.model(img)

            loss = self.criterion(out, action)
            running_loss.append(loss.item())
            bar.set_postfix(loss=loss.item())

        bar.close()
        img_annotated = img[0].cpu().clone()
        img_annotated = draw_on_image(img_annotated, action[0], out[0], True)

        self.logger.log_image(img_annotated.squeeze(),
                              f"{out[0].argmax().item()}", step=self.iteration)

        # average loss
        loss = np.mean(running_loss)

        return loss

    def init_model(self):
        """Initializes the model"""
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - INITIALIZING the model!")
        model = VWBehaviorCloning(**self.cfg.model)

        if 'cuda' in str(self.device) and self.cfg.train_params.device.split(":")[1] == 'a':
            model = torch.nn.DataParallel(model)

        # model.apply(init_weights(**self.cfg.init_model))
        model = model.to(device=self.device)
        return model

    def init_optimizer(self):
        """Initializes the optimizer and learning rate scheduler"""
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - INITIALIZING the optimizer!")
        if self.cfg.train_params.optimizer.lower() == "adam":
            optimizer = optim.Adam(self.model.parameters(), **self.cfg.adam)

        elif self.cfg.train_params.optimizer.lower() == "rmsprop":
            optimizer = optim.RMSprop(
                self.model.parameters(), **self.cfg.rmsprop)

        elif self.cfg.train_params.optimizer.lower() == "sgd":
            optimizer = optim.SGD(self.model.parameters(), **self.cfg.sgd)

        else:
            raise ValueError(f"Unknown optimizer {self.cfg.train_params.optimizer}" +
                             "; valid optimizers are 'adam' and 'rmsprop'.")

        # initialize the learning rate scheduler
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.cfg.train_params.epochs
        )
        return optimizer, lr_scheduler

    def init_device(self):
        """Initializes the device"""
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - INITIALIZING the device!")
        is_cuda_available = torch.cuda.is_available()
        device = self.cfg.train_params.device

        if 'cpu' in device:
            print(f"Performing all the operations on CPU.")
            return torch.device(device)

        elif 'cuda' in device:
            if is_cuda_available:
                device_idx = device.split(":")[1]
                if device_idx == 'a':
                    print(
                        f"Performing all the operations on CUDA; {torch.cuda.device_count()} devices.")
                    self.cfg.dataloader.batch_size *= torch.cuda.device_count()
                    return torch.device(device.split(":")[0])
                else:
                    print(
                        f"Performing all the operations on CUDA device {device_idx}.")
                    return torch.device(device)
            else:
                print("CUDA device is not available, falling back to CPU!")
                return torch.device('cpu')
        else:
            raise ValueError(f"Unknown {device}!")

    def init_dataloader(self):
        """Initializes the dataloaders"""
        print(
            f"{datetime.now():%Y-%m-%d %H:%M:%S} - INITIALIZING the train and val dataloaders!")
        dataset = VWDataset(**self.cfg.dataset)
        data = DataLoader(dataset, **self.cfg.dataloader)
        # creating dataset interface and dataloader for val data
        self.cfg.val_dataset.update(self.cfg.dataset)
        self.cfg.val_dataset.update({'train': False})
        val_dataset = VWDataset(**self.cfg.val_dataset)

        self.cfg.dataloader.update({'shuffle': False})  # for val dataloader
        val_data = DataLoader(val_dataset, **self.cfg.dataloader)

        # log dataset status
        self.logger.log_parameters(
            {"train_len": len(dataset), "val_len": len(val_dataset)}
        )
        print(
            f"Training consists of {len(dataset)} samples, and validation consists of {len(val_dataset)} samples.")

        return data, val_data

    def if_resume(self):
        if self.cfg.logger.resume:
            # load checkpoint
            print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - LOADING checkpoint!!!")
            save_dir = self.cfg.directory.load
            checkpoint = load_checkpoint(save_dir, self.device)
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            self.epoch = checkpoint["epoch"] + 1
            self.e_loss = checkpoint["e_loss"]
            self.iteration = checkpoint["iteration"] + 1
            self.best = checkpoint["best"]
            print(
                f"{datetime.now():%Y-%m-%d %H:%M:%S} " +
                f"LOADING checkpoint was successful, start from epoch {self.epoch}" +
                f" and loss {self.best}"
            )
        else:
            self.epoch = 1
            self.iteration = 0
            self.best = np.inf
            self.e_loss = []

        self.logger.set_epoch(self.epoch)

    def init_logger(self, cfg):
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - INITIALIZING the logger!")
        logger = None
        # Check to see if there is a key in environment:
        EXPERIMENT_KEY = cfg.experiment_key

        # First, let's see if we continue or start fresh:
        CONTINUE_RUN = cfg.resume
        if EXPERIMENT_KEY and CONTINUE_RUN:
            # There is one, but the experiment might not exist yet:
            api = comet_ml.API()  # Assumes API key is set in config/env
            try:
                api_experiment = api.get_experiment_by_id(EXPERIMENT_KEY)
            except Exception:
                api_experiment = None
            if api_experiment is not None:
                CONTINUE_RUN = True
                # We can get the last details logged here, if logged:
                # step = int(api_experiment.get_parameters_summary("batch")["valueCurrent"])
                # epoch = int(api_experiment.get_parameters_summary("epochs")["valueCurrent"])

        if CONTINUE_RUN:
            # 1. Recreate the state of ML system before creating experiment
            # otherwise it could try to log params, graph, etc. again
            # ...
            # 2. Setup the existing experiment to carry on:
            logger = comet_ml.ExistingExperiment(
                previous_experiment=EXPERIMENT_KEY,
                log_env_details=True,  # to continue env logging
                log_env_gpu=True,  # to continue GPU logging
                log_env_cpu=True,  # to continue CPU logging
                auto_histogram_weight_logging=True,
                auto_histogram_gradient_logging=True,
                auto_histogram_activation_logging=True,
            )
            # Retrieved from above APIExperiment
            # self.logger.set_epoch(epoch)

        else:
            # 1. Create the experiment first
            #    This will use the COMET_EXPERIMENT_KEY if defined in env.
            #    Otherwise, you could manually set it here. If you don't
            #    set COMET_EXPERIMENT_KEY, the experiment will get a
            #    random key!
            if cfg.online:
                logger = comet_ml.Experiment(
                    disabled=cfg.disabled,
                    project_name=cfg.project_name,
                    workspace=cfg.workspace,
                    auto_histogram_weight_logging=True,
                    auto_histogram_gradient_logging=True,
                    auto_histogram_activation_logging=True,
                )
                logger.set_name(cfg.experiment_name)
                logger.add_tags(cfg.tags.split())
                logger.log_parameters(self.cfg)
            else:
                logger = comet_ml.OfflineExperiment(
                    disabled=cfg.disabled,
                    project_name=cfg.project_name,
                    workspace=cfg.workspace,
                    offline_directory=cfg.offline_directory,
                    auto_histogram_weight_logging=True,
                )
                logger.set_name(cfg.experiment_name)
                logger.add_tags(cfg.tags.split())
                logger.log_parameters(self.cfg)

        return logger

    def save(self, name=None):
        model = self.model
        if isinstance(self.model, torch.nn.DataParallel):
            model = model.module

        checkpoint = {
            "time": str(datetime.now()),
            "epoch": self.epoch,
            "iteration": self.iteration,
            "model": model.state_dict(),
            "model_name": type(model).__name__,
            "optimizer": self.optimizer.state_dict(),
            "optimizer_name": type(self.optimizer).__name__,
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "best": self.best,
            "e_loss": self.e_loss,
        }

        if name is None and self.epoch >= self.cfg.train_params.swa_start:
            save_name = self.cfg.directory.model_name + \
                str(self.epoch) + "-swa"
            checkpoint["model-swa"] = self.swa_model.state_dict()

        elif name is None:
            save_name = self.cfg.directory.model_name + str(self.epoch)

        else:
            save_name = name

        if self.e_loss[-1] < self.best:
            self.best = self.e_loss[-1]
            checkpoint["best"] = self.best
            save_checkpoint(checkpoint, True,
                            self.cfg.directory.save, save_name)
        else:
            save_checkpoint(checkpoint, False,
                            self.cfg.directory.save, save_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', default='conf/config', type=str)
    args = parser.parse_args()
    cfg_path = args.conf
    learner = Learner(cfg_path)
    learner.train()
