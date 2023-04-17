[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains PyTorch training code for [Verti-Wheelers](https://cs.gmu.edu/~xiao/Research/Verti-Wheelers/).

## Installation
Main libraries:
* [PyTorch](pytorch.org/): as the main ML framework
* [Comet.ml](https://www.comet.ml): tracking code, logging experiments
* [OmegaConf](https://omegaconf.readthedocs.io/en/latest/): for managing configuration files


First create a virtual env for the project. 
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Then install the latest version of PyTorch from the [official site](pytorch.org/). Finally, run the following:
```bash
pip install -r requirements
```
To set up Comet.Ml follow the [official documentations](https://www.comet.ml/docs/).

## Dataset
To download the dataset please follow the instructions [here](https://cs.gmu.edu/~xiao/Research/Verti-Wheelers/).

## Run
To run parser, first edit the config file in [`conf/parser_config`](verti_wheelers/conf/parser_config.yaml) directory. Then run:
```bash
./run.sh parse
```

To run the training pipeline, make sure everything in the [config file](verti_wheelers/conf/config.yaml) is correct, then run:
```bash
./run.sh train
```

## Deployment
To deploy the trained model, please follow the instructions [here](https://github.com/RobotiXX/Verti-Wheelers).

## Reference
If you find this repo to be useful in your research, please consider citing our work:
```
@article{datar2023a,
  title={Toward Wheeled Mobility on Vertically Challenging Terrain: Platforms, Datasets, and Algorithms},
  author={Datar, Aniket and Pan, Chenhui and Nazeri, Mohammad and Xiao, Xuesu},
  journal={arXiv preprint arXiv:2303.00998},
  year={2023}
}
```