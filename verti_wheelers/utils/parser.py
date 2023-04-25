"""
Parses bag files. Courtesy of Weisen Zhao for implementing the parser.
"""
from pathlib import Path
import pickle
import argparse

import rosbag
import cv2
import numpy as np
from tqdm import tqdm
from scipy.signal import savgol_filter

from utils.helpers import get_conf


def parse_bags(cfg) -> None:
    id = 0
    bag_files = Path(cfg.bags_dir).resolve()
    save_dir = Path(cfg.save_dir).resolve()
    bag_files = [x for x in bag_files.iterdir() if x.suffix == ".bag"]
    save_depth = save_dir / "depth"
    Path.mkdir(save_depth, exist_ok=True)
    save_labels = str(save_dir / "labels.pickle")
    field_names = {"rpm": [], "opti_flow": [], "cmd_vel": [], "img_address": []}
    # csvfile = open(save_labels, "w", newline="\n")
    # writer = csv.DictWriter(csvfile, fieldnames=field_names)
    # writer.writeheader()
    for bag in tqdm(bag_files, desc="Bags processed"):
        bag = rosbag.Bag(bag)
        depthMsg = bag.read_messages(topics=cfg.topics.depth)
        rpmMsg = bag.read_messages(topics=cfg.topics.rpm)
        groundSpeedMsg = bag.read_messages(topics=cfg.topics.optiflow)
        labelMsg = bag.read_messages(topics=cfg.topics.cmd_vel)
        depthMsgList = list(depthMsg)
        rpmMsgList = list(rpmMsg)
        groundSpeedMsgList = list(groundSpeedMsg)
        labelMsgList = list(labelMsg)
        for k in range(len(depthMsgList)):
            currentDepthTime = depthMsgList[k].message.header.stamp
            depthImageData = depthMsgList[k].message.data
            cloestMsgToDepthTime = rpmMsgList[0]
            depthToAdd = np.frombuffer(depthImageData, np.uint8)
            rpmToAdd = None
            groundSpeedToAdd = None
            for j in range(len(rpmMsgList)):
                currentRpmTime = rpmMsgList[j].timestamp
                if currentRpmTime <= currentDepthTime:
                    cloestMsgToDepthTime = rpmMsgList[j]
                    if j == len(rpmMsgList) - 1:
                        rpmToAdd = rpmMsgList[j].message.data
                else:
                    rpmToAdd = cloestMsgToDepthTime.message.data
                    break
            for j in range(len(groundSpeedMsgList)):
                currentGroundSpeedTime = groundSpeedMsgList[j].timestamp
                if currentGroundSpeedTime > currentDepthTime:
                    groundSpeedToAdd = groundSpeedMsgList[j].message.data
                    break
            for j in range(len(labelMsgList)):
                currentlabelTime = labelMsgList[j].timestamp
                if currentlabelTime > currentDepthTime:
                    lableToAdd = labelMsgList[j].message.data
                    break

            id += 1
            img = cv2.imdecode(depthToAdd, cv2.IMREAD_ANYDEPTH)
            img = img.astype(float)
            save_path = str((save_depth / f"{id}.jpeg").resolve())
            cv2.imwrite(save_path, img)
            # writer.writerow(
            #     {
            #         "id": id,
            #         "rpm": rpmToAdd,
            #         "opti_flow": groundSpeedToAdd,
            #         "cmd_vel": lableToAdd,
            #         "img_address": save_path,
            #     }
            # )
            field_names['rpm'].append(rpmToAdd)
            field_names['opti_flow'].append(groundSpeedToAdd)
            field_names['cmd_vel'].append(lableToAdd[:2])
            field_names['img_address'].append(save_path)
    # csvfile.close()
    with open(save_labels, "wb") as f:
        pickle.dump(field_names, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", default="../conf/parser_config", type=str)
    args = parser.parse_args()
    cfg_dir = args.conf
    cfg = get_conf(cfg_dir)
    parse_bags(cfg)
