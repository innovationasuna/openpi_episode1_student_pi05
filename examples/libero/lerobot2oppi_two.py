#!/usr/bin/env python

"""Script to convert a LeRobot dataset with angles in radians to degrees format.

This script loads an existing LeRobot dataset where angles are stored in radians (0-2π),
creates a new dataset with the same structure, and converts all angle values to degrees (0-360)
while preserving all other data.

The conversion is done by multiplying each angle value by 180/π, which transforms:
- Input range: 0 to 2π radians
- Output range: 0 to 360 degrees

特殊处理：
- 对于第7个关节（夹爪），当use_radian=True时，其值被映射到[0,1]范围而不是真正的弧度值
- 脚本会将夹爪值从[0,1]范围转换回角度范围[20, 110]度

Example usage: 
python convert_radians_to_degrees.py --source_repo_id=enpeicv/move_fruit_radians --target_repo_id=enpeicv/move_fruit_degrees --source_dataset_root=/path/to/source/dataset --push_to_hub=False
"""

import math
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import tqdm
import tyro
import torch
from torchvision.utils import save_image
import torchvision.transforms as T


from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset, LeRobotDatasetMetadata


def radians_to_degrees(radians):
    """
    Convert angles from radians to degrees.
    
    Args:
        radians: Tensor or array of angles in radians
        
    Returns:
        Tensor or array of angles in degrees
    """
    return radians * (180.0 / math.pi)


def convert_dataset(source_repo_id: str, target_repo_id: str, *, push_to_hub: bool = False, max_episodes: int = None, output_path: str = None, source_dataset_root: str = None):
    """Convert a dataset from radians to degrees.
    
    Args:
        source_repo_id: The repository ID of the source dataset (with angles in radians)
        target_repo_id: The repository ID for the new dataset (with angles in degrees)
        push_to_hub: Whether to push the new dataset to the Hugging Face Hub
        max_episodes: Maximum number of episodes to convert. If None, all episodes are converted.
        output_path: Custom output path for the dataset. If None, uses HF_LEROBOT_HOME/target_repo_id.
        source_dataset_root: Custom root path for the source dataset. If None, uses HF_LEROBOT_HOME/source_repo_id.
    
    注意：
        对于第7个关节（夹爪），当use_radian=True时，其值被映射到[0,1]范围而不是真正的弧度值。
        本函数会将夹爪值从[0,1]范围转换回角度范围[20, 110]度。
    """
    # Clean up any existing dataset in the output directory
    if output_path is None:
        dataset_path = HF_LEROBOT_HOME / target_repo_id
    else:
        dataset_path = Path(output_path)
    
    if dataset_path.exists():
        shutil.rmtree(dataset_path)
    
    # Load metadata from the source dataset
    source_meta = LeRobotDatasetMetadata(source_repo_id, root=source_dataset_root)
    print(f"Loaded metadata from {source_repo_id}")
    print(f"Total episodes: {source_meta.total_episodes}")
    print(f"Total frames: {source_meta.total_frames}")
    
    # Load the source dataset
    source_dataset = LeRobotDataset(source_repo_id, root=source_dataset_root)
    print(f"Loaded source dataset with {source_dataset.num_frames} frames")

    num_episodes = min(source_meta.total_episodes, max_episodes) if max_episodes else source_meta.total_episodes
    print(f"Converting {num_episodes} episodes out of {source_meta.total_episodes}")
    
    # ----------
    # 这里续写，创建新的数据集
    dataset = LeRobotDataset.create(
        repo_id=target_repo_id,
        root=dataset_path,
        robot_type="enpei_episode1",
        fps=30,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image_left": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image_right": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (14,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (14,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )



    # ----------
    # Process each episode
    for episode_idx in tqdm.tqdm(range(num_episodes)):
        # Get the frame indices for this episode
        from_idx = source_dataset.episode_data_index["from"][episode_idx].item()
        to_idx = source_dataset.episode_data_index["to"][episode_idx].item()
        
        # Get the task for this episode
        episode_info = source_dataset.meta.episodes[episode_idx]
        task = episode_info["tasks"][0] if "tasks" in episode_info and episode_info["tasks"] else None
        
        # Process each frame in the episode
        for frame_idx in range(from_idx, to_idx):
            # Get the frame data
            frame = source_dataset[frame_idx]
            # 原始数据
            # torch.Size([3, 480, 640])
            # torch.Size([3, 480, 640])
            # torch.Size([7])
            # torch.Size([7])
            image_fixed = frame["observation.images.fixed"]  # 原始形状 [3, 480, 640]
            image_handeye_left = frame["observation.images.handeye_left"]  # 原始形状 [3, 480, 640]
            image_handeye_right = frame["observation.images.handeye_right"]  # 原始形状 [3, 480, 640]
            state = frame["observation.state"]
            action = frame["action"]
            
            # 转换图像格式从 [3, 480, 640] 到 [256, 256, 3]
            # 首先使用 torch 的 resize 操作调整大小
            # 调整大小
            resize_transform = T.Resize((256, 256))
            image_fixed_resized = resize_transform(image_fixed)
            image_handeye_left_resized = resize_transform(image_handeye_left)
            image_handeye_right_resized = resize_transform(image_handeye_right)
            
            # 手动转换通道顺序从 [C, H, W] 到 [H, W, C]
            image_fixed = image_fixed_resized.permute(1, 2, 0)
            image_handeye_left = image_handeye_left_resized.permute(1, 2, 0)
            image_handeye_right = image_handeye_right_resized.permute(1, 2, 0)

            # print shape
            # print(image_fixed.shape)
            # print(image_handeye.shape)
            # print(state.shape)
            # print(action.shape)
            


            dataset.add_frame(
                    {
                        "image": image_fixed,
                        "wrist_image_left": image_handeye_left,
                        "wrist_image_right": image_handeye_right,
                        "state": state,
                        "actions": action,
                        "task": task,
                    }
                )
            
        dataset.save_episode() 


    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["enpeicv", "episode1", "6dof"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )
    


def main(source_repo_id: str = "enpeicv/move_fruit_radians", 
         target_repo_id: str = "enpeicv/move_fruit_degrees", 
         *, push_to_hub: bool = False,
         max_episodes: int = None,
         output_path: str = None,
         source_dataset_root: str = None):
    """Main function to convert a dataset from radians to degrees.
    
    Args:
        source_repo_id: The repository ID of the source dataset (with angles in radians)
        target_repo_id: The repository ID for the new dataset (with angles in degrees)
        push_to_hub: Whether to push the new dataset to the Hugging Face Hub
        max_episodes: Maximum number of episodes to convert. If None, all episodes are converted.
        output_path: Custom output path for the dataset. If None, uses HF_LEROBOT_HOME/target_repo_id.
        source_dataset_root: Custom root path for the source dataset. If None, uses HF_LEROBOT_HOME/source_repo_id.
    
    注意：
        对于第7个关节（夹爪），当use_radian=True时，其值被映射到[0,1]范围而不是真正的弧度值。
        本函数会将夹爪值从[0,1]范围转换回角度范围[20, 110]度。
    """
    convert_dataset(source_repo_id, target_repo_id, push_to_hub=push_to_hub, max_episodes=max_episodes, output_path=output_path, source_dataset_root=source_dataset_root)


if __name__ == "__main__":
    tyro.cli(main)


# python ./examples/libero/lerobot2oppi.py \
# --source-repo_id=enpeicv/move_fruit_0805 \
# --target-repo_id=enpeicv/move_fruit_openpi \
# --output-path=./enpei_dataset/move_fruit_openpi \
# --source-dataset-root=/root/autodl-tmp/lerobot-main/enpei_dataset/move_fruit_0805 \
# --max-episodes=3 \
# --push-to-hub
#
# Note: For boolean flags, use --push-to-hub to enable or --no-push-to-hub to disable
# Do NOT use --push-to-hub=true or --push-to-hub=false
