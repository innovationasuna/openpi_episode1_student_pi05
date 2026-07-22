#!/usr/bin/env python

"""Repack a single-arm Enpei LeRobot dataset for the OpenPI LIBERO adapter.

State and action values are copied without unit conversion. The first six joints in the source
dataset must already use radians. Camera images are resized with aspect-ratio-preserving black
padding to match online inference.
"""

import shutil
from pathlib import Path

import numpy as np
import torch
import tqdm
import tyro

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset, LeRobotDatasetMetadata
from openpi_client import image_tools


IMAGE_SIZE = 224


def prepare_image(image: torch.Tensor) -> np.ndarray:
    """Convert a LeRobot CHW image to the padded HWC uint8 format used online."""
    image = image.detach().cpu().permute(1, 2, 0).numpy()
    image = image_tools.convert_to_uint8(image)
    return image_tools.resize_with_pad(image, IMAGE_SIZE, IMAGE_SIZE)


def convert_dataset(source_repo_id: str, target_repo_id: str, *, push_to_hub: bool = False, max_episodes: int = None, output_path: str = None, source_dataset_root: str = None):
    """Repack a single-arm dataset without changing state or action values."""
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
                "shape": (IMAGE_SIZE, IMAGE_SIZE, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (IMAGE_SIZE, IMAGE_SIZE, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
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
            image_handeye = frame["observation.images.handeye"]  # 原始形状 [3, 480, 640]
            state = frame["observation.state"]
            action = frame["action"]
            
            # 与在线客户端保持一致：CHW -> HWC uint8，再等比例缩放并补黑边到 224x224。
            image_fixed = prepare_image(image_fixed)
            image_handeye = prepare_image(image_handeye)

            # print shape
            # print(image_fixed.shape)
            # print(image_handeye.shape)
            # print(state.shape)
            # print(action.shape)
            


            dataset.add_frame(
                    {
                        "image": image_fixed,
                        "wrist_image": image_handeye,
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
    


def main(source_repo_id: str = "enpeicv/demo_move_block",
         target_repo_id: str = "enpeicv/demo_move_block_openpi",
         *, push_to_hub: bool = False,
         max_episodes: int = None,
         output_path: str = None,
         source_dataset_root: str = None):
    """CLI entry point for single-arm dataset repacking."""
    convert_dataset(source_repo_id, target_repo_id, push_to_hub=push_to_hub, max_episodes=max_episodes, output_path=output_path, source_dataset_root=source_dataset_root)


if __name__ == "__main__":
    tyro.cli(main)
