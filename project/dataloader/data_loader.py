"""
File: data_loader.py
Project: dataloader
Created Date: 2023-10-19 02:24:47
Author: chenkaixu
-----
Comment:
A pytorch lightning data module based dataloader, for train/val/test dataset prepare.
Have a good code time!
-----
Last Modified: 2023-10-29 12:18:59
Modified By: chenkaixu
-----
HISTORY:
Date 	By 	Comments
------------------------------------------------

04-04-2024	Kaixu Chen	when use temporal mix, need keep same data process method for train/val dataset.

25-03-2024	Kaixu Chen	change batch size for train/val dataloader. Now set bs=1 for gait cycle dataset, set bs=32 for default dataset (without gait cycle).
Because in experiment, I found bs=1 will have large loss when train. Maybe also need gait cycle datset in val/test dataloader?

22-03-2024	Kaixu Chen	add different class num mapping dict. In collate_fn, re-mapping the label from .json disease key.

"""

# %%

import logging, json, shutil, os
from pathlib import Path

from torchvision.transforms import (
    Compose,
    Resize,
)
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    UniformTemporalSubsample,
    Div255,
)

from typing import Any, Callable, Dict, Optional, Type
from pytorch_lightning import LightningDataModule

import torch
from torch.utils.data import DataLoader
from pytorchvideo.data import make_clip_sampler
from pytorchvideo.data.labeled_video_dataset import labeled_video_dataset

from auto_fuse_dataset import labeled_gait_video_dataset

from project.train import GaitCycleLightningModule

disease_to_num_mapping_Dict: Dict = {
    2: {"ASD": 0, "non-ASD": 1},
    3: {"ASD": 0, "DHS": 1, "LCS_HipOA": 2},
    4: {"ASD": 0, "DHS": 1, "LCS_HipOA": 2, "normal": 3},
}

class WalkDataModule(LightningDataModule):
    def __init__(self, opt, dataset_idx: Dict = None):
        super().__init__()

        self._seg_path = opt.data.seg_data_path
        self._gait_seg_path = opt.data.gait_seg_data_path
        self.gait_cycle = opt.train.gait_cycle

        # ? 感觉batch size对最后的结果有影响，所以分开使用不同的batch size
        self._gait_cycle_batch_size = opt.data.gait_cycle_batch_size
        self._default_batch_size = opt.data.default_batch_size

        self._NUM_WORKERS = opt.data.num_workers
        self._IMG_SIZE = opt.data.img_size

        # frame rate
        self._CLIP_DURATION = opt.train.clip_duration
        self.uniform_temporal_subsample_num = opt.train.uniform_temporal_subsample_num

        # load geit model 
        self.stance_model = GaitCycleLightningModule(opt).load_from_checkpoint(opt.gait_model.stance_path)
        self.swing_model = GaitCycleLightningModule(opt).load_from_checkpoint(opt.gait_model.swing_path)

        # * this is the dataset idx, which include the train/val dataset idx.
        self._dataset_idx = dataset_idx
        self._class_num = opt.model.model_class_num

        self._auto_fuse = opt.train.auto_fuse

        self.mapping_transform = Compose(
            [
                UniformTemporalSubsample(self.uniform_temporal_subsample_num),
                Div255(),
                Resize(size=[self._IMG_SIZE, self._IMG_SIZE]),
            ]
        )

        self.val_video_transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            Div255(),
                            Resize(size=[self._IMG_SIZE, self._IMG_SIZE]),
                            UniformTemporalSubsample(
                                self.uniform_temporal_subsample_num
                            ),
                        ]
                    ),
                ),
            ]
        )

    def prepare_data(self) -> None:
        """here prepare the temp val data path,
        because the val dataset not use the gait cycle index,
        so we directly use the pytorchvideo API to load the video.
        AKA, use whole video to validate the model.
        """
        ...

    def setup(self, stage: Optional[str] = None) -> None:
        """
        assign tran, val, predict datasets for use in dataloaders

        Args:
            stage (Optional[str], optional): trainer.stage, in ('fit', 'validate', 'test', 'predict'). Defaults to None.
        """

        if stage in ("fit", None):
            # judge if use split the gait cycle, or use the whole video.
            if self.gait_cycle == -1:
                self.train_gait_dataset = labeled_video_dataset(
                    data_path=self._dataset_idx[2],
                    clip_sampler=make_clip_sampler("uniform", self._CLIP_DURATION),
                    transform=self.train_video_transform,
                )

            else:
                # * labeled dataset, where first define the giat cycle, and from .json file to load the video.
                # * here only need dataset idx, mean first split the dataset, and then load the video.
                self.train_gait_dataset = labeled_gait_video_dataset(
                    gait_cycle=self.gait_cycle,
                    dataset_idx=self._dataset_idx[0],  # [train, val]
                    transform=self.mapping_transform,
                    gait_model = [self.stance_model, self.swing_model],
                    auto_fuse=self._auto_fuse,
                )

        if stage in ("fit", "validate", None):
            self.val_gait_dataset = labeled_video_dataset(
                data_path=self._dataset_idx[3],
                clip_sampler=make_clip_sampler("uniform", self._CLIP_DURATION),
                transform=self.val_video_transform,
            )

        if stage in ("test", None):
            self.test_gait_dataset = labeled_video_dataset(
                data_path=self._dataset_idx[3],
                clip_sampler=make_clip_sampler("uniform", self._CLIP_DURATION),
                transform=self.val_video_transform,
            )

    def collate_fn(self, batch):
        """this function process the batch data, and return the batch data.

        Args:
            batch (list): the batch from the dataset.
            The batch include the one patient info from the json file.
            Here we only cat the one patient video tensor, and label tensor.

        Returns:
            dict: {video: torch.tensor, label: torch.tensor, info: list}
        """

        batch_label = []
        batch_video = []

        # mapping label

        for i in batch:
            # logging.info(i['video'].shape)
            gait_num, c, t, h, w = i["video"].shape
            disease = i["disease"]

            batch_video.append(i["video"])
            for _ in range(gait_num):

                if disease in disease_to_num_mapping_Dict[self._class_num].keys():

                    batch_label.append(
                        disease_to_num_mapping_Dict[self._class_num][disease]
                    )
                else:
                    # * if the disease not in the mapping dict, then set the label to non-ASD.
                    batch_label.append(
                        disease_to_num_mapping_Dict[self._class_num]["non-ASD"]
                    )

        # video, b, c, t, h, w, which include the video frame from sample info
        # label, b, which include the video frame from sample info
        # sample info, the raw sample info from dataset
        return {
            "video": torch.cat(batch_video, dim=0),
            "label": torch.tensor(batch_label),
            "info": batch,
        }

    def train_dataloader(self) -> DataLoader:
        """
        create the Walk train partition from the list of video labels
        in directory and subdirectory. Add transform that subsamples and
        normalizes the video before applying the scale, crop and flip augmentations.
        """
        if self.gait_cycle == -1:
            train_data_loader = DataLoader(
                self.train_gait_dataset,
                batch_size=self._default_batch_size,
                num_workers=self._NUM_WORKERS,
                pin_memory=True,
                shuffle=False,
                drop_last=True,
            )

        else:
            train_data_loader = DataLoader(
                self.train_gait_dataset,
                batch_size=self._gait_cycle_batch_size,
                num_workers=self._NUM_WORKERS,
                pin_memory=True,
                shuffle=True,
                drop_last=True,
                collate_fn=self.collate_fn,
            )

        return train_data_loader

    def val_dataloader(self) -> DataLoader:
        """
        create the Walk train partition from the list of video labels
        in directory and subdirectory. Add transform that subsamples and
        normalizes the video before applying the scale, crop and flip augmentations.
        """

        return DataLoader(
            self.val_gait_dataset,
            batch_size=self._default_batch_size,
            num_workers=self._NUM_WORKERS,
            pin_memory=True,
            shuffle=False,
            drop_last=True,
        )

    def test_dataloader(self) -> DataLoader:
        """
        create the Walk train partition from the list of video labels
        in directory and subdirectory. Add transform that subsamples and
        normalizes the video before applying the scale, crop and flip augmentations.
        """

        return DataLoader(
            self.test_gait_dataset,
            batch_size=self._default_batch_size,
            num_workers=self._NUM_WORKERS,
            pin_memory=True,
            shuffle=False,
            drop_last=True,
        )
