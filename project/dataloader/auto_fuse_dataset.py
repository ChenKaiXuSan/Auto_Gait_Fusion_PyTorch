#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/Auto_Gait_Fusion_PyTorch/project/dataloader/auto_fuse_dataset.py
Project: /workspace/Auto_Gait_Fusion_PyTorch/project/dataloader
Created Date: Saturday April 6th 2024
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Saturday April 6th 2024 10:28:03 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2024 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

from __future__ import annotations

import logging, sys, json

sys.path.append("/workspace/Auto_Gait_Fusion_PyTorch")

from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Type

import torch

from torchvision.io import read_video, write_png
from pytorchvideo.transforms.functional import uniform_temporal_subsample

logger = logging.getLogger(__name__)


def split_gait_cycle(
    video_tensor: torch.Tensor, gait_cycle_index: list, gait_cycle: int
):
    # 也就是说需要根据给定的参数，能够区分不同的步行周期
    # 例如， 2分的话，需要能区分前后， 4分的话，需要能区分前后，中间，等

    use_idx = []
    ans_list = []
    if gait_cycle == 0 or len(gait_cycle_index) == 2:
        for i in range(0, len(gait_cycle_index) - 1, 2):
            ans_list.append(
                video_tensor[gait_cycle_index[i] : gait_cycle_index[i + 1], ...]
            )
            use_idx.append(gait_cycle_index[i])

    elif gait_cycle == 1:
        # if len(gait_cycle_index) == 2:
        #     ans_list.append(video_tensor[gait_cycle_index[0]:gait_cycle_index[1], ...])
        #     use_idx.append(gait_cycle_index[0])
        #     print('the gait cycle index is less than 2, so use first gait cycle')

        # FIXME: maybe here do not -1 for upper limit.
        for i in range(1, len(gait_cycle_index) - 1, 2):
            ans_list.append(
                video_tensor[gait_cycle_index[i] : gait_cycle_index[i + 1], ...]
            )
            use_idx.append(gait_cycle_index[i])

    logging.info(f"used split gait cycle index: {use_idx}")
    return ans_list, use_idx  # needed gait cycle video tensor

class AutoFuse(object):

    def __init__(self, stance_model, swing_model, transform) -> None:
        self.st = stance_model
        self.sw = swing_model
        self._transform = transform

    def compare_phase(self, first_phase, second_phase, label):

        res_best_vframes = []

        for i in range(len(first_phase)):

            # move to functional
            if self._transform is not None:

                transformed_img = self._transform(first_phase[i].permute(1, 0, 2, 3))
                self.st.eval()
                with torch.no_grad():
                    st_pred = self.st(transformed_img.unsqueeze(0))
                st_pred_softmax = torch.nn.functional.softmax(st_pred, dim=1).squeeze()

                transformed_img = self._transform(second_phase[i].permute(1, 0, 2, 3))
                self.sw.eval()
                with torch.no_grad():
                    sw_pred = self.sw(transformed_img.unsqueeze(0))
                sw_pred_softmax = torch.nn.functional.softmax(sw_pred, dim=1).squeeze()

                # compare phase score
                if st_pred_softmax[label] > sw_pred_softmax[label]:
                    res_best_vframes.append(first_phase[i])
                else:
                    res_best_vframes.append(second_phase[i])

        return res_best_vframes

    def __call__(
        self, video_tensor: torch.Tensor, gait_cycle_index: list, label: list
    ) -> torch.Tensor:

        # * step1: first find the phase frames (pack) and phase index.
        first_phase, first_phase_idx = split_gait_cycle(
            video_tensor, gait_cycle_index, 0
        )
        second_phase, second_phase_idx = split_gait_cycle(
            video_tensor, gait_cycle_index, 1
        )

        # check if first phse and second phase have the same length
        if len(first_phase) > len(second_phase):
            second_phase.append(second_phase[-1])
            second_phase_idx.append(second_phase_idx[-1])
        elif len(first_phase) < len(second_phase):
            first_phase.append(first_phase[-1])
            first_phase_idx.append(first_phase_idx[-1])

        assert len(first_phase) == len(
            second_phase
        ), "first phase and second phase have different length"

        # * step2: compare the phase frame, find the best match
        best_vframes = self.compare_phase(first_phase, second_phase, label)

        # * step3: process on pack, fuse the frame

        return best_vframes


class LabeledGaitVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        gait_cycle: int,
        labeled_video_paths: list[Tuple[str, Optional[dict]]],
        transform: Optional[Callable[[dict], Any]] = None,
        auto_fuse: bool = False,
        stance_model: torch.nn.Module = None,
        swing_model: torch.nn.Module = None,
    ) -> None:
        super().__init__()

        self._transform = transform
        self._labeled_videos = labeled_video_paths
        self._gait_cycle = gait_cycle

        if auto_fuse:
            self._auto_fuse = AutoFuse(stance_model, swing_model, transform)

        else:
            self._temporal_mix = False

        self.stance_model = stance_model
        self.swing_model = swing_model

    def __len__(self):
        return len(self._labeled_videos)

    def __getitem__(self, index) -> Any:

        # load the video tensor from json file
        with open(self._labeled_videos[index]) as f:
            file_info_dict = json.load(f)

        # load video info from json file
        video_name = file_info_dict["video_name"]
        video_path = file_info_dict["video_path"]
        vframes, _, _ = read_video(video_path, output_format="TCHW")
        label = file_info_dict["label"]
        disease = file_info_dict["disease"]
        gait_cycle_index = file_info_dict["gait_cycle_index"]
        bbox_none_index = file_info_dict["none_index"]
        bbox = file_info_dict["bbox"]

        logging.info(f"video name: {video_name}, gait cycle index: {gait_cycle_index}")

        if self._auto_fuse:
            # should return the new frame, named temporal mix.
            defined_vframes = self._auto_fuse(vframes, gait_cycle_index, label)
        else:
            # split gait by gait cycle index, first phase or second phase
            defined_vframes, used_gait_idx = split_gait_cycle(
                vframes, gait_cycle_index, self._gait_cycle
            )

        sample_info_dict = {
            "video": defined_vframes,
            "label": label,
            "disease": disease,
            "video_name": video_name,
            "video_index": index,
            "gait_cycle_index": gait_cycle_index,
            "bbox_none_index": bbox_none_index,
        }

        # move to functional
        if self._transform is not None:
            video_t_list = []
            for video_t in defined_vframes:
                transformed_img = self._transform(video_t.permute(1, 0, 2, 3))
                video_t_list.append(transformed_img)

            sample_info_dict["video"] = torch.stack(video_t_list, dim=0)  # c, t, h, w
        else:
            print("no transform")

        return sample_info_dict


def labeled_gait_video_dataset(
    gait_cycle: int,
    transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    dataset_idx: Dict = None,
    auto_fuse: bool = False,
    gait_model: List[Type[Union[torch.nn.Module, Any]]] = None,
) -> LabeledGaitVideoDataset:

    stance_model = gait_model[0]
    swing_model = gait_model[1]

    dataset = LabeledGaitVideoDataset(
        gait_cycle, dataset_idx, transform, auto_fuse, stance_model, swing_model
    )

    return dataset
