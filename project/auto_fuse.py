#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/Auto_Gait_Fusion_PyTorch/project/auto_fuse.py
Project: /workspace/Auto_Gait_Fusion_PyTorch/project
Created Date: Monday April 8th 2024
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Monday April 8th 2024 5:11:12 am
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
from functools import partial

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize
from pytorchvideo.transforms import (
    UniformTemporalSubsample,
    Div255,
)

from torchvision.io import read_video, write_png
from train import GaitCycleLightningModule

logger = logging.getLogger(__name__)

disease_to_num_mapping_Dict: Dict = {
    2: {"ASD": 0, "non-ASD": 1},
    3: {"ASD": 0, "DHS": 1, "LCS_HipOA": 2},
    4: {"ASD": 0, "DHS": 1, "LCS_HipOA": 2, "normal": 3},
}


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

        # FIXME: maybe here do not -1 for upper limit.
        for i in range(1, len(gait_cycle_index) - 1, 2):
            ans_list.append(
                video_tensor[gait_cycle_index[i] : gait_cycle_index[i + 1], ...]
            )
            use_idx.append(gait_cycle_index[i])

    logging.info(f"used split gait cycle index: {use_idx}")
    return ans_list, use_idx  # needed gait cycle video tensor


class PrePredictGaitVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        labeled_video_paths: list[Tuple[str, Optional[dict]]],
        transform: Optional[Callable[[dict], Any]] = None,
    ) -> None:

        super().__init__()

        self._transform = transform
        self._labeled_videos = labeled_video_paths

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

        # should return the new frame, named temporal mix.
        # defined_vframes = self._auto_fuse(vframes, gait_cycle_index, label)

        first_phase, first_phase_idx = split_gait_cycle(vframes, gait_cycle_index, 0)
        second_phase, second_phase_idx = split_gait_cycle(vframes, gait_cycle_index, 1)

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

        if self._transform is not None:
            video_t_list = []
            for video_t in first_phase:
                transformed_img = self._transform(video_t.permute(1, 0, 2, 3))
                video_t_list.append(transformed_img)

            def_first_vframes = torch.stack(video_t_list, dim=0)

            video_t_list = []
            for video_t in second_phase:
                transformed_img = self._transform(video_t.permute(1, 0, 2, 3))
                video_t_list.append(transformed_img)

            def_second_vframes = torch.stack(video_t_list, dim=0)

        defined_vframes = torch.stack(
            [def_first_vframes, def_second_vframes], dim=5
        )  # b, c, t, h, w, SWT

        sample_info_dict = {
            "video": defined_vframes,  # b, c, t, h, w, SWT
            "label": label,
            "disease": disease,
            "video_name": video_name,
            "video_index": index,
            "gait_cycle_index": gait_cycle_index,
            "bbox_none_index": bbox_none_index,
        }

        return sample_info_dict


def collate_fn(batch, class_num):

    batch_label = []
    batch_video = []
    batch_v_name = {}
    # mapping label

    for i in batch:
        b, c, t, h, w, phase = i["video"].shape
        disease = i["disease"]

        batch_video.append(i["video"])
        batch_v_name[i["video_name"]] = b

        for _ in range(b):

            if disease in disease_to_num_mapping_Dict[class_num].keys():

                batch_label.append(disease_to_num_mapping_Dict[class_num][disease])
            else:
                batch_label.append(disease_to_num_mapping_Dict[class_num]["non-ASD"])

    return {
        "video": torch.cat(batch_video, dim=0),
        "label": torch.tensor(batch_label),
        "video_name": batch_v_name,
    }


def pre_predict(hparams, dataset_idx, fold):

    gpu = f"cuda:{hparams.train.gpu_num}"

    st_model = GaitCycleLightningModule(hparams).load_from_checkpoint(
        hparams.gait_model.stance_path
    )
    sw_model = GaitCycleLightningModule(hparams).load_from_checkpoint(
        hparams.gait_model.swing_path
    )

    uniform_temporal_subsample_num = hparams.train.uniform_temporal_subsample_num
    _IMG_SIZE = hparams.data.img_size

    transform = Compose(
        [
            UniformTemporalSubsample(uniform_temporal_subsample_num),
            Div255(),
            Resize(size=[_IMG_SIZE, _IMG_SIZE]),
        ]
    )

    dataset = PrePredictGaitVideoDataset(
        labeled_video_paths=dataset_idx[0],
        transform=transform,
    )
    batch_size = hparams.gait_model.batch_size
    n_works = hparams.data.num_workers
    class_num = hparams.model.model_class_num

    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=n_works, collate_fn=partial(collate_fn, class_num=class_num)
    )

    res_mapp = {}

    for i, batch in enumerate(data_loader):

        phase_video = batch["video"]
        b, c, t, h, w, phase = phase_video.shape

        stance_frames = phase_video[..., 0].to(gpu)
        swing_frames = phase_video[..., 1].to(gpu)

        phase_label = batch["label"]
        phase_info = batch["video_name"]

        st_model.to(gpu)
        sw_model.to(gpu)

        st_model.eval()
        sw_model.eval()

        with torch.no_grad():
            st_pred = st_model(stance_frames)
            sw_pred = sw_model(swing_frames)

        st_pred_softmax = torch.nn.functional.softmax(st_pred, dim=1)
        sw_pred_softmax = torch.nn.functional.softmax(sw_pred, dim=1)

        one_video_res = []
        # compare the phase score
        for i in range(b):
            if st_pred_softmax[i][phase_label[i]] > sw_pred_softmax[i][phase_label[i]]:
                one_video_res.append(0)
            else:
                one_video_res.append(1)

        idx = 0
        for k, v in phase_info.items():
            res_mapp[k] = one_video_res[idx : idx + v]
            idx = v

    map_file_save_path = hparams.train.log_path + f'/fold{fold}_predict_mapping.json'

    with open(map_file_save_path, 'w') as f:
        json.dump(res_mapp, f, indent=4)

    return res_mapp
