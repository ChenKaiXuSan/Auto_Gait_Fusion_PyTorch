#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/skeleton/project/dataloader/gait_video_dataset.py
Project: /workspace/skeleton/project/dataloader
Created Date: Monday December 11th 2023
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Monday December 11th 2023 11:26:34 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2023 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------

27-03-2024	Kaixu Chen	make temporal mix a separate class.
"""

from __future__ import annotations

import logging, sys, json

sys.path.append("/workspace/skeleton")

from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Type

import torch

from torchvision.io import read_video, write_png
from pytorchvideo.transforms.functional import uniform_temporal_subsample

logger = logging.getLogger(__name__)

def split_gait_cycle(video_tensor: torch.Tensor, gait_cycle_index: list, gait_cycle: int):
    # 也就是说需要根据给定的参数，能够区分不同的步行周期
    # 例如， 2分的话，需要能区分前后， 4分的话，需要能区分前后，中间，等

    use_idx = []
    ans_list = []
    if gait_cycle == 0 or len(gait_cycle_index) == 2 :
        for i in range(0, len(gait_cycle_index)-1, 2):
            ans_list.append(video_tensor[gait_cycle_index[i]:gait_cycle_index[i+1], ...])
            use_idx.append(gait_cycle_index[i])

    elif gait_cycle == 1:
        # if len(gait_cycle_index) == 2:
        #     ans_list.append(video_tensor[gait_cycle_index[0]:gait_cycle_index[1], ...])
        #     use_idx.append(gait_cycle_index[0])
        #     print('the gait cycle index is less than 2, so use first gait cycle')

        # FIXME: maybe here do not -1 for upper limit.
        for i in range(1, len(gait_cycle_index)-1, 2):
            ans_list.append(video_tensor[gait_cycle_index[i]:gait_cycle_index[i+1], ...])
            use_idx.append(gait_cycle_index[i])

    logging.info(f"used split gait cycle index: {use_idx}")
    return ans_list, use_idx # needed gait cycle video tensor

class TemporalMix(object):
    """
    This class is temporal mix, which is used to mix the first phase and second phase of gait cycle.
    """    

    def __init__(self) -> None:
        pass

    @staticmethod
    def process_phase(phase_frame: List[torch.Tensor], phase_idx: List[int], bbox: List[torch.Tensor]) -> List[torch.Tensor]:
        """Crop the human area with bbox, and normalize them with max width.

        Args:
            phase_frame (List[torch.Tensor]): procedded frame by gait index
            phase_idx (List[int]): gait cycle index
            bbox (List[torch.Tensor]): _description_

        Returns:
            List[torch.Tensor]: _description_
        """
        # find the max width of phase and crop them.
        cropped_frame_list: List[torch.Tensor] = []

        for i in range(len(phase_idx)):

            one_pack_frames = phase_frame[i] # frame pack
            one_pack_start_idx = phase_idx[i]
            b, c, h, w = one_pack_frames.shape # b is one frame pack size

            stored_x_max = float("-inf")
            stored_xmax = 0
            stored_xmin = 0
            
            one_pack_frames_list: List[torch.Tensor] = []

            # * step1: find the max width and max height for one frame pack.
            for k in range(b):

                frame_bbox = bbox[one_pack_start_idx+k]
                x, y, w, h = frame_bbox
                xmin = int(x - w / 2)
                xmax = int(x + w / 2)

                if xmax - xmin > stored_x_max:
                    stored_x_max = xmax - xmin
                    stored_xmax = xmax
                    stored_xmin = xmin

            # * step2: crop human area with bbox, and normalized with max width
            for k in range(b):

                frame_bbox = bbox[i+k]
                x, y, w, h = frame_bbox
                
                frame = one_pack_frames[k]
                cropped_one_frame_human = frame[:, :, stored_xmin:stored_xmax]
                one_pack_frames_list.append(cropped_one_frame_human)

                # write_png(input=cropped_one_frame_human, filename=f'/workspace/skeleton/logs/img/test{k}.png')

            # * step3: stack the cropped frame, for next step to fuse them
            cropped_frame_list.append(torch.stack(one_pack_frames_list, dim=0)) # b, c, h, w

        # shape check 
        assert len(cropped_frame_list) == len(phase_frame) == len(phase_idx), "frame pack length is not equal"
        for i in range(len(phase_frame)):
            assert cropped_frame_list[i].size()[0] == phase_frame[i].size()[0], f"the {i} frame pack size is not equal"

        return cropped_frame_list

    @staticmethod
    def fuse_frames(processed_first_phase: List[torch.Tensor], processed_second_phase: List[torch.Tensor]) -> torch.Tensor:

        assert len(processed_first_phase) == len(processed_second_phase), "first phase and second phase have different length"

        res_fused_frames: List[torch.Tensor] = []
        # TODO: fuse the frame with different phase
        for pack in range(len(processed_first_phase)):

            uniform_first_phase = uniform_temporal_subsample(processed_first_phase[pack], 8, temporal_dim=-4) # t, c, h, w
            uniform_second_phase = uniform_temporal_subsample(processed_second_phase[pack], 8, temporal_dim=-4)

            # fuse width dim 
            fused_frames = torch.cat([uniform_first_phase, uniform_second_phase], dim=3)   

            # write the fused frame to png
            for i in range(fused_frames.size()[0]): 
                write_png(input=fused_frames[i], filename=f'/workspace/skeleton/logs/img/fused{i}.png')

            res_fused_frames.append(fused_frames)
            
        return res_fused_frames
        
    def __call__(self, video_tensor: torch.Tensor, gait_cycle_index: list, bbox: List[torch.Tensor]) -> torch.Tensor:

        # * step1: first find the phase frames (pack) and phase index.
        first_phase, first_phase_idx = split_gait_cycle(video_tensor, gait_cycle_index, 0)
        second_phase, second_phase_idx = split_gait_cycle(video_tensor, gait_cycle_index, 1)

        # check if first phse and second phase have the same length
        if len(first_phase) > len(second_phase):
            second_phase.append(second_phase[-1])
            second_phase_idx.append(second_phase_idx[-1])
        elif len(first_phase) < len(second_phase):
            first_phase.append(first_phase[-1])
            first_phase_idx.append(first_phase_idx[-1])
            
        assert len(first_phase) == len(second_phase), "first phase and second phase have different length"

        # * step2: process the phase frame, crop and normalize them
        processed_first_phase = self.process_phase(first_phase, first_phase_idx, bbox)
        processed_second_phase = self.process_phase(second_phase, second_phase_idx, bbox)

        # * step3: process on pack, fuse the frame
        fused_vframes = self.fuse_frames(processed_first_phase, processed_second_phase)

        return fused_vframes

class LabeledGaitVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        gait_cycle: int,
        labeled_video_paths: list[Tuple[str, Optional[dict]]],
        transform: Optional[Callable[[dict], Any]] = None,
        temporal_mix: bool = False,
        stance_model: torch.nn.Module = None,
        swing_model: torch.nn.Module = None,

    ) -> None:
        super().__init__()

        self._transform = transform
        self._labeled_videos = labeled_video_paths
        self._gait_cycle = gait_cycle
        if temporal_mix:
            self._temporal_mix = TemporalMix()
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

        if self._temporal_mix:
            # should return the new frame, named temporal mix.
            defined_vframes = self._temporal_mix(vframes, gait_cycle_index, bbox)
        else:
            # split gait by gait cycle index, first phase or second phase
            defined_vframes, used_gait_idx = split_gait_cycle(vframes, gait_cycle_index, self._gait_cycle)

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


            sample_info_dict["video"] = torch.stack(video_t_list, dim=0) # c, t, h, w
        else:
            print("no transform")

        return sample_info_dict

def labeled_gait_video_dataset(
    gait_cycle: int,
    transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    dataset_idx: Dict = None,
    temporal_mix: bool = False,
    gait_model: List[Type[Union[torch.nn.Module, Any]]] = None,
) -> LabeledGaitVideoDataset:

    stance_model = gait_model[0]
    swing_model = gait_model[1]

    dataset = LabeledGaitVideoDataset(
        gait_cycle,
        dataset_idx,
        transform,
        temporal_mix,
        stance_model,
        swing_model
    )

    return dataset