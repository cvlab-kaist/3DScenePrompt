import os
PATH = '/mnt/cache' # in the case of singularity setup
os.environ['TRANSFORMERS_CACHE'] = PATH
os.environ['HF_HOME'] = PATH
os.environ['HF_DATASETS_CACHE'] = PATH
os.environ['TORCH_HOME'] = PATH


# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Preprocess flow for MegaSaM."""

import glob
import os
import sys

# pylint: disable=g-bad-import-order
# pylint: disable=g-import-not-at-top

import numpy as np
import torch
# FLOW ESTIMATOR
sys.path.append('cvd_opt/core')
from raft import RAFT
from core.utils.utils import InputPadder
from pathlib import Path  # pylint: disable=g-importing-member

import argparse
import tqdm
import cv2
import time

def warp_flow(img, flow):
  h, w = flow.shape[:2]
  flow_new = flow.copy()
  flow_new[:, :, 0] += np.arange(w)
  flow_new[:, :, 1] += np.arange(h)[:, np.newaxis]

  res = cv2.remap(
      img, flow_new, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
  )
  return res


def resize_flow(flow, img_h, img_w):
  # flow = np.load(flow_path)
  flow_h, flow_w = flow.shape[0], flow.shape[1]
  flow[:, :, 0] *= float(img_w) / float(flow_w)
  flow[:, :, 1] *= float(img_h) / float(flow_h)
  flow = cv2.resize(flow, (img_w, img_h), cv2.INTER_LINEAR)

  return flow


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--model', default='raft-things.pth', help='restore checkpoint'
  )
  parser.add_argument('--small', action='store_true', help='use small model')
  parser.add_argument('--scene_name', type=str, help='use small model')
  parser.add_argument('--outdir')

  parser.add_argument('--base_path', help='dataset for evaluation')
  parser.add_argument(
      '--num_heads',
      default=1,
      type=int,
      help='number of heads in attention and aggregation',
  )
  parser.add_argument(
      '--position_only',
      default=False,
      action='store_true',
      help='only use position-wise attention',
  )
  parser.add_argument(
      '--position_and_content',
      default=False,
      action='store_true',
      help='use position and content-wise attention',
  )
  parser.add_argument(
      '--mixed_precision', action='store_true', help='use mixed precision'
  )
  parser.add_argument(
      '--batch', type=int, default=-1, help='batch size for training'
  )
  parser.add_argument('--inverse', action='store_true', help='inverse the flow')
  parser.add_argument('--shuffle', action='store_true', help='shuffle the data')
  parser.add_argument("--total_batch", type=int, default=9)
  
  args = parser.parse_args()

  model = torch.nn.DataParallel(RAFT(args))
  model.load_state_dict(torch.load(args.model))
  print(f'Loaded checkpoint at {args.model}')
  flow_model = model.module
  flow_model.cuda()  # .eval()
  flow_model.eval()

  out_base_path = args.outdir  # "./outputs"
  input_base_path = args.base_path  # "/home/zhengqili/filestore/DAVIS/DAVIS/JPEGImages/480p"
  # os.makedirs(outdir, exist_ok=True)
  folder_names = [d for d in os.listdir(input_base_path) if os.path.isdir(os.path.join(input_base_path, d))]
  from natsort import natsorted
  folder_names = natsorted(folder_names)
  batch_size = 20
  
  if args.batch != -1:
      data_len = len(folder_names)
      folder_names = folder_names[int(args.batch/args.total_batch* data_len):int((args.batch+1)/args.total_batch* data_len)]
      print(f"Processing batch {args.batch} with {len(folder_names)} scenes.")
  if args.inverse:
      folder_names = folder_names[::-1]
  if args.shuffle:
      import random
    #   random.seed(42)  # For reproducibility
      random.shuffle(folder_names)
      
  for _, scene_name in enumerate(folder_names):
    try:
        outdir_scene = os.path.join(out_base_path, scene_name, 'cache_flow')
        input_dir = os.path.join(input_base_path, scene_name, 'color')
        time1 = time.time()
        print('percentage : ', _ / len(folder_names))
        if os.path.exists(os.path.join(outdir_scene,'flows.npy')):
            print(f"Skipping {scene_name} as output directory already exists.")
            continue
        image_list = sorted(
            glob.glob(os.path.join(input_dir, '*.png'))
        )  # [::stride]
        image_list += sorted(
            glob.glob(os.path.join(input_dir, '*.jpg'))
        )  # [::stride]
        img_data = []

        for t, (image_file) in tqdm.tqdm(enumerate(image_list)):
            image = cv2.imread(image_file)[..., ::-1]  # rgb
            h0, w0, _ = image.shape
            h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
            w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))
            image = cv2.resize(image, (w1, h1))
            image = image[: h1 - h1 % 8, : w1 - w1 % 8].transpose(2, 0, 1)
            img_data.append(image)

        img_data = np.array(img_data)

        flows_low = []

        flows_high = []
        flow_masks_high = []

        flow_init = None
        flows_arr_low_bwd = {}
        flows_arr_low_fwd = {}

        ii = []
        jj = []
        flows_arr_up = []
        masks_arr_up = []

        for step in [1, 2, 4, 8, 15]:
            flows_arr_low = []
            for i in tqdm.tqdm(range(max(0, -step), img_data.shape[0] - max(0, step))):
                image1 = (
                    torch.as_tensor(np.ascontiguousarray(img_data[i : i + 1]))
                    .float()
                    .cuda()
                )
                image2 = (
                    torch.as_tensor(
                        np.ascontiguousarray(img_data[i + step : i + step + 1])
                    )
                    .float()
                    .cuda()
                )

                ii.append(i)
                jj.append(i + step)

                with torch.no_grad():
                    padder = InputPadder(image1.shape)
                    image1, image2 = padder.pad(image1, image2)
                    if np.abs(step) > 1:
                        flow_init = np.stack(
                            [flows_arr_low_fwd[i], flows_arr_low_bwd[i + step]], axis=0
                        )
                        flow_init = (
                            torch.as_tensor(np.ascontiguousarray(flow_init))
                            .float()
                            .cuda()
                            .permute(0, 3, 1, 2)
                        )
                    else:
                        flow_init = None

                    flow_low, flow_up, _ = flow_model(
                        torch.cat([image1, image2], dim=0),
                        torch.cat([image2, image1], dim=0),
                        iters=22,
                        test_mode=True,
                        flow_init=flow_init,
                    )

                    flow_low_fwd = flow_low[0].cpu().numpy().transpose(1, 2, 0)
                    flow_low_bwd = flow_low[1].cpu().numpy().transpose(1, 2, 0)

                    flow_up_fwd = resize_flow(
                        flow_up[0].cpu().numpy().transpose(1, 2, 0),
                        flow_up.shape[-2] // 2,
                        flow_up.shape[-1] // 2,
                    )
                    flow_up_bwd = resize_flow(
                        flow_up[1].cpu().numpy().transpose(1, 2, 0),
                        flow_up.shape[-2] // 2,
                        flow_up.shape[-1] // 2,
                    )

                    bwd2fwd_flow = warp_flow(flow_up_bwd, flow_up_fwd)
                    fwd_lr_error = np.linalg.norm(flow_up_fwd + bwd2fwd_flow, axis=-1)
                    fwd_mask_up = fwd_lr_error < 1.0

                    # flows_arr_low.append(flow_low_fwd)
                    flows_arr_low_bwd[i + step] = flow_low_bwd
                    flows_arr_low_fwd[i] = flow_low_fwd

                    # masks_arr_low.append(fwd_mask_low)
                    flows_arr_up.append(flow_up_fwd)
                    masks_arr_up.append(fwd_mask_up)
                    
        
            
            
            # num_batches = (img_data.shape[0] - step) // batch_size + ((img_data.shape[0] - step) % batch_size != 0)

            # for batch_idx in tqdm.tqdm(range(num_batches)):
            #     start_idx = batch_idx * batch_size
            #     end_idx = min(start_idx + batch_size, img_data.shape[0] - step)

            #     image1_batch = torch.as_tensor(img_data[start_idx:end_idx]).float().cuda()
            #     image2_batch = torch.as_tensor(img_data[start_idx + step:end_idx + step]).float().cuda()

            #     ii.extend(range(start_idx, end_idx))
            #     jj.extend(range(start_idx + step, end_idx + step))

            #     with torch.no_grad():
            #         padder = InputPadder(image1_batch.shape)
            #         image1_batch, image2_batch = padder.pad(image1_batch, image2_batch)

            #         input_batch = torch.cat([image1_batch, image2_batch], dim=0)
            #         reversed_input_batch = torch.cat([image2_batch, image1_batch], dim=0)

                    
            #         if np.abs(step) > 1:
            #             flows_arr_low_list = []
            #             breakpoint()
            #             for i in range(start_idx, end_idx):flows_arr_low_list.append(flows_arr_low_fwd[i])
            #             for i in range(start_idx, end_idx):flows_arr_low_list.append(flows_arr_low_bwd[i + step])
            #             flow_init = np.stack(flows_arr_low_list, axis=0)
                        
            #             # flow_init = np.stack(
            #             #     [flows_arr_low_fwd[i], flows_arr_low_bwd[i + step]], axis=0
            #             # )
            #             flow_init = (
            #                 torch.as_tensor(np.ascontiguousarray(flow_init))
            #                 .float()
            #                 .cuda()
            #                 .permute(0, 3, 1, 2)
            #             )
            #         else:
            #             flow_init = None

            #         flow_low, flow_up, _ = flow_model(
            #             input_batch,
            #             reversed_input_batch,
            #             iters=22,
            #             test_mode=True,
            #             flow_init=flow_init,
            #         )

            #         flow_up_fwd_all = flow_up[0::2]  # Even indices for forward flows
            #         flow_up_bwd_all = flow_up[1::2]  # Odd indices for backward flows

            #         for i in range(end_idx - start_idx):
            #             flow_up_fwd = flow_up_fwd_all[i].cpu().numpy()
            #             flow_up_bwd = flow_up_bwd_all[i].cpu().numpy()

            #             # numpy로 리사이징
            #             resized_fwd = resize_flow(flow_up_fwd.transpose(1, 2, 0), flow_up[0].shape[1] // 2, flow_up[0].shape[2] // 2)
            #             resized_bwd = resize_flow(flow_up_bwd.transpose(1, 2, 0), flow_up[0].shape[1] // 2, flow_up[0].shape[2] // 2)

            #             bwd2fwd_flow = warp_flow(resized_bwd, resized_fwd)
            #             fwd_lr_error = np.linalg.norm(resized_fwd + bwd2fwd_flow, axis=-1)
            #             fwd_mask_up = fwd_lr_error < 1.0
                        
            #             flows_arr_low.append(flow_low_fwd)
            #             flows_arr_low_bwd[i + step] = flow_low_bwd
            #             flows_arr_low_fwd[i] = flow_low_fwd


            #             flows_arr_up.append(resized_fwd.transpose(2, 0, 1))  # 다시 (C, H, W) 형태로 변환
            #             masks_arr_up.append(fwd_mask_up)


        os.makedirs(outdir_scene, exist_ok=True)
        iijj = np.stack((ii, jj), axis=0)
        flows_high = np.array(flows_arr_up).transpose(0, 3, 1, 2)
        flow_masks_high = np.array(masks_arr_up)[:, None, ...]
        # Path('./cache_flow/%s' % scene_name).mkdir(parents=True, exist_ok=True)
        np.save('%s/flows.npy' % outdir_scene, np.float16(flows_high))
        np.save('%s/flows_masks.npy' % outdir_scene, flow_masks_high)
        np.save('%s/ii-jj.npy' % outdir_scene, iijj)
        
        time2 = time.time()
        print(f"Processing time for scene {scene_name}: {time2 - time1} seconds")        
        # breakpoint()
    except: 
        pass# except Exception as e: