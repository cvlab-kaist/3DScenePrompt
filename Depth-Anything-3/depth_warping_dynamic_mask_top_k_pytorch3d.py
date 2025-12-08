import os
PATH = '/mnt/cache' # in the case of singularity setup
os.environ['TRANSFORMERS_CACHE'] = PATH
os.environ['HF_HOME'] = PATH
os.environ['HF_DATASETS_CACHE'] = PATH
os.environ['TORCH_HOME'] = PATH

#!/usr/bin/env python3
"""
3D Point Cloud Inference and Visualization Script

This script performs inference using the ARCroco3DStereo model and visualizes the
resulting 3D point clouds with the PointCloudViewer. Use the command-line arguments
to adjust parameters such as the model checkpoint path, image sequence directory,
image size, device, etc.

Usage:
    python usage.py [--model_path MODEL_PATH] [--seq_path SEQ_PATH] [--size IMG_SIZE]
                            [--device DEVICE] [--vis_threshold VIS_THRESHOLD] [--output_dir OUT_DIR]

Example:
    python usage.py --model_path src/cut3r_512_dpt_4_64.pth \
        --seq_path examples/001 --device cuda --size 512
"""

import os
import numpy as np
import torch
import time
import random
import cv2
import argparse

from tqdm import tqdm
from natsort import natsorted
random.seed(42)
from cvd_opt.geometry_utils import NormalGenerator
from torchvision.utils import save_image
from PIL import Image
import torch
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
    AlphaCompositor,
    PointsRenderer
)
from pytorch3d.structures import Pointclouds
from typing import Tuple
from pytorch3d.renderer import compositing, OrthographicCameras
import torch.nn.functional as F
import imageio



def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run 3D point cloud inference and visualization using ARCroco3DStereo."
    )

    parser.add_argument(
        "--base_path",
        type=str,
        default="",
        help="Path to the directory containing the image sequence.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (e.g., 'cuda' or 'cpu').",
    )
   
    parser.add_argument(
        "--outdir",
        type=str,
        default="./demo_tmp",
        help="value for tempfile.tempdir",
    )
    parser.add_argument("--batch", type=int, default=-1, help="batch size")
    parser.add_argument("--total_batch", type=int, default=5, help="batch size")
    parser.add_argument("--inverse", action='store_true', help="inverse order")
    parser.add_argument("--shuffle", action='store_true', help="shuffle order")
    parser.add_argument("--top_k", type=int, default=7, help="top_k for depth selection")

    return parser.parse_args()


def prepare_dataset(file_path, dynamic_mask_dir, device):
    data = np.load(file_path)
    
    # clip max frames            
    images = data['images']
    depth = data['depths']
    camera_c2w = data['cam_c2w']
    camera_intrinsic = data['intrinsic']

    images = torch.from_numpy(images).permute(0, 3, 1, 2).float()
    camera_c2w = torch.from_numpy(camera_c2w).float()
    camera_intrinsic = torch.from_numpy(camera_intrinsic).float()
    depth = torch.from_numpy(depth).float()
    
    dynamic_mask = []
    for frame_idx in range(images.shape[0]):
        mask = Image.open(os.path.join(dynamic_mask_dir, f'{frame_idx:06d}.png'))
        mask = mask.resize((images.shape[3], images.shape[2]), Image.BILINEAR)
        mask_np = np.array(mask)
        mask_bin = (mask_np > 0).astype(np.uint8)
        # 3) Dilate by ~5px radius (kernel size = 11×11)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        mask_dilated = cv2.dilate(mask_bin, kernel, iterations=3)  # still 0/1

        # 4) Stack to 3 channels and scale back to 0–255
        # mask_stack = np.stack([mask_dilated]*3, axis=-1) * 255    # shape: (H, W, 3)

        # 5) To tensor [1, C, H, W]
        if len(mask_dilated.shape) == 2:
            mask_dilated = np.stack([mask_dilated], axis=-1)
        mask_tensor = torch.from_numpy(mask_dilated).unsqueeze(0).permute(0,3,1,2).float() * 255
        dynamic_mask.append(mask_tensor)
        
    dynamic_mask = torch.cat(dynamic_mask)
    dynamic_mask = dynamic_mask[..., 0, :, :].unsqueeze(dim=1)
    
    return images.to(device), camera_c2w.to(device), camera_intrinsic[0].to(device), depth.to(device), dynamic_mask.to(device)


def depth_to_3d_batch(depth_map, intrinsic_matrix):
    batch, height, width = depth_map.shape
    
    # i, j = torch.meshgrid(torch.arange(width), torch.arange(height)).to(depth_map.device)
    i, j = np.meshgrid(np.arange(width), np.arange(height))
    i = torch.from_numpy(i).float().to(depth_map.device)
    j = torch.from_numpy(j).float().to(depth_map.device)
    # Convert pixel coordinates and depth values to 3D points
    x = (i - intrinsic_matrix[0 ,0, 2]) * depth_map / intrinsic_matrix[:, 0, 0].unsqueeze(-1).unsqueeze(-1)
    y = (j - intrinsic_matrix[0 ,1, 2]) * depth_map / intrinsic_matrix[:, 1, 1].unsqueeze(-1).unsqueeze(-1)
    z = depth_map
    
    points_3d = torch.stack([x, y, z], axis=-1)
    return points_3d


    

def make_projection_matrix(intrinsic_matrix, img_width, img_height, device):
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]

    near = 0.1
    far  = 50.0

    W = img_width
    H = img_height

    # 1) intrinsics -> l, r, b, t (near plane)
    l = (0.0 - cx) * near / fx
    r = (W  - cx) * near / fx
    b = (0.0 - cy) * near / fy
    t = (H  - cy) * near / fy

    # 2) OpenGL style projection matrix
    P = torch.zeros((4, 4), device=device)

    P[0, 0] =  2.0 * near / (r - l)           # 2n/(r-l)
    P[0, 2] =  (r + l) / (r - l)              # (r+l)/(r-l)

    P[1, 1] =  2.0 * near / (t - b)           # 2n/(t-b)
    P[1, 2] =  (t + b) / (t - b)              # (t+b)/(t-b)

    P[2, 2] = -(far + near) / (far - near)    # -(f+n)/(f-n)
    P[2, 3] = -(2.0 * far * near) / (far - near)  # -2fn/(f-n)

    P[3, 2] = -1.0                            # -1
    # P[3, 3] = 0.0                           # 이미 0

    return P

def rasterize_points_func(pts3D: Pointclouds, image_size: Tuple[int, int], radius: float, **kwargs):

    cameras = OrthographicCameras(device='cuda')

    raster_settings = PointsRasterizationSettings(
        image_size=(image_size[0],image_size[0]),
        radius=radius[0],
        points_per_pixel=5,
    )
    
    # 3. Rasterizer + Renderer (AlphaCompositor)
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    renderer = PointsRenderer(rasterizer=rasterizer, compositor=AlphaCompositor())
    
    # 4. rendering (B, H, W, 4) -> (B, C, H, W)
    rendered_images = renderer(pts3D)
    output = rendered_images.permute(0, 3, 1, 2)
    
    # # (B, H, W, C) -> (B, C, H, W)
    output = F.interpolate(
            output, 
            size=image_size, # (H, W)
            mode='bilinear', 
            align_corners=False
        )
    
    fragments = rasterizer(pts3D)  # (B, H_r, W_r, K)
    visible_mask = (fragments.idx[..., 0] >= 0)  # (B, H_r, W_r), bool
    
    visible_mask = visible_mask.unsqueeze(1).float()  # (B, 1, H_r, W_r)
    visible_mask = F.interpolate(
        visible_mask,
        size=image_size,
        mode="nearest",
    )  # (B, 1, H, W)

    
    visible_ratio = visible_mask.mean(dim=(2, 3))  # (B, 1)
    
    visible_ratio = visible_ratio.squeeze(1)

    return output, visible_ratio

   
def run_inference(args):
    """
    Execute the full inference and visualization pipeline.

    Args:
        args: Parsed command-line arguments.
    """
    # Set up the computation device.
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available. Switching to CPU.")
        device = "cpu"


    out_base_path = args.outdir
    input_base_path = args.base_path
    folder_names = [d for d in os.listdir(input_base_path) if os.path.isdir(os.path.join(input_base_path, d))]
    folder_names = natsorted(folder_names)

    if args.batch!=  -1:
        data_len = len(folder_names)
        folder_names = folder_names[int(args.batch/args.total_batch* data_len):int((args.batch+1)/args.total_batch* data_len)]
            
    for _, scene_name in enumerate(folder_names):  
        # try: 
            time1 = time.time()
            output_path = os.path.join(out_base_path, f'{scene_name}.mp4')
                
            print(f'gpu : {args.batch} percentage : {_}/{len(folder_names)}')
            print(f"gpu : {args.batch} output_dir : {output_path}")
            
            folder_path = os.path.join(input_base_path, scene_name, 'DA3.npz')
            dynamic_mask_dir = os.path.join(input_base_path, scene_name, 'dynamic_mask')
            images, camera_c2w, camera_intrinsic, depth, dynamic_mask = prepare_dataset(folder_path, dynamic_mask_dir, device)
            # P = make_projection_matrix(camera_intrinsic, images.shape[2], images.shape[3], device)
            
            B, _, W, H = images.shape
            normal_generator = NormalGenerator(images.shape[-2], images.shape[-1])
            
            K_inv = torch.linalg.inv(camera_intrinsic)
            normal = normal_generator(depth.unsqueeze(dim=1), K_inv[None])
            
            # frame_start, frame_end = 60, 100
            frame_start, frame_end = len(images) - 40, len(images)
            warped_img = []
            for i in range(frame_start, frame_end):
                depth1 = depth[:frame_start] # (B, H, W)
                dynamic_mask1 = dynamic_mask[:frame_start] # (B, 1, H, W)
                B, H, W = depth1.shape
                intrinsic_matrix1 = camera_intrinsic.unsqueeze(0).repeat(B, 1, 1)
                intrinsic_matrix2 = camera_intrinsic.unsqueeze(0).repeat(B, 1, 1)
                
                points_3d_frame1 = depth_to_3d_batch(depth1, intrinsic_matrix1).reshape(B, -1, 3)
                # points_3d_frame1_hom = np.concatenate([points_3d_frame1, np.ones((points_3d_frame1.shape[0], 1))], axis=1).T
                points_3d_frame1_hom = torch.cat([points_3d_frame1, torch.ones((B, points_3d_frame1.shape[1], 1)).to(depth1.device)], dim=-1).permute(0, 2, 1)
                # Calculate the transformation matrix from frame 1 to frame 2
                camera_w2c_1 = torch.linalg.inv(camera_c2w[:frame_start])
                camera_w2c_2 = torch.linalg.inv(camera_c2w[i]).repeat(len(camera_w2c_1),1,1)
                transformation_matrix = (camera_w2c_2) @ torch.linalg.inv(camera_w2c_1)
                points_3d_frame2_hom = transformation_matrix @ points_3d_frame1_hom
                points_3d_frame2 = (points_3d_frame2_hom[:,:3, :])
                
                points_world_hom = torch.linalg.inv(camera_w2c_1) @ points_3d_frame1_hom
                projected_2d_hom = intrinsic_matrix2 @ points_3d_frame2
                # Convert from homogeneous coordinates to 2D image coordinates
                projected_2d = projected_2d_hom[:,:2, :] / projected_2d_hom[:,2, :].unsqueeze(1)
                near = 0.1
                far = 50.0
                x_ndc = -(projected_2d[:,0,:] / W) * 2 + 1
                y_ndc = -(projected_2d[:,1,:] / H) * 2 + 1
                z_ndc = (projected_2d_hom[:,2,:] - near) / (far - near)

                ndc_points = torch.stack([x_ndc, y_ndc, z_ndc], dim=-1)
                pts3Ds = []
                transposed_normal = transformation_matrix[:,:3,:3] @ normal[:frame_start].reshape(B, 3, -1)
                
                ndcs = []
                features = []
                for idx in range(B):
                    ndc_point_ = ndc_points[idx:idx+1]
                    feature_ = images[:frame_start][idx:idx+1].permute(0,2,3,1).reshape(1, -1, 3)/255
                    dynamic_mask_ = dynamic_mask1[idx:idx+1].reshape(-1)
                    total_mask = (transposed_normal[idx,2] > 0) & (dynamic_mask_.bool()) & (ndc_point_[0,:,2] > 0) & (ndc_point_[0,:,2] < 1)
                    
                    masked_ndc = ndc_point_[:, total_mask, :]
                    masked_feature = feature_[:, total_mask, :]
                    pts3D = Pointclouds(points=masked_ndc, features=masked_feature)
                    pts3Ds.append(pts3D)
                    ndcs.append(masked_ndc.squeeze(0))
                    features.append(masked_feature.squeeze(0))
                radius = 1.0 / H * 2.0, 
                
                
                img_list = []
                visible_ratio_list = []
                for idx in range(B):
                    img, visible_ratio = rasterize_points_func(pts3Ds[idx], (H,W), radius)
                    img_list.append(img)
                    visible_ratio_list.append(visible_ratio)
                    
                visible_ratio = torch.cat(visible_ratio_list, dim=0)
                idx_sorted = torch.argsort(visible_ratio, descending=True)
                top_k_indices = idx_sorted[:args.top_k].tolist()
                
                top_k_points = [ndcs[i] for i in top_k_indices]
                top_k_colors = [features[i] for i in top_k_indices]
                top_k_points = torch.cat(top_k_points, dim=0)
                top_k_colors = torch.cat(top_k_colors, dim=0)
                
                pts3D = Pointclouds(points=top_k_points.unsqueeze(0), features=top_k_colors.unsqueeze(0))
                imgs, _ = rasterize_points_func(pts3D, (H,W), radius)
                warped_img.append(imgs)
                
            # del view2_tensor, view1_tensor, flow_ij, importance_ij, top_k_indices, top_k_depth, 
            time2 = time.time()
            print(f"gpu : {args.batch} time taken : {time2-time1}")
            warped_img = torch.cat(warped_img, dim=0) * 255
            cond_video = torch.cat([images[frame_start-9:frame_start], warped_img], dim=0)
            resized = F.interpolate(
                cond_video, 
                size=(480, 720),   # (H, W)
                mode="bilinear",
                align_corners=False
            )   
            frames = resized.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            writer = imageio.get_writer(output_path, fps=16, codec='libx264')
            for f in frames:
                writer.append_data(f)
            writer.close()

def main():
    args = parse_args()
    
    run_inference(args)


if __name__ == "__main__":
    main()

