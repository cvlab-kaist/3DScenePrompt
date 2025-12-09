import os
PATH = '/mnt/cache' # in the case of singularity setup
os.environ['TRANSFORMERS_CACHE'] = PATH
os.environ['HF_HOME'] = PATH
os.environ['HF_DATASETS_CACHE'] = PATH
os.environ['TORCH_HOME'] = PATH



import os
import numpy as np
import torch
import time
import glob
import random
import cv2
import argparse
import tempfile
import shutil
from copy import deepcopy
import imageio as iio

from einops import rearrange
# from src.dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from tqdm import tqdm
from natsort import natsorted
from camera_tracking_scripts.sintel_get_dynamics import compute_optical_flow_batch_top3_reference_batch, compute_optical_flow_batch
random.seed(42)
from cvd_opt.geometry_utils import NormalGenerator
from torchvision.utils import save_image
from third_party.raft import load_RAFT
from third_party.sam2.sam2.build_sam import build_sam2_video_predictor

from sam2_utils.track_utils import sample_points_from_masks, sample_points_from_all_dbscan_clusters, sample_points_from_all_dbscan_clusters_w_margin
from torch.nn import functional as F

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
    parser.add_argument("--inverse", action='store_true', help="inverse order")
    parser.add_argument("--shuffle", action='store_true', help="shuffle order")
    parser.add_argument("--total_batch", type=int, default=9)
    return parser.parse_args()



def get_flow(view1, view2, img, flow_net, sintel_ckpt=False): #TODO: test with gt flow
    print('precomputing flow...')
    # get_valid_flow_mask = OccMask(th=3.0)
    
    pair_imgs = [img[view1], img[view2]]
    with torch.no_grad():
        chunk_size = 64
        flow_ij = []
        flow_ji = []
        num_pairs = len(pair_imgs[0])
        for i in tqdm(range(0, num_pairs, chunk_size)):
            end_idx = min(i + chunk_size, num_pairs)
            imgs_ij = [torch.tensor(pair_imgs[0][i:end_idx]).float().to(device),
                    torch.tensor(pair_imgs[1][i:end_idx]).float().to(device)]
            flow_ij.append(flow_net(imgs_ij[0], 
                                    imgs_ij[1], 
                                    iters=20, test_mode=True)[1])
            flow_ji.append(flow_net(imgs_ij[1], 
                                    imgs_ij[0], 
                                    iters=20, test_mode=True)[1])

        flow_ij = torch.cat(flow_ij, dim=0)
        flow_ji = torch.cat(flow_ji, dim=0)
        # valid_mask_i = get_valid_flow_mask(flow_ij, flow_ji)
        # valid_mask_j = get_valid_flow_mask(flow_ji, flow_ij)
    print('flow precomputed')
    # delete the flow net
    if flow_net is not None: del flow_net
    # return flow_ij, flow_ji, valid_mask_i, valid_mask_j
    return flow_ij, flow_ji


def prepare_dataset(file_path, cache_dir=None):
    data = np.load(file_path)
            
    
    images = data['images']
    depth = data['depths']
    camera_extrinsic = data['cam_c2w']
    camera_intrinsic = data['intrinsic']

        
    images = torch.from_numpy(images).permute(0, 3, 1, 2).float()
    camera_extrinsic = torch.from_numpy(camera_extrinsic).float()
    camera_intrinsic = torch.from_numpy(camera_intrinsic).float()
    depth = torch.from_numpy(depth).float()
    
    # iijj = np.load(f"{cache_dir}/ii-jj.npy")        
    def get_iijj_list(num_frames, steps=[1, 2, 4, 8, 15]):
        ii = []
        jj = []

        for step in steps:
            for i in range(0, num_frames - step):
                ii.append(i)
                jj.append(i + step)
                
        iijj = np.stack((ii, jj), axis=0).astype(np.int32)
        return iijj
    
    iijj = get_iijj_list(len(images), steps=[1,2,4,8,15])
    
    iijj = torch.from_numpy(np.ascontiguousarray(iijj)).float()
    ii = iijj[0, ...].long()
    jj = iijj[1, ...].long()

    return images, camera_extrinsic, camera_intrinsic[0], depth, ii, jj 

from sklearn.cluster import DBSCAN

def cluster_and_select(clusters: torch.Tensor,
                       eps: float = 10.0,
                       min_samples: int = 1,
                       keep_noise: bool = False) -> torch.Tensor:
    """
    clusters: (C, 3, 2) — C개의 클러스터, 각 클러스터는 3개의 (x,y) 포인트
    eps: DBSCAN eps 파라미터 (픽셀 단위 거리)
    min_samples: DBSCAN min_samples 파라미터
    keep_noise: noise 군집(label==-1)을 결과에 포함할지 여부

    returns: (M, 3, 2) — 필터링/대표 추출된 클러스터
    """
    # 1) 각 클러스터의 centroid 계산: (C,2)
    centroids = clusters.mean(dim=1)

    # 2) NumPy로 변환하여 DBSCAN 수행
    centroids_np = centroids.cpu().numpy()
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(centroids_np)
    labels = db.labels_  # shape: (C,)

    # 3) label별로 대표 클러스터 선택
    selected = []
    unique_labels = set(labels)
    for lbl in unique_labels:
        if lbl == -1 and not keep_noise:
            # noise 클러스터를 버리려면 continue
            continue

        # 같은 lbl에 속한 인덱스들
        idxs = np.where(labels == lbl)[0]
        # 그중 첫 번째 클러스터를 대표로 선택
        chosen_idx = idxs[0]
        selected.append(clusters[chosen_idx])

    if len(selected) == 0:
        # 혹시 전부 버려졌다면, 원본 반환
        return clusters

    return torch.stack(selected, dim=0)  # (M, 3, 2)

def min_pairwise_dist(cluster):
    # cluster: (3,2)
    d01 = (cluster[0] - cluster[1]).norm()
    d02 = (cluster[0] - cluster[2]).norm()
    d12 = (cluster[1] - cluster[2]).norm()
    return torch.min(torch.stack([d01, d02, d12]))



def get_dynamic_mask(depth, edges, view1_tensor, view2_tensor, flow_ij, flow_ji, gt_flow_ij, gt_flow_ji):
    err_map_i = torch.norm(flow_ij[:, :2, ...].cuda() - gt_flow_ij, dim=1)
    err_map_j = torch.norm(flow_ji[:, :2, ...].cuda() - gt_flow_ji, dim=1)
    # normalize the error map for each pair
    err_map_i = (err_map_i - err_map_i.amin(dim=(1, 2), keepdim=True)) / (err_map_i.amax(dim=(1, 2), keepdim=True) - err_map_i.amin(dim=(1, 2), keepdim=True))
    err_map_j = (err_map_j - err_map_j.amin(dim=(1, 2), keepdim=True)) / (err_map_j.amax(dim=(1, 2), keepdim=True) - err_map_j.amin(dim=(1, 2), keepdim=True))
    
    dynamic_masks = [[] for _ in range(len(depth))]
    # motion_mask_thre = 0.35
    motion_mask_thre = 0.7
    for i in range(len(edges)):
        i_idx = int(view1_tensor[i])
        j_idx = int(view2_tensor[i])
        dynamic_masks[i_idx].append(err_map_i[i])
        dynamic_masks[j_idx].append(err_map_j[i])
        
        
    for i in range(len(depth)):
        dynamic_masks[i] = torch.stack(dynamic_masks[i]).mean(dim=0) > motion_mask_thre
    return dynamic_masks




def refine_motion_mask_w_sam2(images, dynamic_masks, predictor):
    n_imgs = len(images)
    try:
        frame_tensors = images.to(device)
        # inference_state = predictor.init_state(video_path=frame_tensors/255)
        import imageio
        imageio.mimwrite('original_video.mp4', (frame_tensors.cpu().numpy().transpose(0,2,3,1)).astype(np.uint8), fps=16, macro_block_size=4)
        inference_state = predictor.init_state(video_path='original_video.mp4')
        mask_list = [dynamic_masks[i] for i in range(n_imgs)]
        
        # Process even frames
        # predictor.reset_state(inference_state)
        masks = torch.stack(mask_list)
        masks = masks.cpu().float().numpy()
        # all_sample_points = sample_points_from_masks(masks=masks, num_points=5)
        
        clustered_points, clustered_labels = sample_points_from_all_dbscan_clusters_w_margin(masks, num_points_per_cluster=3, margin=20)
        # if clustered_points[0].shape[0] == 0:
        #     breakpoint()
        #     clustered_points, clustered_labels = sample_points_from_all_dbscan_clusters(masks, num_points_per_cluster=3)
        
        queries = torch.cat([
            torch.cat([
                torch.full((torch.tensor(cluster).reshape(-1, 2).shape[0], 1), i),
                torch.tensor(cluster).reshape(-1, 2)
            ], dim=1)
            for i, cluster in enumerate(clustered_points)
        ], dim=0)
        
        # if no query points, return original masks
        if queries.shape[0] == 0:
            return torch.stack(mask_list).float().cpu().unsqueeze(1).repeat(1,3,1,1)
        
        # from cotracker.utils.visualizer import Visualizer
        queries = queries.cuda()
        batch_size = 512

        tracks_chunks = []
        vis_chunks    = []

        for i in range(0, queries.shape[0], batch_size):
            q_chunk = queries[i : i + batch_size]       # shape: [bs, ...]
            pt, pv = cotracker(images[None].to(device), queries=q_chunk[None])
            # pt: [1, T, N_chunk, 2], pv: [1, T, N_chunk, 1]
            tracks_chunks.append(pt)
            vis_chunks.append(pv)

        pred_tracks     = torch.cat(tracks_chunks, dim=2)  # -> [1, T, N, 2]
        pred_visibility = torch.cat(vis_chunks,    dim=2)  # -> [1, T, N, 1]
        # vis = Visualizer(save_dir="./saved_videos", pad_value=120, linewidth=3)
        # vis.visualize(images[None].float().to(device), pred_tracks, pred_visibility)
        
        
        clusters = pred_tracks[0, 0]        # (459, 2)
        clusters = clusters.view(-1, 3, 2) # (153, 3, 2), 459/3 = 153

        try:     
            filtered = cluster_and_select(
                clusters,
                eps=5.0,
                min_samples=1,
                keep_noise=False
            )
        except:
            filtered = clusters
        
        k = 30
        N = filtered.shape[0]
        if N > k:
            idx = torch.randperm(N, device=filtered.device)[:k]
            subsampled = filtered[idx]
        else:
            subsampled = filtered
        
        # sampled_point_vis(frame_tensors, clusters, image_name = 'clusters.png')
        # sampled_point_vis(frame_tensors, filtered, image_name = 'filtered.png')
        # sampled_point_vis(frame_tensors, subsampled, image_name = 'subsampled.png')
        
        autocast_dtype = torch.bfloat16 if device == 'cuda' else torch.float32
        with torch.autocast(device_type=device, dtype=autocast_dtype):
            for object_id, (label, points) in enumerate(zip(subsampled, subsampled), start=1):
                labels = np.ones((points.shape[0]), dtype=np.int32)
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=object_id,
                    points=points,
                    labels=labels,
                )
            
                
            video_segments = {}  # video_segments contains the per-frame segmentation results
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
                
                
            total_mask = []
            for frame_idx, segments in video_segments.items():
                object_ids = list(segments.keys())
                masks = list(segments.values())
                masks = np.concatenate(masks, axis=0)
                total_mask.append(np.any(masks, axis=0))
            total_mask = np.array(total_mask, dtype=np.uint8)
            
            sam_refined = torch.from_numpy(total_mask).cpu().unsqueeze(1).repeat(1,3,1,1)
            original_mask = torch.stack(mask_list).float().cpu().unsqueeze(1).repeat(1,3,1,1)
            # original_mask = F.interpolate(original_mask, size=sam_refined.shape[2:], mode='nearest')
            sam_refined = F.interpolate(sam_refined.float(), size=original_mask.shape[2:], mode='nearest') 
            union_mask = torch.logical_or(sam_refined, original_mask)
          
    finally:
        # Restore previous TF32 settings
        if device == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = prev_allow_tf32
            torch.backends.cudnn.allow_tf32 = prev_allow_cudnn_tf32
            
    return union_mask[:, 0, ...].float()

def sampled_point_vis(frame_tensors, subsampled, image_name = 'points_000000.png'):
    img = frame_tensors[0].cpu().numpy().transpose(1, 2, 0)[:,:,::-1]
    img = (img).astype(np.uint8)

            # draw only subsampled points
    radius = 5
    color = (0,255,0)
    for cluster in subsampled:
        for x,y in cluster:
            cv2.circle(img, (int(x), int(y)), radius, color, -1)
            # save
    output_path = './test'
    os.makedirs(output_path, exist_ok=True)
    cv2.imwrite(os.path.join(output_path, image_name), img)


if __name__ == "__main__":
    args = parse_args()
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

    raft_checkpoint = "/mnt/cache/RAFT/Tartan-C-T-TSKH-spring540x960-M.pth" if os.path.exists("/mnt/cache/RAFT/Tartan-C-T-TSKH-spring540x960-M.pth") else 'third_party/RAFT/models/Tartan-C-T-TSKH-spring540x960-M.pth'
    flow_net = load_RAFT(raft_checkpoint)
    flow_net = flow_net.to(device)
    flow_net.eval()
    

    out_base_path = args.outdir
    input_base_path = args.base_path
    from natsort import natsorted
    folder_names = [d for d in os.listdir(input_base_path) if os.path.isdir(os.path.join(input_base_path, d))]
    folder_names = natsorted(folder_names)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Save previous TF32 settings
    if device == 'cuda':
        prev_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        prev_allow_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        # Enable TF32 for Ampere GPUs
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    sam2_checkpoint = '/mnt/cache/sam2/sam2.1_hiera_large.pt' if os.path.exists('/mnt/cache/sam2/sam2.1_hiera_large.pt') else "third_party/sam2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
    
    
    if args.batch!=  -1:
        data_len = len(folder_names)
        folder_names = folder_names[int(args.batch/args.total_batch* data_len):int((args.batch+1)/args.total_batch* data_len)]
        

    for _, scene_name in enumerate(folder_names):   
        # try:
            time1 = time.time()
            print(f'gpu : {args.batch} percentage : {_}/{len(folder_names)}')
            print(f"gpu : {args.batch} output_dir : {out_base_path}/{scene_name}/dynamic_mask")
            if os.path.exists(f"{out_base_path}/{scene_name}/dynamic_mask"):
                print(f"already processed {scene_name}")
                continue
            folder_path = os.path.join(input_base_path, scene_name, 'DA3.npz')
            output_path = os.path.join(out_base_path, scene_name, 'dynamic_mask')
            
            images, camera_extrinsic, camera_intrinsic, depth, ii, jj  = prepare_dataset(folder_path)
            
            view1_tensor = ii
            view2_tensor = jj
            camera_intrinsic = camera_intrinsic.unsqueeze(dim=0).repeat(view1_tensor.shape[0], 1, 1)
            
            flow_ij, importance_ij  = compute_optical_flow_batch(depth[view1_tensor], depth[view2_tensor], 
                                                        camera_extrinsic[view1_tensor], camera_extrinsic[view2_tensor], 
                                                        camera_intrinsic, camera_intrinsic)

            flow_ji, importance_ji  = compute_optical_flow_batch(depth[view2_tensor], depth[view1_tensor], 
                                                        camera_extrinsic[view2_tensor], camera_extrinsic[view1_tensor], 
                                                        camera_intrinsic, camera_intrinsic)


            gt_flow_ij, gt_flow_ji = get_flow(view1_tensor, view2_tensor, images, flow_net, sintel_ckpt=False)

            
            dynamic_masks = get_dynamic_mask(depth, ii, view1_tensor, view2_tensor, flow_ij, flow_ji, gt_flow_ij, gt_flow_ji)
            del flow_ij, flow_ji, gt_flow_ij, gt_flow_ji
            del ii, jj
            del view1_tensor, view2_tensor
            del camera_extrinsic, camera_intrinsic
            del depth
            del importance_ij, importance_ji
            ############### refine with sam2 ##################
            dynamic_masks = refine_motion_mask_w_sam2(images, dynamic_masks, predictor)
            
            os.makedirs(output_path, exist_ok=True)
            for i in range(len(dynamic_masks)):
                save_image(dynamic_masks[i].cpu().float(), f"{output_path}/{i:06d}.png")
            print(f"gpu : {args.batch} time : {time.time()-time1}")
        # except:
        #     pass