import numpy as np
import cv2
import os
from tqdm import tqdm
import argparse
import torch
from einops import rearrange
# from torchvision.utils import save_image

TAG_FLOAT = 202021.25
def flow_read(filename):
    """ Read optical flow from file, return (U,V) tuple.
    
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    f = open(filename,'rb')
    check = np.fromfile(f,dtype=np.float32,count=1)[0]
    assert check == TAG_FLOAT, 'flow_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine?'.format(TAG_FLOAT,check)
    width = np.fromfile(f,dtype=np.int32,count=1)[0]
    height = np.fromfile(f,dtype=np.int32,count=1)[0]
    size = width*height
    assert width > 0 and height > 0 and size > 1 and size < 100000000, 'flow_read:: Invalid input size (width = {0}, height = {1}).'.format(width,height)
    tmp = np.fromfile(f,dtype=np.float32,count=-1).reshape((height,width*2))
    u = tmp[:,np.arange(width)*2]
    v = tmp[:,np.arange(width)*2 + 1]
    return u,v

def cam_read(filename):
    """ Read camera data, return (M,N) tuple.
    
    M is the intrinsic matrix, N is the extrinsic matrix, so that

    x = M*N*X,
    where x is a point in homogeneous image pixel coordinates, and X is a
    point in homogeneous world coordinates.
    """
    f = open(filename,'rb')
    check = np.fromfile(f,dtype=np.float32,count=1)[0]
    assert check == TAG_FLOAT, 'cam_read:: Wrong tag in cam file (should be: {0}, is: {1}). Big-endian machine?'.format(TAG_FLOAT,check)
    M = np.fromfile(f,dtype='float64',count=9).reshape((3,3))
    N = np.fromfile(f,dtype='float64',count=12).reshape((3,4))
    return M,N

def depth_read(filename):
    """ Read depth data from file, return as numpy array. """
    f = open(filename,'rb')
    check = np.fromfile(f,dtype=np.float32,count=1)[0]
    assert check == TAG_FLOAT, 'depth_read:: Wrong tag in depth file (should be: {0}, is: {1}). Big-endian machine?'.format(TAG_FLOAT,check)
    width = np.fromfile(f,dtype=np.int32,count=1)[0]
    height = np.fromfile(f,dtype=np.int32,count=1)[0]
    size = width*height
    assert width > 0 and height > 0 and size > 1 and size < 100000000, 'depth_read:: Invalid input size (width = {0}, height = {1}).'.format(width,height)
    depth = np.fromfile(f,dtype=np.float32,count=-1).reshape((height,width))
    return depth

def RT_to_extrinsic_matrix(R, T):
    extrinsic_matrix = np.concatenate([R, T], axis=-1)
    extrinsic_matrix = np.concatenate([extrinsic_matrix, np.array([[0, 0, 0, 1]])], axis=0)
    return np.linalg.inv(extrinsic_matrix)

def depth_to_3d(depth_map, intrinsic_matrix):
    height, width = depth_map.shape
    i, j = np.meshgrid(np.arange(width), np.arange(height))
    
    # Convert pixel coordinates and depth values to 3D points
    x = (i - intrinsic_matrix[0, 2]) * depth_map / intrinsic_matrix[0, 0]
    y = (j - intrinsic_matrix[1, 2]) * depth_map / intrinsic_matrix[1, 1]
    z = depth_map
    
    points_3d = np.stack([x, y, z], axis=-1)
    return points_3d

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

def project_3d_to_2d(points_3d, intrinsic_matrix):
    # Convert 3D points to homogeneous coordinates
    projected_2d_hom = intrinsic_matrix @ points_3d.T
    # Convert from homogeneous coordinates to 2D image coordinates
    projected_2d = projected_2d_hom[:2, :] / projected_2d_hom[2, :]
    return projected_2d.T

def project_3d_to_2d_batch(points_3d, intrinsic_matrix):
    # breakpoint()
    # Convert 3D points to homogeneous coordinates
    projected_2d_hom = intrinsic_matrix @ points_3d.permute(0, 2, 1)
    # Convert from homogeneous coordinates to 2D image coordinates
    projected_2d = projected_2d_hom[:,:2, :] / projected_2d_hom[:,2, :].unsqueeze(1)
    return projected_2d.permute(0, 2, 1)

def compute_optical_flow(depth1, depth2, pose1, pose2, intrinsic_matrix1, intrinsic_matrix2):
    # Input: All inputs as numpy arrays; convert torch tensors to numpy arrays if needed
    if isinstance(depth1, torch.Tensor):
        depth1 = depth1.cpu().numpy()
    if isinstance(depth2, torch.Tensor):
        depth2 = depth2.cpu().numpy()
    if isinstance(pose1, torch.Tensor):
        pose1 = pose1.cpu().numpy()
    if isinstance(pose2, torch.Tensor):
        pose2 = pose2.cpu().numpy()
    if isinstance(intrinsic_matrix1, torch.Tensor):
        intrinsic_matrix1 = intrinsic_matrix1.cpu().numpy()
    if isinstance(intrinsic_matrix2, torch.Tensor):
        intrinsic_matrix2 = intrinsic_matrix2.cpu().numpy()

    points_3d_frame1 = depth_to_3d(depth1, intrinsic_matrix1).reshape(-1, 3)
    points_3d_frame1_hom = np.concatenate([points_3d_frame1, np.ones((points_3d_frame1.shape[0], 1))], axis=1).T
    
    # Calculate the transformation matrix from frame 1 to frame 2
    transformation_matrix = (pose2) @ np.linalg.inv(pose1)
    points_3d_frame2_hom = transformation_matrix @ points_3d_frame1_hom
    points_3d_frame2 = (points_3d_frame2_hom[:3, :]).T

    points_2d_frame1 = project_3d_to_2d(points_3d_frame1, intrinsic_matrix1)
    points_2d_frame2 = project_3d_to_2d(points_3d_frame2, intrinsic_matrix2)
    
    
    points_3d_frame2 = torch.from_numpy(points_3d_frame2).float().unsqueeze(0)
    
    importance = 0.3 / points_3d_frame2[..., 2:3]
    importance -= importance.amin((1, 2), keepdim=True)
    importance /= importance.amax((1, 2), keepdim=True) + 1e-6
    importance = importance * 10 - 10

    # Compute optical flow vectors
    optical_flow = points_2d_frame2 - points_2d_frame1
    optical_flow = torch.from_numpy(optical_flow).float().unsqueeze(0)
    return optical_flow, importance



def compute_optical_flow_batch(depth1, depth2, pose1, pose2, intrinsic_matrix1, intrinsic_matrix2):
    # Input: All inputs as numpy arrays; convert torch tensors to numpy arrays if needed
    # if isinstance(depth1, torch.Tensor):
    #     depth1 = depth1.cpu().numpy()
    # if isinstance(depth2, torch.Tensor):
    #     depth2 = depth2.cpu().numpy()
    # if isinstance(pose1, torch.Tensor):
    #     pose1 = pose1.cpu().numpy()
    # if isinstance(pose2, torch.Tensor):
    #     pose2 = pose2.cpu().numpy()
    # if isinstance(intrinsic_matrix1, torch.Tensor):
    #     intrinsic_matrix1 = intrinsic_matrix1.cpu().numpy()
    # if isinstance(intrinsic_matrix2, torch.Tensor):
    #     intrinsic_matrix2 = intrinsic_matrix2.cpu().numpy()
    pose1 = torch.linalg.inv(pose1)
    pose2 = torch.linalg.inv(pose2)
    B, H, W = depth1.shape
    points_3d_frame1 = depth_to_3d_batch(depth1, intrinsic_matrix1).reshape(B, -1, 3)
    # points_3d_frame1_hom = np.concatenate([points_3d_frame1, np.ones((points_3d_frame1.shape[0], 1))], axis=1).T
    points_3d_frame1_hom = torch.cat([points_3d_frame1, torch.ones((B, points_3d_frame1.shape[1], 1)).to(depth1.device)], dim=-1).permute(0, 2, 1)
    
    # Calculate the transformation matrix from frame 1 to frame 2
    transformation_matrix = (pose2) @ torch.linalg.inv(pose1)
    points_3d_frame2_hom = transformation_matrix @ points_3d_frame1_hom
    points_3d_frame2 = (points_3d_frame2_hom[:,:3, :]).permute(0, 2, 1)

    points_2d_frame1 = project_3d_to_2d_batch(points_3d_frame1, intrinsic_matrix1)
    points_2d_frame2 = project_3d_to_2d_batch(points_3d_frame2, intrinsic_matrix2)
    
    
    # breakpoint()
    # points_3d_frame2 = torch.from_numpy(points_3d_frame2).float().unsqueeze(0)
    importance = 0.3 / points_3d_frame2[..., 2:3]
    importance -= importance.amin((1, 2), keepdim=True)
    importance /= importance.amax((1, 2), keepdim=True) + 1e-6
    importance = importance * 10 - 10

    # Compute optical flow vectors
    optical_flow = points_2d_frame2 - points_2d_frame1
    # optical_flow = torch.from_numpy(optical_flow).float().unsqueeze(0)
    
    optical_flow = rearrange(optical_flow, 'b (h w) c -> b c h w', h=H, w=W)
    importance = rearrange(importance, 'b (h w) c -> b c h w', h=H, w=W)

    return optical_flow, importance


def rgb_to_bgr(tensor):
    breakpoint()
    if tensor.shape[0] == 3:  # Assuming the color channels are the first dimension
        return tensor[[2, 1, 0], :, :]
    elif tensor.shape[-1] == 3:  # Assuming the color channels are the last dimension
        return tensor[..., [2, 1, 0]]
        
    elif tensor.shape[1] == 3:  # Assuming the color channels are the last dimension
        
        return tensor[:, [2, 1, 0]]
    else:
        raise ValueError("Input tensor does not have 3 color channels")

def compute_optical_flow_batch_top3_reference_batch(depth1, depth2, pose1, pose2, intrinsic_matrix1, intrinsic_matrix2, dynamic_masks = None, normal=None, top_k=7):
    # Input: All inputs as numpy arrays; convert torch tensors to numpy arrays if needed
    # if isinstance(depth1, torch.Tensor):
    #     depth1 = depth1.cpu().numpy()
    # if isinstance(depth2, torch.Tensor):
    #     depth2 = depth2.cpu().numpy()
    # if isinstance(pose1, torch.Tensor):
    #     pose1 = pose1.cpu().numpy()
    # if isinstance(pose2, torch.Tensor):
    #     pose2 = pose2.cpu().numpy()
    # if isinstance(intrinsic_matrix1, torch.Tensor):
    #     intrinsic_matrix1 = intrinsic_matrix1.cpu().numpy()
    # if isinstance(intrinsic_matrix2, torch.Tensor):
    #     intrinsic_matrix2 = intrinsic_matrix2.cpu().numpy()
    B, H, W = depth1.shape
    intrinsic_matrix1 = intrinsic_matrix1.unsqueeze(0).repeat(B, 1, 1)
    intrinsic_matrix2 = intrinsic_matrix2.unsqueeze(0).repeat(B, 1, 1)

    points_3d_frame1 = depth_to_3d_batch(depth1, intrinsic_matrix1).reshape(B, -1, 3)
    # points_3d_frame1_hom = np.concatenate([points_3d_frame1, np.ones((points_3d_frame1.shape[0], 1))], axis=1).T
    points_3d_frame1_hom = torch.cat([points_3d_frame1, torch.ones((B, points_3d_frame1.shape[1], 1)).to(depth1.device)], dim=-1).permute(0, 2, 1)
    
    # Calculate the transformation matrix from frame 1 to frame 2
    pose1 = torch.linalg.inv(pose1)
    pose2 = torch.linalg.inv(pose2)
    transformation_matrix = (pose2) @ torch.linalg.inv(pose1)
    points_3d_frame2_hom = transformation_matrix @ points_3d_frame1_hom
    points_3d_frame2 = (points_3d_frame2_hom[:,:3, :]).permute(0, 2, 1)
    
    points_2d_frame1 = project_3d_to_2d_batch(points_3d_frame1, intrinsic_matrix1)
    points_2d_frame2 = project_3d_to_2d_batch(points_3d_frame2, intrinsic_matrix2)
    
    # breakpoint()
    # points_3d_frame2 = torch.from_numpy(points_3d_frame2).float().unsqueeze(0)
    depth_output = rearrange(points_3d_frame2[..., 2:3] , 'b (h w) c -> b c h w', h=H, w=W)
    importance = 0.3 / points_3d_frame2[..., 2:3]
    importance -= importance.amin((1, 2), keepdim=True)
    importance /= importance.amax((1, 2), keepdim=True) + 1e-6
    importance = importance * 10 - 10

    # Compute optical flow vectors
    optical_flow = points_2d_frame2 - points_2d_frame1
    # optical_flow = torch.from_numpy(optical_flow).float().unsqueeze(0)
    
    optical_flow = rearrange(optical_flow, 'b (h w) c -> b c h w', h=H, w=W)
    importance = rearrange(importance, 'b (h w) c -> b c h w', h=H, w=W)
    # transposed_normal = rearrange(transposed_normal, 'b c (h w) -> b c h w', h=H, w=W)
    
    # transposed_normal_co_visible = transposed_normal[:,2:3] > 0
    # save_image(transposed_normal_co_visible.float(), 'transposed_normal_co_visible.png')
    # image_normal = rearrange(normal, 'b c (h w)  -> b c h w', h=H, w=W)
    # save_image((image_normal[:,0:1]>0).float(), 'normal_0.png')
    # save_image((image_normal[:,1:2]>0).float(), 'normal_1.png')
    # save_image((image_normal[:,2:3]>0).float(), 'normal_2.png')
    # save_image(image_normal[0], 'normal.png')
    # 1:2 -> y축 
    
    # calculate valid pixels 
    points2d_dst = points_2d_frame2.round().long() # to int    
    
    depths = points_3d_frame2[..., -1]  # (B, N)
    
    if dynamic_masks is not None:
        dynamic_masks = dynamic_masks.reshape(B,-1)
        valid_pixels_bool = (
                (points_2d_frame2[..., 0] >= 0) & (points_2d_frame2[..., 0] < W) &
                (points_2d_frame2[..., 1] >= 0) & (points_2d_frame2[..., 1] < H) &
                (depths > 0)
            ).squeeze(-1)
        # valid_pixels_bool = (
        #         (points_2d_frame2[..., 0] >= 0) & (points_2d_frame2[..., 0] < W) &
        #         (points_2d_frame2[..., 1] >= 0) & (points_2d_frame2[..., 1] < H) &
        #         (depths > 0) & (dynamic_masks == 0) 
        #     ).squeeze(-1)
    elif normal is not None:
        transposed_normal = transformation_matrix[:,:3,:3] @ normal
        normal = normal.reshape(B, 3, -1)
        valid_pixels_bool = (
                (points_2d_frame2[..., 0] >= 0) & (points_2d_frame2[..., 0] < W) &
                (points_2d_frame2[..., 1] >= 0) & (points_2d_frame2[..., 1] < H) &
                (depths > 0) & (transposed_normal[:,2] > 0)
            ).squeeze(-1)
        
    else:
        valid_pixels_bool = (
                (points_2d_frame2[..., 0] >= 0) & (points_2d_frame2[..., 0] < W) &
                (points_2d_frame2[..., 1] >= 0) & (points_2d_frame2[..., 1] < H) &
                (depths > 0) 
            ).squeeze(-1)
    
    points2d_dst[~valid_pixels_bool] = torch.tensor(
            [[0, 0]], device=points2d_dst.device, dtype=points2d_dst.dtype
        )
    
    depths_sort, depths_ranking = depths.sort(dim=-1, descending=True)
    
    depth_out = torch.zeros(
            B, H+1, W + 1,
            device=depth1.device, dtype=depth1.dtype
        )
    for batch in range(B):
        depth_batch = depth_out[batch]
        pixels_batch = points2d_dst[batch][depths_ranking[batch]]
        horizontals  = pixels_batch[:, 0]
        verticals    = pixels_batch[:, 1]
        depth_batch[verticals, horizontals] = depths_sort[batch]
        
        
    nonzero_counts = (depth_out > 0).sum(dim=(1, 2))  # (B, H, W) -> (B,)
    # 상위 3개의 인덱스 찾기
    top_k_indices = torch.argsort(nonzero_counts, descending=True)[:top_k]
    # 상위 3개 배치 선택
    top_k_batches = depth_out[top_k_indices]  # Shape: (3, 37, 65)
    # save_image(top_k_batches.unsqueeze(dim=1), 'top_k_batches.png')
        # final reshape and return
    # depth_out = depth_out.unsqueeze(1)[..., :-1]  # Bx1xHxW

    # points2d_dst = rearrange(points2d_dst, 'b (h w) c -> b c h w', h=H, w=W)
    valid_pixels_bool = rearrange(valid_pixels_bool, 'b (h w) -> b h w', h=H, w=W ).unsqueeze(dim=1)
    return optical_flow, importance, top_k_indices, depth_output, valid_pixels_bool




def depth_warping_from_key_frames(depth1, depth2, pose1, pose2, intrinsic_matrix1, intrinsic_matrix2, dynamic_masks, images):
    # Input: All inputs as numpy arrays; convert torch tensors to numpy arrays if needed
    # if isinstance(depth1, torch.Tensor):
    #     depth1 = depth1.cpu().numpy()
    # if isinstance(depth2, torch.Tensor):
    #     depth2 = depth2.cpu().numpy()
    # if isinstance(pose1, torch.Tensor):
    #     pose1 = pose1.cpu().numpy()
    # if isinstance(pose2, torch.Tensor):
    #     pose2 = pose2.cpu().numpy()
    # if isinstance(intrinsic_matrix1, torch.Tensor):
    #     intrinsic_matrix1 = intrinsic_matrix1.cpu().numpy()
    # if isinstance(intrinsic_matrix2, torch.Tensor):
    #     intrinsic_matrix2 = intrinsic_matrix2.cpu().numpy()
    # breakpoint()
    B, H, W = depth1.shape
    points_3d_frame1 = depth_to_3d_batch(depth1, intrinsic_matrix1).reshape(B, -1, 3)
    # points_3d_frame1_hom = np.concatenate([points_3d_frame1, np.ones((points_3d_frame1.shape[0], 1))], axis=1).T
    points_3d_frame1_hom = torch.cat([points_3d_frame1, torch.ones((B, points_3d_frame1.shape[1], 1)).to(depth1.device)], dim=-1).permute(0, 2, 1)
    
    # Calculate the transformation matrix from frame 1 to frame 2
    transformation_matrix = (pose2) @ torch.linalg.inv(pose1)
    points_3d_frame2_hom = transformation_matrix @ points_3d_frame1_hom
    points_3d_frame2 = (points_3d_frame2_hom[:,:3, :]).permute(0, 2, 1)
    
    
    # depth1과는 무관
    points_3d_frame2 = points_3d_frame2.reshape(1,-1,3)
    base_coordinate = depth_to_3d_batch(depth1, intrinsic_matrix2).reshape(B, -1, 3)[:1,:,:2]
    
    color_map = torch.zeros((H*W, 3)).to(depth1.device)
    point_list = [[] for _ in range(H*W)]
    
    radius = 0.2
    z_buffer = points_3d_frame2[..., 2:3]
    images = images.permute(0,2,3,1).reshape(-1,3)
    for i in range(0, H*W):
        CheckPixelInsidePoint = ((points_3d_frame2[:,:,:2]- base_coordinate[:,i])**2).sum(dim=-1)**0.5 < radius
        point_list[i].append(torch.nonzero(CheckPixelInsidePoint, as_tuple=True)[1])

    for i in range(0, H*W):
        valid_idx = point_list[i][0]
        
        z_values = z_buffer[0, valid_idx, 0]  # (1, 20) -> (20,)
        # z_buffer 값 기준으로 valid_idx 정렬
        sorted_indices = torch.argsort(z_values, descending=False)  # 오름차순 정렬
        sorted_valid_idx = valid_idx[sorted_indices]

        color_map[i] = images[sorted_valid_idx[0]]
        
    color_map = color_map = rearrange(color_map, '(h w) c -> c h w', h=H, w=W)
    save_image(color_map[[2, 1, 0], :, :].unsqueeze(0)/255, 'color_map.png')
    breakpoint()
    
    return color_map




def get_dynamic_label(base_dir, seq, continuous=False, threshold=13.75, save_dir='dynamic_label'):
    depth_dir = os.path.join(base_dir, 'depth', seq)
    cam_dir = os.path.join(base_dir, 'camdata_left', seq)
    flow_dir = os.path.join(base_dir, 'flow', seq)
    dynamic_label_dir = os.path.join(base_dir, save_dir, seq)
    os.makedirs(dynamic_label_dir, exist_ok=True)
    
    frames = sorted([f for f in os.listdir(depth_dir) if f.endswith('.dpt')])
    for i, frame1 in enumerate(frames):
        if i == len(frames) - 1:
            continue
        frame2 = frames[i + 1]
        
        frame1_id = frame1.split('.')[0]
        frame2_id = frame2.split('.')[0]

        # Load depth maps
        depth_map_frame1 = depth_read(os.path.join(depth_dir, frame1))
        depth_map_frame2 = depth_read(os.path.join(depth_dir, frame2))
        
        # Load camera intrinsics and poses
        intrinsic_matrix1, pose_frame1 = cam_read(os.path.join(cam_dir, f'{frame1_id}.cam'))
        intrinsic_matrix2, pose_frame2 = cam_read(os.path.join(cam_dir, f'{frame2_id}.cam'))
        
        # Pad pose with [0,0,0,1]
        pose_frame1 = np.concatenate([pose_frame1, np.array([[0, 0, 0, 1]])], axis=0)
        pose_frame2 = np.concatenate([pose_frame2, np.array([[0, 0, 0, 1]])], axis=0)
        
        # Compute optical flow
        optical_flow = compute_optical_flow(depth_map_frame1, depth_map_frame2, pose_frame1, pose_frame2, intrinsic_matrix1, intrinsic_matrix2)
        
        # Reshape the optical flow to the image dimensions
        height, width = depth_map_frame1.shape
        optical_flow_image = optical_flow.reshape(height, width, 2)
        
        # Load ground truth optical flow
        u, v = flow_read(os.path.join(flow_dir, f'{frame1_id}.flo'))
        gt_flow = np.stack([u, v], axis=-1)
        
        # Compute the error map
        error_map = np.linalg.norm(gt_flow - optical_flow_image, axis=-1)
        if not continuous:
            binary_error_map = error_map > threshold
            
            # Save the binary error map
            cv2.imwrite(os.path.join(dynamic_label_dir, f'{frame1_id}.png'), binary_error_map.astype(np.uint8) * 255)
        else:
            # Normalize the error map
            error_map = error_map / error_map.max()
            cv2.imwrite(os.path.join(dynamic_label_dir, f'{frame1_id}.png'), (error_map * 255).astype(np.uint8))

if __name__ == '__main__':
    # Process all sequences
    sequences = sorted(os.listdir('data/sintel/training/depth'))
    base_dir = 'data/sintel/training'
    parser = argparse.ArgumentParser()
    parser.add_argument('--continuous', action='store_true')
    parser.add_argument('--threshold', type=float, default=13.75)
    parser.add_argument('--save_dir', type=str, default='dynamic_label')
    args = parser.parse_args()
    for seq in tqdm(sequences):
        get_dynamic_label(base_dir, seq, continuous=args.continuous, threshold=args.threshold, save_dir=args.save_dir)
        print(f'Finished processing {seq}')
