import glob, os, torch
from depth_anything_3.api import DepthAnything3
import numpy as np
from natsort import natsorted
import argparse
import time
from tqdm import tqdm

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
    parser.add_argument("--total_batch", type=int, default=-1, help="batch size")
    parser.add_argument("--inverse", action='store_true', help="inverse order")
    parser.add_argument("--shuffle", action='store_true', help="shuffle order")

    return parser.parse_args()



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
        
        
    model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
    model = model.to(device=device)


    out_base_path = args.outdir
    input_base_path = args.base_path
    from natsort import natsorted
    folder_names = [d for d in os.listdir(input_base_path) if os.path.isdir(os.path.join(input_base_path, d))]
    folder_names = natsorted(folder_names)
    
    if args.batch!=  -1:
        data_len = len(folder_names)
        folder_names = folder_names[int(args.batch/args.total_batch* data_len):int((args.batch+1)/args.total_batch* data_len)]
    
    
    if args.inverse:
        folder_names = folder_names[::-1]

    # for _, scene_name in enumerate(folder_names):  
    for _, scene_name in enumerate(tqdm(folder_names, desc="Processing scenes")):
        # try: 
            time1 = time.time()
            output_path = os.path.join(out_base_path, scene_name, 'DA3.npz')
            if os.path.exists(output_path):
                print(f"skip {output_path}")
                continue
            
            images = natsorted(os.listdir(os.path.join(input_base_path, scene_name,'color')))
            images_path = [os.path.join(input_base_path, scene_name,'color',img_name) for img_name in images]
            prediction = model.inference(
                images_path,
            )

            extrinsic_w2c = np.concatenate([prediction.extrinsics, np.tile(np.array([0, 0, 0, 1])[None, None, :], (prediction.extrinsics.shape[0], 1, 1))],axis=1,)


            data = {}
            data['images'] = prediction.processed_images
            data['depths'] = prediction.depth
            data['cam_c2w'] = np.linalg.inv(extrinsic_w2c)
            data['intrinsic'] = prediction.intrinsics

                   
            time2 = time.time()
            print(f"gpu : {args.batch} time taken : {time2-time1}")
            
            np.savez_compressed(output_path, **data)


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)

