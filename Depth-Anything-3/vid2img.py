import os
from natsort import natsorted
import imageio
import argparse

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run 3D point cloud inference and visualization using ARCroco3DStereo."
    )

    parser.add_argument(
        "--video_path",
        type=str,
        default="",
        help="Path to the directory containing the image sequence.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="./demo_tmp",
        help="value for tempfile.tempdir",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    video_lists = natsorted(os.listdir(args.video_path))
    for video_name in video_lists:
        video_path = os.path.join(args.video_path, video_name)
        video_reader = imageio.get_reader(video_path)
        
        output_folder = os.path.join(args.outdir, os.path.splitext(video_name)[0], 'color')
        os.makedirs(output_folder, exist_ok=True)
        
        for i, frame in enumerate(video_reader):
            frame_path = os.path.join(output_folder, f"{i:06d}.png")
            imageio.imwrite(frame_path, frame)
        
        print(f"Extracted frames from {video_name} to {output_folder}")
        
        
        images = natsorted(os.listdir(output_folder))[-49:]
        
        video_output_path = os.path.join(os.path.dirname(args.outdir), 'video', os.path.splitext(video_name)[0] + ".mp4")
        os.makedirs(os.path.dirname(video_output_path), exist_ok=True)
        
        video_writer = imageio.get_writer(video_output_path, fps=16)
        for img_name in images:
            img_path = os.path.join(output_folder, img_name)
            img = imageio.imread(img_path)
            video_writer.append_data(img)
        video_writer.close()
        print(f"Created video {video_output_path} from extracted frames.")