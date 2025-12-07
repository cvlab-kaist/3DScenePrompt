import numpy as np
import ipdb

def load_data(npz_path, npy_path1, npy_path2):
    # Load the .npz file
    camera = np.load(npz_path)
    
    # Load the .npy files
    conf = np.load(npy_path1)
    depth = np.load(npy_path2)
    
    # Set a breakpoint
    import ipdb; ipdb.set_trace()
    
    return None

# Example usage (replace with actual file paths)
npz_file = "/mnt/DL3DV/DL3DV-10K_output/1K/001dccbc1f78146a9f03861026613d8e73f39f372b545b26118e37a23c740d5f/camera.npz"
npy_file1 = "/mnt/DL3DV/DL3DV-10K_output/1K/001dccbc1f78146a9f03861026613d8e73f39f372b545b26118e37a23c740d5f/conf.npy"
npy_file2 = "/mnt/DL3DV/DL3DV-10K_output/1K/001dccbc1f78146a9f03861026613d8e73f39f372b545b26118e37a23c740d5f/depth.npy"

data = load_data(npz_file, npy_file1, npy_file2)
