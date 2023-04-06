"""Convert PCD files to h5 data files."""
import os

import open3d as o3d
import numpy as np
import h5py
from tqdm import tqdm


def pc_to_np(pc: o3d.geometry.PointCloud) -> np.ndarray:
    """Convert open3d PointCloud to numpy array."""
    return np.asarray(pc.points)


def get_all_files_in_dir(dir_path: str) -> list:
    """Get all files in a directory."""
    return [os.path.join(dir_path, f) for f in os.listdir(dir_path)]


def np_to_formatted_hd5(pc_arrays: np.ndarray, h5_path: str):
    """Write numpy array to h5 file.

    Saves the point cloud under the "data" key and an arbitrary label under the
    "label" key.
    
    Args:
        pc_arrays: numpy array of point cloud data (N_point_clouds, N_points, 3)
        h5_path: path to h5 file
    """
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset("data", data=pc_arrays)
        f.create_dataset("label", data=np.zeros((pc_arrays.shape[0], 1)))


if __name__ == "__main__":
    # Parameters
    train_fraction = 0.75  # fraction of dataset to use for training
    n_pts_per_cloud = 2048  # number of points per point cloud (uniformly sampled)

    # Get all files in directory
    dir_path = "original_pc"
    files = get_all_files_in_dir(dir_path)

    # Convert all files to a single numpy array
    pc_arrays = []
    for file in tqdm(files):
        # Load from PCD file
        pc = o3d.io.read_point_cloud(file)

        # Convert to numpy array
        pc = pc_to_np(pc).astype(np.float32)  # convert to float for PyTorch

        # Uniformly downsample
        pc = pc[np.random.choice(pc.shape[0], n_pts_per_cloud, replace=False)]

        # Normalize
        pcmax = pc.max()
        pcmin = pc.min()
        pcn = (pc - pcmin) / (pcmax - pcmin)
        pcn = 2.0 * (pcn - 0.5)

        # Append to list
        pc_arrays.append(pcn.reshape(1, -1, 3))
    pc_arrays = np.concatenate(pc_arrays, axis=0)
    
    # Sanity check
    assert pc_arrays.ndim == 3
    assert pc_arrays.shape[0] == len(files)
    assert pc_arrays.shape[-1] == 3

    # Split into training/validation
    train_size = int(pc_arrays.shape[0] * train_fraction)
    permuted_indices = np.random.permutation(pc_arrays.shape[0])
    train_indices = permuted_indices[:train_size]
    val_indices = permuted_indices[train_size:]
    train_arrays = pc_arrays[train_indices]
    val_arrays = pc_arrays[val_indices]

    # Convert numpy arrays to h5 file
    os.makedirs("point_cloud/train", exist_ok=True)
    h5_path = "point_cloud/train/training_data.h5"
    np_to_formatted_hd5(train_arrays, h5_path)
    os.makedirs("point_cloud/val", exist_ok=True)
    h5_path = "point_cloud/val/validation_data.h5"
    np_to_formatted_hd5(val_arrays, h5_path)