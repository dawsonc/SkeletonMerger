"""Predict and save keypoints for all point clouds in a directory."""
import os

import open3d as o3d
import torch
import merger.merger_net as merger_net
import tqdm
import numpy as np
import matplotlib.pyplot as plt


def pc_to_np(pc: o3d.geometry.PointCloud) -> np.ndarray:
    """Convert open3d PointCloud to numpy array."""
    return np.asarray(pc.points)


def get_all_files_in_dir(dir_path: str) -> list:
    """Get all files in a directory."""
    return [os.path.join(dir_path, f) for f in os.listdir(dir_path)]


if __name__ == "__main__":
    # Parameters
    dir_path = "original_pc"  # path to directory of PCD files
    n_pts_per_cloud = 2048  # number of points per point cloud (uniformly sampled)
    device = "cpu"
    model_path = "merger_k10.pt"
    n_keypoints = 10  # should match what was used to train the model

    # Create network from saved weights
    net = merger_net.Net(n_pts_per_cloud, n_keypoints).to(device)
    net.load_state_dict(torch.load(model_path, map_location=torch.device(device))['model_state_dict'])
    net.eval()

    # Get all files in directory
    files = get_all_files_in_dir(dir_path)

    # Load all files, predict keypoints, and save the results
    for file in tqdm.tqdm(files):
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

        # Add a batch dimension
        pcn = pcn[np.newaxis, ...]

        # The model needs at least 2 inputs?
        pcn = np.concatenate([pcn, pcn])

        # Run the model
        with torch.no_grad():
            _, keypoints, _, _, edge_strengths = net(torch.Tensor(pcn).to(device))
        
        # Re-scale keypoints to original point cloud dimensions
        keypoints = 0.5 * keypoints + 0.5
        keypoints = keypoints * (pcmax - pcmin) + pcmin
    
        # Visualize the point cloud
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=1, c='k')
        ax.set_aspect('equal')

        # Visualize the keypoints (also offset by 0.5 to display skeleton alongside)
        ax.scatter(keypoints[0, :, 0], keypoints[0, :, 1], keypoints[0, :, 2], s=15, c='r')
        ax.scatter(keypoints[0, :, 0] + 0.5, keypoints[0, :, 1], keypoints[0, :, 2], s=15, c='r')

        # Visualize the skeleton
        edge_num = 0
        for i in range(n_keypoints):
            keypoint_i = keypoints[0, i, :]
            for j in range(i):
                keypoint_j = keypoints[0, j, :]
                ax.plot(
                    [keypoint_i[0] + 0.5, keypoint_j[0] + 0.5],
                    [keypoint_i[1], keypoint_j[1]],
                    [keypoint_i[2], keypoint_j[2]],
                    c='b',
                    alpha=edge_strengths[0, edge_num].item(),
                    linewidth=3,
                )
                edge_num += 1

        plt.savefig(os.path.join(
            "10_keypoints",
            os.path.basename(file)[:-4] + "_" + model_path[:-3] +  ".png"
        ))
        plt.close()

