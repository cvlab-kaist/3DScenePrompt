import numpy as np

from sklearn.cluster import DBSCAN

def sample_points_from_masks(masks, num_points):
    """
    sample points from masks and return its absolute coordinates

    Args:
        masks: np.array with shape (n, h, w)
        num_points: int

    Returns:
        points: np.array with shape (n, points, 2)
    """
    n, h, w = masks.shape
    points = []

    for i in range(n):
        # find the valid mask points
        indices = np.argwhere(masks[i] == 1)  
        # the output format of np.argwhere is (y, x) and the shape is (num_points, 2)
        # we should convert it to (x, y)
        indices = indices[:, ::-1]  # (num_points, [y x]) to (num_points, [x y])
        
        # import pdb; pdb.set_trace()
        if len(indices) == 0:
            # if there are no valid points, append an empty array
            points.append(np.array([]))
            continue
        
        # resampling if there's not enough points
        if len(indices) < num_points:
            sampled_indices = np.random.choice(len(indices), num_points, replace=True)
        else:
            sampled_indices = np.random.choice(len(indices), num_points, replace=False)
        
        sampled_points = indices[sampled_indices]
        points.append(sampled_points)

    # convert to np.array
    points = np.array(points, dtype=np.float32)
    return points



def sample_points_from_masks_with_dbscan(masks, num_points, eps=5, min_samples=5):
    """
    Sample points from masks using DBSCAN clustering and return their absolute coordinates.

    Args:
        masks: np.array with shape (n, h, w)
        num_points: int
        eps: float, DBSCAN eps parameter
        min_samples: int, DBSCAN min_samples parameter

    Returns:
        points: np.array with shape (n, num_points, 2)
    """
    n, h, w = masks.shape
    all_sampled_points = []

    for i in range(n):
        indices = np.argwhere(masks[i] == 1)
        indices = indices[:, ::-1]  # convert (y, x) -> (x, y)

        if len(indices) == 0:
            all_sampled_points.append(np.zeros((num_points, 2), dtype=np.float32))
            continue

        # Run DBSCAN
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(indices)
        labels = clustering.labels_

        if np.all(labels == -1):
            # No clusters found, fallback to uniform sampling
            if len(indices) < num_points:
                sampled_indices = np.random.choice(len(indices), num_points, replace=True)
            else:
                sampled_indices = np.random.choice(len(indices), num_points, replace=False)
            sampled_points = indices[sampled_indices]
        else:
            # Select points from the largest cluster
            unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
            main_label = unique_labels[np.argmax(counts)]
            main_cluster = indices[labels == main_label]

            if len(main_cluster) < num_points:
                sampled_indices = np.random.choice(len(main_cluster), num_points, replace=True)
            else:
                sampled_indices = np.random.choice(len(main_cluster), num_points, replace=False)
            sampled_points = main_cluster[sampled_indices]

        all_sampled_points.append(sampled_points.astype(np.float32))

    return np.stack(all_sampled_points)





def sample_points_from_all_dbscan_clusters(masks, num_points_per_cluster=10, eps=3, min_samples=3):
    """
    Sample points from each DBSCAN cluster in the masks and assign them distinct classes.

    Returns:
        all_points: List of np.arrays with shape (num_clusters, num_points, 2)
        all_labels: List of np.arrays with shape (num_clusters,) indicating cluster class id
    """
    n, h, w = masks.shape
    all_points = []
    all_labels = []

    for i in range(n):
        indices = np.argwhere(masks[i] == 1)
        indices = indices[:, ::-1]  # (y, x) -> (x, y)

        if len(indices) == 0:
            all_points.append(np.zeros((0, num_points_per_cluster, 2), dtype=np.float32))
            all_labels.append(np.zeros((0,), dtype=np.int32))
            continue

        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(indices)
        labels = clustering.labels_

        clusters = []
        cluster_ids = []

        for cluster_id in np.unique(labels):
            if cluster_id == -1:
                continue  # skip noise

            cluster_points = indices[labels == cluster_id]

            if len(cluster_points) < num_points_per_cluster:
                sampled = cluster_points[np.random.choice(len(cluster_points), num_points_per_cluster, replace=True)]
            else:
                sampled = cluster_points[np.random.choice(len(cluster_points), num_points_per_cluster, replace=False)]

            clusters.append(sampled.astype(np.float32))
            cluster_ids.append(cluster_id)

        if clusters:
            all_points.append(np.stack(clusters))
            all_labels.append(np.array(cluster_ids))
        else:
            all_points.append(np.zeros((0, num_points_per_cluster, 2), dtype=np.float32))
            all_labels.append(np.zeros((0,), dtype=np.int32))

    return all_points, all_labels


from scipy.ndimage import distance_transform_edt
def sample_points_from_all_dbscan_clusters_w_margin(
        masks: np.ndarray,
        num_points_per_cluster: int = 10,
        eps: float = 3,
        min_samples: int = 3,
        margin: int = 10
    ):
    """
    Sample points from each DBSCAN cluster in the masks *after* shrinking the mask
    by `margin` pixels.

    Args:
        masks: np.ndarray of shape (N, H, W), binary masks (0 or 1)
        num_points_per_cluster: how many points to sample per cluster
        eps, min_samples: DBSCAN parameters
        margin: number of pixels to erode (shrink) the mask border inward

    Returns:
        all_points: List of np.ndarray with shape (num_clusters_i, num_points_per_cluster, 2)
        all_labels: List of np.ndarray with shape (num_clusters_i,) indicating cluster IDs
    """
    n, h, w = masks.shape
    all_points = []
    all_labels = []

    for i in range(n):
        mask = masks[i].astype(bool)
        if not mask.any():
            all_points.append(np.zeros((0, num_points_per_cluster, 2), dtype=np.float32))
            all_labels.append(np.zeros((0,), dtype=np.int32))
            continue

        # 1) Shrink the mask by margin pixels
        dist_map = distance_transform_edt(mask)
        eroded_mask = dist_map > margin  # keep only pixels at least `margin` inside

        if not eroded_mask.any():
            # After erosion, nothing remains
            all_points.append(np.zeros((0, num_points_per_cluster, 2), dtype=np.float32))
            all_labels.append(np.zeros((0,), dtype=np.int32))
            continue

        # 2) Extract coordinates from the eroded mask
        coords = np.argwhere(eroded_mask)[:, ::-1]  # (x, y)

        # 3) DBSCAN on eroded-mask coords
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
        labels = clustering.labels_

        clusters = []
        cluster_ids = []

        # 4) For each valid cluster, sample points
        for cluster_id in np.unique(labels):
            if cluster_id == -1:
                continue  # skip noise

            pts = coords[labels == cluster_id]  # (M, 2)

            # Sample with/without replacement
            if len(pts) >= num_points_per_cluster:
                choice = np.random.choice(len(pts), num_points_per_cluster, replace=False)
            else:
                choice = np.random.choice(len(pts), num_points_per_cluster, replace=True)

            sampled = pts[choice].astype(np.float32)  # (num_points_per_cluster, 2)
            clusters.append(sampled)
            cluster_ids.append(cluster_id)

        # 5) Append results (or empty placeholders)
        if clusters:
            all_points.append(np.stack(clusters, axis=0))       # (num_clusters_i, num_points_per_cluster, 2)
            all_labels.append(np.array(cluster_ids, dtype=np.int32))
        else:
            all_points.append(np.zeros((0, num_points_per_cluster, 2), dtype=np.float32))
            all_labels.append(np.zeros((0,), dtype=np.int32))

    return all_points, all_labels