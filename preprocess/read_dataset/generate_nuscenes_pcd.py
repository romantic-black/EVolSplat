import numpy as np
import os
import imageio.v2 as imageio
import cv2
import open3d as o3d
from typing import Literal
from rich.console import Console

from copy import deepcopy

CONSOLE = Console(width=120)

# Default bounding box for nuScenes (similar to Waymo, can be customized)
X_MIN, X_MAX = -20, 20
Y_MIN, Y_MAX = -20, 4.8
Z_MIN, Z_MAX = -20, 70

class NuScenesPCDGenerator():
    """Point cloud generator for nuScenes dataset."""
    sparsity: Literal['Drop90','Drop50',"Drop80","Drop25","full"] = "full"
    use_bbx: bool = True
    """Accumulate the number of pointcloud frames"""

    def __init__(self, spars='full', save_dir="Drop50", frame_start=0, 
                 filer_sky=True, depth_cosistency=True) -> None:
        self.sparsity = spars
        self.save_dir = save_dir
        self.dir_name = None
        self.frame_start = frame_start
        self.filter_sky = filer_sky
        self.depth_cosistency = depth_cosistency

    def get_bbx(self):
        """Get bounding box for point cloud filtering."""
        return np.array([X_MIN, Y_MIN, Z_MIN]), np.array([X_MAX, Y_MAX, Z_MAX])

    def crop_pointcloud(self, bbx_min, bbx_max, points, color):
        """Crop point cloud to bounding box."""
        mask = (points[:, 0] > bbx_min[0]) & (points[:, 0] < bbx_max[0]) & \
            (points[:, 1] > bbx_min[1]) & (points[:, 1] < bbx_max[1]) & \
            (points[:, 2] > bbx_min[2]) & (points[:, 2] < bbx_max[2] + 50)  # Extended Z for background
        
        return points[mask], color[mask]
    
    def split_pointcloud(self, bbx_min, bbx_max, points, color):
        """Split point cloud into inside and outside bounding box."""
        mask = (points[:, 0] > bbx_min[0]) & (points[:, 0] < bbx_max[0]) & \
            (points[:, 1] > bbx_min[1]) & (points[:, 1] < bbx_max[1]) & \
            (points[:, 2] > bbx_min[2]) & (points[:, 2] < bbx_max[2])
        
        inside_pnt, inside_rgb = points[mask], color[mask]
        outside_pnt, outside_rgb = points[~mask], color[~mask]
        return inside_pnt, inside_rgb, outside_pnt, outside_rgb

    def forward(self, dir_name, poses, intrinsics, H, W, down_scale=2):
        """
        Generate point cloud from depth maps.
        
        Args:
            dir_name: Directory containing images and depth maps
            poses: Camera-to-world poses (num_frames * num_cameras, 4, 4)
            intrinsics: Camera intrinsics (num_frames * num_cameras, 4, 4)
            H: Image height
            W: Image width
            down_scale: Downsampling scale for point cloud generation
        """
        self.dir_name = dir_name
        self.depth_dir = os.path.join(self.dir_name, 'depth')
        self.H, self.W = H, W

        if not os.path.exists(self.depth_dir):
            raise ValueError(f"Depth directory not found: {self.depth_dir}")

        depth_files = []
        selected_index = []
        self.c2w = []
        self.intri = []
        
        # Process different sparsity levels
        all_depth_files = sorted([f for f in os.listdir(self.depth_dir) if f.endswith('.npy')])
        
        # Depth files should match poses/intrinsics in order
        # Both are ordered as: frame0_cam0, frame0_cam1, ..., frame1_cam0, ...
        # We filter by frame, not by individual file
        # Group files by frame index for sparsity filtering
        frame_groups = {}
        for file_name in all_depth_files:
            try:
                # Parse frame index from filename: {frame_idx:03d}_{cam_id}.npy
                frame_idx = int(file_name.split('_')[0])
                if frame_idx not in frame_groups:
                    frame_groups[frame_idx] = []
                frame_groups[frame_idx].append(file_name)
            except (ValueError, IndexError):
                CONSOLE.log(f"Warning: Cannot parse filename {file_name}, skipping")
                continue
        
        # Sort frame indices and apply sparsity filtering
        sorted_frame_indices = sorted(frame_groups.keys())
        selected_frames = []
        for frame_pos, frame_idx in enumerate(sorted_frame_indices):
            # Apply sparsity filtering
            if self.sparsity == "Drop50":
                if frame_pos % 4 == 2 or frame_pos % 4 == 3:
                    continue
            elif self.sparsity == 'Drop80':
                if frame_pos % 5 != 0:  # Keep 20% of frames
                    continue 
            elif self.sparsity == 'Drop25':
                if frame_pos % 4 == 2:
                    continue 
            elif self.sparsity == 'Drop90':
                if frame_pos % 10 != 0:
                    continue
            
            selected_frames.append(frame_idx)
        
        # Collect depth files for selected frames (sorted by frame_idx then cam_id)
        for frame_idx in selected_frames:
            depth_files.extend(sorted(frame_groups[frame_idx]))
        
        # Create mapping from (frame_idx, cam_id) to pose/intrinsic index
        # This assumes poses/intrinsics are in the same order as images: frame0_cam0, frame0_cam1, ..., frame1_cam0, ...
        # We need to parse image filenames from the images directory to build the mapping
        images_dir = os.path.join(self.dir_name, 'images')
        if not os.path.exists(images_dir):
            raise ValueError(f"Images directory not found: {images_dir}")
        image_files = sorted([f for f in os.listdir(images_dir) 
                             if f.endswith('.jpg') or f.endswith('.png')])
        
        # Build mapping: (frame_idx, cam_id) -> index in poses/intrinsics
        # Note: poses/intrinsics are ordered as: frame0_cam0, frame0_cam1, ..., frame1_cam0, ...
        # So we need to sort by frame_idx first, then by cam_id
        pose_map = {}
        # Sort image files by frame_idx and cam_id to match poses order
        sorted_image_files = sorted(image_files, key=lambda x: (
            int(x.split('_')[0]) if x.split('_')[0].isdigit() else 999999,
            int(x.split('_')[1].split('.')[0]) if len(x.split('_')) > 1 and x.split('_')[1].split('.')[0].isdigit() else 999999
        ))
        for idx, img_file in enumerate(sorted_image_files):
            try:
                # Image naming: {frame_idx:03d}_{cam_id}.jpg or .png
                parts = img_file.replace('.jpg', '').replace('.png', '').split('_')
                if len(parts) >= 2:
                    frame_idx = int(parts[0])
                    cam_id = int(parts[1])
                    pose_map[(frame_idx, cam_id)] = idx
            except (ValueError, IndexError):
                continue
        
        # Match depth files with poses/intrinsics using filename
        valid_depth_files = []
        for file_name in depth_files:
            try:
                # Parse frame_idx and cam_id from depth filename
                parts = file_name.replace('.npy', '').split('_')
                if len(parts) >= 2:
                    frame_idx = int(parts[0])
                    cam_id = int(parts[1])
                    if (frame_idx, cam_id) in pose_map:
                        idx = pose_map[(frame_idx, cam_id)]
                        if idx < len(poses):
                            valid_depth_files.append(file_name)
                            # Use poses directly without coordinate conversion
                            # (poses are already aligned and in correct coordinate system)
                            self.c2w.append(poses[idx])
                            self.intri.append(intrinsics[idx])
                            selected_index.append(idx)
                        else:
                            CONSOLE.log(f"Warning: Pose index {idx} out of range for {file_name}")
                    else:
                        CONSOLE.log(f"Warning: No pose found for {file_name}")
                else:
                    CONSOLE.log(f"Warning: Cannot parse depth filename {file_name}")
            except (ValueError, IndexError) as e:
                CONSOLE.log(f"Warning: Error parsing {file_name}: {e}")
        
        depth_files = valid_depth_files

        print(f"{self.sparsity} :ALL frames: {len(depth_files)}:, selected indices: {selected_index[:10]}... \n")
        
        if len(depth_files) == 0:
            raise ValueError("No depth files selected after sparsity filtering")
        
        if len(self.c2w) != len(depth_files):
            raise ValueError(f"Mismatch: {len(self.c2w)} poses for {len(depth_files)} depth files")

        # Depth consistency check
        if self.depth_cosistency:
            consistency_mask = self.depth_cosistency_check(depth_files=depth_files)
        else:
            consistency_mask = [np.ones((H, W)) for _ in range(len(depth_files))]

        # Accumulate point cloud
        accumulated_pointcloud = self.accumulat_pcd(
            depth_files=depth_files, 
            consistency_mask=consistency_mask, 
            down_scale=down_scale
        )

        # Output and save the .ply pointcloud
        os.makedirs(os.path.join(self.dir_name, self.save_dir), exist_ok=True)

        points = accumulated_pointcloud[:, :3]
        colors = accumulated_pointcloud[:, 3:]

        if self.use_bbx:
            bbx_min, bbx_max = self.get_bbx()
            print(f"BBX Range: {bbx_min},{bbx_max} \n")
            points, colors = self.crop_pointcloud(bbx_min, bbx_max, points, colors)
            inside_pnt, inside_rgb, outside_pnt, outside_rgb = self.split_pointcloud(
                bbx_min, bbx_max, points, colors
            )

            # Inside filter
            inside_pointcloud = o3d.geometry.PointCloud()
            inside_pointcloud.points = o3d.utility.Vector3dVector(inside_pnt[:, :3])
            inside_pointcloud.colors = o3d.utility.Vector3dVector(inside_rgb)
            cl, ind = inside_pointcloud.remove_statistical_outlier(
                nb_neighbors=35, std_ratio=1.5
            )
            inside_pointcloud = inside_pointcloud.select_by_index(ind)

            # Outside filter
            outside_pointcloud = o3d.geometry.PointCloud()
            outside_pointcloud.points = o3d.utility.Vector3dVector(outside_pnt[:, :3])
            outside_pointcloud.colors = o3d.utility.Vector3dVector(outside_rgb)
            cl, ind = outside_pointcloud.remove_statistical_outlier(
                nb_neighbors=20, std_ratio=2.0
            )
            outside_pointcloud = outside_pointcloud.select_by_index(ind)

            combined_pointcloud = inside_pointcloud + outside_pointcloud
            combined_pointcloud = combined_pointcloud.uniform_down_sample(every_k_points=2)
            
            o3d.io.write_point_cloud(
                os.path.join(self.dir_name, self.save_dir, f'{self.frame_start}.ply'), 
                combined_pointcloud
            )
            print(f"Save the pointcloud in {os.path.join(self.dir_name, self.save_dir)} !")
        else:
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(points[:, :3])
            point_cloud.colors = o3d.utility.Vector3dVector(colors)

            # Filter the noisy pointcloud and downsample the pcd in 3D space
            cl, ind = point_cloud.remove_statistical_outlier(
                nb_neighbors=30, std_ratio=1.5
            )
            point_cloud = point_cloud.select_by_index(ind)
            point_cloud = point_cloud.uniform_down_sample(every_k_points=3)
            
            o3d.io.write_point_cloud(
                os.path.join(self.dir_name, self.save_dir, f'{self.frame_start}.ply'), 
                point_cloud
            )
            print(f"Save the pointcloud in {os.path.join(self.dir_name, self.save_dir)} !")

    def accumulat_pcd(self, depth_files, consistency_mask, down_scale: int = 2):
        """Accumulate point cloud from depth files."""
        color_pointclouds = []

        if down_scale != 1:
            downscale_mask = np.zeros_like(consistency_mask[0])
            downscale_mask[::down_scale, ::down_scale] = 1

        for i, file_name in enumerate(depth_files):
            depth_file = os.path.join(self.depth_dir, file_name)
            depth = np.load(depth_file)
            
            # Find corresponding RGB image
            # Depth file: {frame_idx:03d}_{cam_id}.npy
            # RGB file: images/{frame_idx:03d}_{cam_id}.jpg
            rgb_file = os.path.join(self.dir_name, 'images', file_name.replace('.npy', '.jpg'))
            if not os.path.exists(rgb_file):
                rgb_file = os.path.join(self.dir_name, 'images', file_name.replace('.npy', '.png'))
            
            if not os.path.exists(rgb_file):
                CONSOLE.log(f"Warning: RGB file not found for {file_name}")
                continue
                
            rgb = imageio.imread(rgb_file) / 255.0

            # Apply sky filtering if enabled (use sky_masks instead of semantic/instance)
            if self.filter_sky:
                sky_mask_file = os.path.join(
                    self.dir_name, 'sky_masks', 
                    file_name.replace('.npy', '.png')
                )
                if os.path.exists(sky_mask_file):
                    sky_mask = cv2.imread(sky_mask_file, cv2.IMREAD_GRAYSCALE)
                    # sky_masks: 0 = sky, 255 = non-sky
                    # Convert to boolean: True = non-sky (keep), False = sky (filter)
                    mask = (sky_mask > 0).astype(np.bool_)
                    final_mask = np.logical_and(consistency_mask[i], mask)
                else:
                    CONSOLE.log(f"Warning: Sky mask not found: {sky_mask_file}, skipping sky filtering")
                    final_mask = consistency_mask[i]
            else:
                final_mask = consistency_mask[i]

            # Downsample the 2D image to generate the pointcloud
            if down_scale != 1:
                final_mask = np.logical_and(downscale_mask, final_mask)

            kept = np.argwhere(final_mask)

            if len(kept) == 0:
                continue

            depth_values = depth[kept[:, 0], kept[:, 1]]
            rgb_values = rgb[kept[:, 0], kept[:, 1]]
            
            c2w = self.c2w[i]
            K = self.intri[i]

            # Generate pixel coordinates
            x = np.arange(0, self.W)
            y = np.arange(0, self.H)
            xx, yy = np.meshgrid(x, y)
            pixels = np.vstack((xx.ravel(), yy.ravel())).T.reshape(self.H, self.W, 2)

            # Unproject depth to 3D points
            pixel_coords = pixels[kept[:, 0], kept[:, 1]]
            x_cam = (pixel_coords[:, 0] - K[0, 2]) * depth_values / K[0, 0]
            y_cam = (pixel_coords[:, 1] - K[1, 2]) * depth_values / K[1, 1]
            z_cam = depth_values
            coordinates = np.stack([x_cam, y_cam, z_cam], axis=1)
            coordinates = np.column_stack((coordinates, np.ones(len(coordinates))))

            # Transform to world coordinates
            worlds = np.dot(c2w, coordinates.T).T
            worlds = worlds[:, :3]

            color_pointclouds.append(np.concatenate([worlds, rgb_values.reshape(-1, 3)], axis=-1))

        if len(color_pointclouds) == 0:
            raise ValueError("No valid point cloud generated")
            
        point_clouds = np.concatenate(color_pointclouds, axis=0).reshape(-1, 6)
        return point_clouds

    def depth_cosistency_check(self, depth_files):
        """Check depth consistency between consecutive frames."""
        depth_masks = []
        print("Depth Consistency Check!")
        
        for i, file_name in enumerate(depth_files):
            depth_file = os.path.join(self.depth_dir, file_name)
            depth = np.load(depth_file)

            # Assume the first depth frame is correct
            if i == 0:
                last_depth = deepcopy(depth)
                depth_masks.append(np.ones((self.H, self.W)))
                continue

            c2w = self.c2w[i]
            last_c2w = self.c2w[i-1]
            K = self.intri[i]

            # Unproject pointcloud
            x = np.arange(0, depth.shape[1])
            y = np.arange(0, depth.shape[0])
            xx, yy = np.meshgrid(x, y)
            pixels = np.vstack((xx.ravel(), yy.ravel())).T.reshape(-1, 2)

            # Unproject depth to pointcloud
            cx, cy = K[0, 2], K[1, 2]
            fx, fy = K[0, 0], K[1, 1]
            
            x_cam = (pixels[..., 0] - cx) * depth.reshape(-1) / fx
            y_cam = (pixels[..., 1] - cy) * depth.reshape(-1) / fy
            z_cam = depth.reshape(-1)
            coordinates = np.stack([x_cam, y_cam, z_cam], axis=1)

            depth_mask = self.depth_projection_check(
                coordinates=coordinates, pixels=pixels, 
                last_c2w=last_c2w, c2w=c2w, 
                last_depth=last_depth, depth=depth, K=K
            )
            depth_masks.append(depth_mask)

            # Update status
            last_depth = deepcopy(depth)

        return depth_masks

    def depth_projection_check(self, coordinates, pixels, last_c2w, c2w, last_depth, depth, K):
        """Check depth consistency by projecting current points to previous frame."""
        H, W = last_depth.shape[:2]
        cx, cy = K[0, 2], K[1, 2]
        fx, fy = K[0, 0], K[1, 1]

        trans_mat = np.dot(np.linalg.inv(last_c2w), c2w)
        coordinates_homo = np.column_stack((coordinates.reshape(-1, 3), np.ones(len(coordinates))))
        last_coordinates = np.dot(trans_mat, coordinates_homo.T).T
        
        # Project to previous frame
        last_x = (fx * last_coordinates[:, 0] + cx * last_coordinates[:, 2]) / last_coordinates[:, 2]
        last_y = (fy * last_coordinates[:, 1] + cy * last_coordinates[:, 2]) / last_coordinates[:, 2]
        last_pixels = np.vstack((last_x, last_y)).T.reshape(-1, 2).astype(np.int32)

        # Swap pixel coordinates (row, col) <-> (x, y)
        pixels_swapped = pixels.copy()
        pixels_swapped[:, [0, 1]] = pixels_swapped[:, [1, 0]]
        last_pixels_swapped = last_pixels.copy()
        last_pixels_swapped[:, [0, 1]] = last_pixels_swapped[:, [1, 0]]

        depth_mask = np.ones(depth.shape[0] * depth.shape[1])

        # Reprojection location must be in image plane [0,H] [0,W]
        valid_mask_00 = np.logical_and((last_pixels_swapped[:, 0] < H), (last_pixels_swapped[:, 1] < W))
        valid_mask_01 = np.logical_and((last_pixels_swapped[:, 0] > 0), (last_pixels_swapped[:, 1] > 0))
        valid_mask = np.logical_and(valid_mask_00, valid_mask_01)

        depth_diff = np.abs(
            depth[pixels_swapped[valid_mask, 0], pixels_swapped[valid_mask, 1]] - 
            last_depth[last_pixels_swapped[valid_mask, 0], last_pixels_swapped[valid_mask, 1]]
        )
        depth_mask[valid_mask] = np.where(depth_diff < depth_diff.mean(), 1, 0)
        depth_mask = depth_mask.reshape(*depth.shape)

        return depth_mask

