import numpy as np
import os
import imageio.v2 as imageio
import json
import cv2
from tqdm import tqdm
from rich.console import Console

CONSOLE = Console(width=120)

class ReadNuScenesData():
    def __init__(self, save_dir, sequence, root_dir, cameras=[0]):
        """
        Initialize NuScenes data reader.
        
        Args:
            save_dir: Directory to save processed data
            sequence: Scene/sequence ID (scene index)
            root_dir: Root directory containing processed nuScenes data
            cameras: List of camera IDs to use (0-5, default=[0] for CAM_FRONT)
        """
        self.sequence = sequence
        self.save_root = save_dir
        self.data_root = root_dir
        self.inner_stop = False
        self.dir_name = None
        self.cameras = cameras  # List of camera IDs [0,1,2,3,4,5]
        
        # Camera names mapping
        self.cam_names = {
            0: 'CAM_FRONT',
            1: 'CAM_FRONT_LEFT',
            2: 'CAM_FRONT_RIGHT',
            3: 'CAM_BACK_LEFT',
            4: 'CAM_BACK_RIGHT',
            5: 'CAM_BACK'
        }

    def generate_json(self, frame_start=0, num_frames=40):
        """
        Generate transforms.json and return poses, intrinsics, dir_name, info.
        
        Args:
            frame_start: Starting frame index
            num_frames: Number of frames to process
            
        Returns:
            poses: List of poses (one per frame per camera) or dict structure
            intrinsics: List of intrinsics (one per frame per camera)
            dir_name: Output directory name
            info: Tuple of (H, W) or camera-specific info
        """
        # Construct scene path: assume root_dir contains processed data with scene folders
        # Format: {root_dir}/processed*/{split}/{scene_id:03d}/
        scene_path = None
        if os.path.exists(os.path.join(self.data_root, f'{self.sequence:03d}')):
            scene_path = os.path.join(self.data_root, f'{self.sequence:03d}')
        else:
            # Try to find in subdirectories
            for subdir in os.listdir(self.data_root):
                scene_candidate = os.path.join(self.data_root, subdir, f'{self.sequence:03d}')
                if os.path.exists(scene_candidate):
                    scene_path = scene_candidate
                    break
        
        if scene_path is None:
            raise ValueError(f"Scene {self.sequence:03d} not found in {self.data_root}")
        
        self.dir_name = os.path.join(self.save_root, 
                                     f'seq_{self.sequence:03d}_nuscenes_{frame_start:04d}_{num_frames}')
        os.makedirs(self.dir_name, exist_ok=True)
        CONSOLE.log(f"Create the BaseDir at {self.dir_name} ! \n")
        
        # Read data from processed nuScenes format
        poses, intrinsics, info = self.read_data(scene_path, frame_start, num_frames)
        return poses, intrinsics, self.dir_name, info
    
    def pose_normalization(self, poses):
        """
        Normalize poses using midpoint frame as origin.
        Similar to Waymo implementation.
        """
        camera_type_matrix = np.array([1, -1, -1])
        
        mid_frames = poses.shape[0] // 2
        inv_pose = np.linalg.inv(poses[mid_frames])
        for i, pose in enumerate(poses):
            if i == mid_frames:
                poses[i] = np.eye(4)
            else:
                poses[i] = np.dot(inv_pose, poses[i])  # Note: inv_pose is left-multiplied
        
        # Apply OpenCV to OpenGL coordinate system conversion
        for i in range(poses.shape[0]):
            poses[i, :3, :3] = poses[i, :3, :3] * camera_type_matrix
        
        return poses, inv_pose
    
    def read_data(self, scene_path, frame_start, num_frames):
        """
        Read images, poses, and intrinsics from processed nuScenes data.
        
        Args:
            scene_path: Path to scene directory
            frame_start: Starting frame index
            num_frames: Number of frames to read
            
        Returns:
            poses: Stacked poses array (num_frames * num_cameras, 4, 4)
            intrinsics: Stacked intrinsics array (num_frames * num_cameras, 4, 4)
            info: Tuple (H, W) - image dimensions
        """
        images_dir = os.path.join(scene_path, 'images')
        extrinsics_dir = os.path.join(scene_path, 'extrinsics')
        intrinsics_dir = os.path.join(scene_path, 'intrinsics')
        
        all_images = []
        all_poses = []
        all_intrinsics = []
        all_img_names = []
        
        # Load first camera's first frame pose for alignment
        camera_front_start = None
        if len(self.cameras) > 0:
            first_cam = self.cameras[0]
            first_extrinsic_file = os.path.join(extrinsics_dir, f'{frame_start:03d}_{first_cam}.txt')
            if os.path.exists(first_extrinsic_file):
                camera_front_start = np.loadtxt(first_extrinsic_file)
            else:
                # Fallback: use frame 0 if frame_start doesn't exist
                fallback_file = os.path.join(extrinsics_dir, f'000_{first_cam}.txt')
                if os.path.exists(fallback_file):
                    camera_front_start = np.loadtxt(fallback_file)
                    CONSOLE.log(f"Warning: Using frame 0 for alignment instead of {frame_start}")
        
        # Read frames
        for frame_idx in tqdm(range(frame_start, frame_start + num_frames)):
            for cam_id in self.cameras:
                # Load image
                img_path = os.path.join(images_dir, f'{frame_idx:03d}_{cam_id}.jpg')
                if not os.path.exists(img_path):
                    CONSOLE.log(f"Warning: Image not found: {img_path}")
                    continue
                
                # Check if all required files exist before processing
                intrinsic_file = os.path.join(intrinsics_dir, f'{cam_id}.txt')
                extrinsic_file = os.path.join(extrinsics_dir, f'{frame_idx:03d}_{cam_id}.txt')
                
                if not os.path.exists(intrinsic_file):
                    CONSOLE.log(f"Warning: Intrinsic file not found: {intrinsic_file}, skipping")
                    continue
                if not os.path.exists(extrinsic_file):
                    CONSOLE.log(f"Warning: Extrinsic file not found: {extrinsic_file}, skipping")
                    continue
                
                img = imageio.imread(img_path)
                all_images.append(img)
                
                # Load intrinsics (fixed per camera)
                intrinsic_data = np.loadtxt(intrinsic_file)
                fx, fy, cx, cy = intrinsic_data[0], intrinsic_data[1], intrinsic_data[2], intrinsic_data[3]
                
                # Convert to 4x4 matrix (for compatibility with Waymo format)
                cam_intrinsic = np.eye(4)
                cam_intrinsic[0, 0] = fx
                cam_intrinsic[1, 1] = fy
                cam_intrinsic[0, 2] = cx
                cam_intrinsic[1, 2] = cy
                all_intrinsics.append(cam_intrinsic)
                
                # Load extrinsics (cam_to_world)
                cam2world = np.loadtxt(extrinsic_file)
                
                # Align with first camera's first frame (if available)
                if camera_front_start is not None:
                    cam2world = np.linalg.inv(camera_front_start) @ cam2world
                
                # Note: Extrinsics from nuscenes_preprocess.py are already cam_to_world
                # The coordinate conversion is handled in pose_normalization
                # So we don't apply OPENCV2DATASET here
                
                all_poses.append(cam2world)
                all_img_names.append(f'{frame_idx:03d}_{cam_id}')
        
        if len(all_images) == 0:
            raise ValueError(f"No images found in range {frame_start} to {frame_start + num_frames}")
        
        # Stack images
        imgs = np.stack(all_images, axis=0)
        poses = np.stack(all_poses)
        intrinsics = np.stack(all_intrinsics)
        
        # Normalize poses
        poses, inv_pose = self.pose_normalization(poses)
        
        # Get image dimensions
        H, W = imgs[0].shape[0], imgs[0].shape[1]
        
        # Generate transforms.json
        def listify_matrix(matrix):
            matrix_list = []
            for row in matrix:
                matrix_list.append(list(row))
            return matrix_list
        
        # Use first frame's intrinsics for JSON metadata (assuming same intrinsics per camera)
        first_intrinsic = intrinsics[0]
        out_data = {
            'fl_x': float(first_intrinsic[0, 0]),
            'fl_y': float(first_intrinsic[1, 1]),
            'cx': float(first_intrinsic[0, 2]),
            'cy': float(first_intrinsic[1, 2]),
            'w': W,
            'h': H,
            'inv_pose': listify_matrix(inv_pose),
            'scale': 1,
            'ply_file_path_Drop25': f"Drop25/{frame_start}.ply",
            'ply_file_path_Drop50': f"Drop50/{frame_start}.ply",
            'ply_file_path_full': f"full/{frame_start}.ply",
            'ply_file_path_Drop80': f"Drop80/{frame_start}.ply",
            'ply_file_path_Drop90': f"Drop90/{frame_start}.ply",
        }
        out_data['frames'] = []
        
        # Save images and create frame entries
        for i in range(poses.shape[0]):
            frame_data = {
                'file_path': f'./{all_img_names[i]}.jpg',
                'transform_matrix': listify_matrix(poses[i]),
                'intrinsics': intrinsics[i].tolist(),
            }
            out_data['frames'].append(frame_data)
            
            # Save image
            filename = os.path.join(self.dir_name, f'{all_img_names[i]}.jpg')
            imageio.imwrite(filename, imgs[i])
        
        # Save transforms.json
        with open(os.path.join(self.dir_name, 'transforms.json'), 'w') as out_file:
            json.dump(out_data, out_file, indent=4)
        
        return poses, intrinsics, (H, W)

