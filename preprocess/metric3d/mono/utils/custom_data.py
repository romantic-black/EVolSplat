import glob
import os
import json
from pathlib import Path
import numpy as np

def load_from_annos(anno_path):
    with open(anno_path, 'r') as f:
        annos = json.load(f)['files']

    datas = []
    for i, anno in enumerate(annos):
        rgb = anno['rgb']
        depth = anno['depth'] if 'depth' in anno else None
        depth_scale = anno['depth_scale'] if 'depth_scale' in anno else 1.0
        intrinsic = anno['cam_in'] if 'cam_in' in anno else None

        data_i = {
            'rgb': rgb,
            'depth': depth,
            'depth_scale': depth_scale,
            'intrinsic': intrinsic,
            'filename': os.path.basename(rgb),
            'folder': rgb.split('/')[-3],
        }
        datas.append(data_i)
    return datas

def load_from_json(filename: Path):
    with open(filename, encoding="UTF-8") as file:
        return json.load(file)
    
def load_intrinsics(path:str):
    " Return intrinsics list [fl_x, fl_x, cx, cy] "
    meta = load_from_json(path)
    intris = []
    # for sub_i,frame in enumerate(meta["frames"]):
    #     K = frame['intrinsics']
        # intris.append(np.stack([K[0][0],K[1][1],K[0][2],K[1][2]]))
    return np.stack([meta['fl_x'],meta['fl_y'],meta['cx'], meta['cy']])

def load_data(path: str, dataset: str):
    if dataset == 'nuscenes':
        # NuScenes specific: images are in path/images/ directory
        images_dir = os.path.join(path, 'images')
        intrinsics_dir = os.path.join(path, 'intrinsics')
        
        if not os.path.exists(images_dir):
            raise ValueError(f"Images directory not found: {images_dir}")
        if not os.path.exists(intrinsics_dir):
            raise ValueError(f"Intrinsics directory not found: {intrinsics_dir}")
        
        # Extract scene name (directory name, e.g., "000")
        scene_name = os.path.basename(os.path.abspath(path))
        
        # Load all images
        rgbs = sorted(glob.glob(os.path.join(images_dir, '*.jpg')) + 
                      glob.glob(os.path.join(images_dir, '*.png')))
        
        # For each image, extract cam_id from filename (format: {frame_idx:03d}_{cam_id}.jpg)
        # and load corresponding intrinsics
        data = []
        for rgb in rgbs:
            filename = os.path.basename(rgb)
            # Extract cam_id from filename (e.g., "000_0.jpg" -> "0")
            try:
                cam_id = filename.split('_')[-1].split('.')[0]
                intrinsic_file = os.path.join(intrinsics_dir, f"{cam_id}.txt")
                
                if os.path.exists(intrinsic_file):
                    # Load intrinsics: format is [fx, fy, cx, cy, k1, k2, p1, p2, k3]
                    intrinsic_data = np.loadtxt(intrinsic_file)
                    # Extract first 4 values: [fx, fy, cx, cy]
                    intrinsics = [float(intrinsic_data[0]), float(intrinsic_data[1]), 
                                 float(intrinsic_data[2]), float(intrinsic_data[3])]
                else:
                    # Fallback: use default intrinsics if file not found
                    print(f"Warning: Intrinsic file not found: {intrinsic_file}, using default intrinsics")
                    intrinsics = [1000.0, 1000.0, 640.0, 360.0]  # Default values
            except Exception as e:
                print(f"Warning: Failed to parse cam_id from {filename}: {e}, using default intrinsics")
                intrinsics = [1000.0, 1000.0, 640.0, 360.0]  # Default values
            
            data.append({
                'rgb': rgb,
                'depth': None,
                'intrinsic': intrinsics,
                'filename': filename,
                'folder': f'{scene_name}/depth'  # For identification only, doesn't affect save path
            })
        
        return data
    
    # Original logic for other datasets
    rgbs = sorted(glob.glob(path + '/*.jpg') + glob.glob(path + '/*.png') + 
                  glob.glob(path + '/front_images'+ '/*.jpg') + glob.glob(path + '/front_images'+ '/*.png'))
    # print("This is the rgb path: ", path)
    # exit()
    if dataset == 'kitti360':
        intrinsics = [552.5542602539062, 552.5542602539062, 682.0494384765625, 238.76954650878906]
    else:
        assert os.path.exists(os.path.join(path,"transforms.json"))
        intrinsics = load_intrinsics(os.path.join(path,"transforms.json"))
        # intrinsic = [1038.8481920789602, 1038.8481920789602, 471.19850280484064, 324.1388703680434]

    data = [{'rgb':rgb, 'depth':None,  'intrinsic': intrinsics, 'filename':os.path.basename(rgb), 
             'folder': rgb.split('/')[-3]} for i, rgb in enumerate(rgbs)]
    return data