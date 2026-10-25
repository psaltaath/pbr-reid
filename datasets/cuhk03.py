import json
import os.path as osp
from .bases import BaseImageDataset


class CUHK03(BaseImageDataset):
    """
    CUHK03 Dataset
    
    Reference:
    Li et al. DeepReID: Deep Filter Pairing Neural Network for Person Re-identification. CVPR 2014.
    
    URL: http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html
    
    Dataset statistics (NEW protocol, detected):
    # identities: 767 (train) + 700 (test)
    # images: 7,365 (train) + 1,400 (query) + 5,332 (gallery)
    # cameras: 2 per camera pair (5 pairs total = 10 cameras)
    
    Protocols:
    - NEW (default): Modern protocol with single split, recommended for fair comparison
    - CLASSIC: Original protocol with 20 random splits
    
    Modes:
    - detected (default): Bounding boxes from pedestrian detector (more realistic)
    - labeled: Human-annotated bounding boxes (cleaner but less realistic)
    """

    dataset_dir = 'cuhk03'

    def __init__(self, root='', mode='detected', protocol='new', split_index=0, verbose=True, **kwargs):
        """
        Args:
            root (str): Root directory of datasets
            mode (str): 'detected' or 'labeled'
            protocol (str): 'new' or 'classic'
            split_index (int): Split index for classic protocol (0-19), ignored for new protocol
            verbose (bool): Print dataset statistics
        """
        super(CUHK03, self).__init__()
        
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.mode = mode
        self.protocol = protocol
        self.split_index = split_index
        
        if mode not in ['detected', 'labeled']:
            raise ValueError(f"mode must be 'detected' or 'labeled', got {mode}")
        if protocol not in ['new', 'classic']:
            raise ValueError(f"protocol must be 'new' or 'classic', got {protocol}")
        
        self._check_before_run()
        
        # Load splits from JSON
        split_file = osp.join(
            self.dataset_dir,
            f'splits_{protocol}_{mode}.json'
        )
        
        with open(split_file, 'r') as f:
            splits = json.load(f)
        
        # For new protocol, there's only one split
        # For classic protocol, use specified split_index
        if protocol == 'new':
            split = splits[0]
        else:
            if split_index >= len(splits):
                raise ValueError(
                    f"split_index {split_index} out of range for classic protocol (max: {len(splits)-1})"
                )
            split = splits[split_index]
            if verbose and len(splits) > 1:
                print(f"=> Using classic protocol split {split_index}/{len(splits)-1}")
        
        train = self._process_split(split['train'])
        query = self._process_split(split['query'])
        gallery = self._process_split(split['gallery'])
        
        if verbose:
            print(f"=> CUHK03 ({mode}, {protocol} protocol) loaded")
            self.print_dataset_statistics(train, query, gallery)
        
        self.train = train
        self.query = query
        self.gallery = gallery
        
        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError(f"'{self.dataset_dir}' is not available")
        
        # Check for required files
        images_dir = osp.join(self.dataset_dir, f'images_{self.mode}')
        if not osp.exists(images_dir):
            raise RuntimeError(f"'{images_dir}' is not available")
        
        split_file = osp.join(self.dataset_dir, f'splits_{self.protocol}_{self.mode}.json')
        if not osp.exists(split_file):
            raise RuntimeError(f"'{split_file}' is not available")

    def _process_split(self, split_data):
        """
        Process a split (train/query/gallery) from JSON data.
        
        Args:
            split_data: List of [image_path, person_id, camera_id]
        
        Returns:
            List of tuples (img_path, pid, camid, viewid)
        """
        dataset = []
        
        for item in split_data:
            # JSON format: [path, pid, camid]
            img_path_raw, pid, camid = item
            
            # Fix path: JSON has Windows-style paths like './data\\cuhk03\\images_detected\\1_001_1_01.png'
            # Split by both forward and back slashes to get just the filename
            img_name = img_path_raw.replace('\\', '/').split('/')[-1]
            
            # Build correct path
            img_path = osp.join(
                self.dataset_dir,
                f'images_{self.mode}',
                img_name
            )
            
            # Verify file exists
            if not osp.exists(img_path):
                # Skip if file doesn't exist
                continue
            
            # Camera IDs in CUHK03 are 1-based, convert to 0-based
            camid = camid - 1
            
            # CUHK03 doesn't have viewpoint info, use 0 as placeholder
            viewid = 0
            
            dataset.append((img_path, pid, camid, viewid))
        
        return dataset
