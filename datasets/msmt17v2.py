import os.path as osp

from .bases import BaseImageDataset


class MSMT17V2(BaseImageDataset):
    """
    MSMT17 Version 2
    
    Reference:
    Wei et al. Person Transfer GAN to Bridge Domain Gap for Person Re-Identification. CVPR 2018.
    
    URL: http://www.pkuvmc.com/publications/msmt17.html
    
    Dataset statistics:
    # identities: 1041
    # images: 30248 (train) + 11659 (query) + 82161 (gallery)
    # cameras: 15
    
    Note: This version includes auxiliary time-of-day information extracted from filenames.
    """
    dataset_dir = 'MSMT17_V2'
    
    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(MSMT17V2, self).__init__()
        self.pid_begin = pid_begin
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'mask_train_v2')
        self.test_dir = osp.join(self.dataset_dir, 'mask_test_v2')
        self.list_train_path = osp.join(self.dataset_dir, 'list_train.txt')
        self.list_val_path = osp.join(self.dataset_dir, 'list_val.txt')
        self.list_query_path = osp.join(self.dataset_dir, 'list_query.txt')
        self.list_gallery_path = osp.join(self.dataset_dir, 'list_gallery.txt')
        
        self._check_before_run()
        
        # Process standard datasets
        train = self._process_dir(self.train_dir, self.list_train_path)
        val = self._process_dir(self.train_dir, self.list_val_path)
        train += val
        query = self._process_dir(self.test_dir, self.list_query_path)
        gallery = self._process_dir(self.test_dir, self.list_gallery_path)
        
        # Process auxiliary datasets with time-of-day information
        train_aux = self._process_aux(self.train_dir, self.list_train_path)
        val_aux = self._process_aux(self.train_dir, self.list_val_path)
        train_aux += val_aux
        query_aux = self._process_aux(self.test_dir, self.list_query_path)
        gallery_aux = self._process_aux(self.test_dir, self.list_gallery_path)
        
        if verbose:
            print("=> MSMT17 V2 loaded")
            self.print_dataset_statistics(train, query, gallery)
        
        self.train = train
        self.query = query
        self.gallery = gallery
        
        # Auxiliary datasets with temporal metadata
        self.train_aux = train_aux
        self.query_aux = query_aux
        self.gallery_aux = gallery_aux
        
        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)
    
    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.test_dir):
            raise RuntimeError("'{}' is not available".format(self.test_dir))
    
    def _process_dir(self, dir_path, list_path):
        """Process directory with standard tuple format: (img_path, pid, camid, trackid)"""
        with open(list_path, 'r') as txt:
            lines = txt.readlines()
        dataset = []
        pid_container = set()
        cam_container = set()
        for img_idx, img_info in enumerate(lines):
            img_path, pid = img_info.split(' ')
            pid = int(pid)  # no need to relabel
            camid = int(img_path.split('_')[2])
            img_path = osp.join(dir_path, img_path)
            dataset.append((img_path, self.pid_begin + pid, camid - 1, 0))
            pid_container.add(pid)
            cam_container.add(camid)
        
        if len(dataset) > 0:
            print(f"  Loaded {len(dataset)} images with {len(pid_container)} identities and {len(cam_container)} cameras")
        
        return dataset
    
    def _process_aux(self, dir_path, list_path):
        """Process directory with auxiliary time-of-day information"""
        with open(list_path, "r") as txt:
            lines = txt.readlines()
        dataset = []
        for img_idx, img_info in enumerate(lines):
            img_path, pid = img_info.split(" ")
            pid = int(pid)  # no need to relabel
            camid = int(img_path.split('_')[2])
            todid = self.parse_time_of_day(img_path)
            img_path = osp.join(dir_path, img_path)
            dataset.append((img_path, self.pid_begin + pid, {"time_of_day": todid, "camid": camid - 1}))
        return dataset
    
    def parse_time_of_day(self, img_path):
        """Extract time-of-day information from image filename"""
        timeslot = img_path.split("_")[3]
        
        # Check in specific order: afternoon contains 'noon', so check it first
        if "afternoon" in timeslot:
            return 2
        elif "noon" in timeslot:
            return 1
        elif "morning" in timeslot:
            return 0
        else:
            return -1  # Unknown
    
    @property
    def aux_info(self):
        """Return metadata about auxiliary information available"""
        return {"time_of_day_num": 3}
