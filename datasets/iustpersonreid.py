# encoding: utf-8
"""
IUST PersonReID Dataset

Reference:
IUST Person Re-identification Dataset
"""

import glob
import re
import os.path as osp

from .bases import BaseImageDataset


class IUSTPersonReID(BaseImageDataset):
    """
    IUST PersonReID
    
    Dataset statistics:
    # identities: 1193 (train)
    # images: 72393 (train) + 1428 (query) + 45062 (gallery)
    # cameras: 17 (non-contiguous IDs: 1-11, 21-24, 31-32)
    """
    dataset_dir = 'IUSTPersonReID'

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(IUSTPersonReID, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()
        self.pid_begin = pid_begin
        
        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> IUST PersonReID loaded")
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
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        """
        Process directory with camera ID mapping for non-contiguous camera IDs.
        
        IUST has non-contiguous camera IDs (1-11, 21-24, 31-32).
        We map them to contiguous 0-indexed range.
        """
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([\d]+)_c([\d]+)s')

        # First pass: collect all unique camera IDs
        raw_camids = set()
        for img_path in img_paths:
            match = pattern.search(img_path)
            if match:
                _, camid = map(int, match.groups())
                raw_camids.add(camid)
        
        # Create camera ID mapping: non-contiguous -> contiguous 0-indexed
        cam_mapping = {cam: idx for idx, cam in enumerate(sorted(raw_camids))}
        
        # Second pass: collect person IDs for relabeling
        pid_container = set()
        for img_path in img_paths:
            match = pattern.search(img_path)
            if match:
                pid, _ = map(int, match.groups())
                if pid == -1:
                    continue  # junk images
                pid_container.add(pid)
        
        pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}
        
        # Third pass: process images
        dataset = []
        for img_path in sorted(img_paths):
            match = pattern.search(img_path)
            if not match:
                continue
            
            pid, camid = map(int, match.groups())
            if pid == -1:
                continue  # junk images
            
            # Map camera ID to contiguous range
            camid = cam_mapping[camid]
            
            if relabel:
                pid = pid2label[pid]
            
            dataset.append((img_path, self.pid_begin + pid, camid, 0))
        
        return dataset
