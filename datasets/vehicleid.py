import os.path as osp
from typing import Optional

from .bases import BaseImageDataset


class VehicleID(BaseImageDataset):
    """
    VehicleID Dataset

    Reference:
    Liu et al. Deep Relative Distance Learning: Tell the Difference Between Similar Vehicles. CVPR 2016.

    URL: https://pkuml.org/resources/pku-vehicleid.html

    Dataset statistics:
    # identities: 26328 total (13164 train + 13164 test)
    # images: 221567 total (113346 train)
    # Test set variations:
      - S (Small): 800 vehicles, 6493 images
      - M (Medium): 1600 vehicles, 13377 images
      - L (Large): 2400 vehicles, 19777 images
      - XL (Extra Large): 3200 vehicles, 26353 images
      - XXL (Extra Extra Large): 6000 vehicles, 48922 images
      - Full: 13164 vehicles, 108221 images

    Note: VehicleID has no camera information. All camera IDs are set to 0.

    File format in train/test lists: <image_id> <vehicle_id>
    Example: 0265099 22551
    """

    dataset_dir = "VehicleID_V1.0"

    # Mapping from variant names to test set sizes
    VARIANT_SIZES = {
        "S": 800,  # Small
        "M": 1600,  # Medium
        "L": 2400,  # Large
        "XL": 3200,  # Extra Large
        "XXL": 6000,  # Extra Extra Large
        "Full": 13164,  # Full test set
    }

    def __init__(
        self,
        root: str = "",
        test_variants: Optional[list[str]] = None,
        verbose: bool = True,
        **kwargs,
    ):
        super(VehicleID, self).__init__()

        if test_variants is None:
            test_variants = ["S", "M", "L"]

        # Handle string input (in case it comes from CLI as a string)
        if isinstance(test_variants, str):
            # Remove brackets and quotes if present
            test_variants = test_variants.strip("[]'\"")
            # Split by comma
            test_variants = [v.strip().strip("'\"") for v in test_variants.split(",")]

        # Validate test variants
        for variant in test_variants:
            if variant not in self.VARIANT_SIZES:
                raise ValueError(
                    f"Invalid test variant '{variant}'. "
                    f"Valid options: {list(self.VARIANT_SIZES.keys())}"
                )

        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.image_dir = osp.join(self.dataset_dir, "image")
        self.split_dir = osp.join(self.dataset_dir, "train_test_split")
        self.train_list_path = osp.join(self.split_dir, "train_list.txt")

        self._check_before_run()

        # Load training set (same for all variants) with relabeling
        train = self._process_list(self.train_list_path, relabel=True)

        # Load all requested test variants (no relabeling for test sets)
        self.test_sets = {}
        for variant in test_variants:
            test_size = self.VARIANT_SIZES[variant]
            test_list_path = osp.join(self.split_dir, f"test_list_{test_size}.txt")

            # VehicleID uses same images for query and gallery
            # Following common practice: all test images serve as both
            test_data = self._process_list(test_list_path, relabel=False)

            # For VehicleID, we use all test images as both query and gallery
            # This follows the standard VehicleID evaluation protocol
            self.test_sets[variant] = {
                "query": test_data,
                "gallery": test_data,
            }

        if verbose:
            print("=> VehicleID loaded")
            self._print_dataset_statistics(train, test_variants)

        # Store training set
        self.train = train

        # For backward compatibility, expose first variant as default query/gallery
        first_variant = test_variants[0]
        self.query = self.test_sets[first_variant]["query"]
        self.gallery = self.test_sets[first_variant]["gallery"]

        # Compute statistics
        (
            self.num_train_pids,
            self.num_train_imgs,
            self.num_train_cams,
            self.num_train_vids,
        ) = self.get_imagedata_info(self.train)
        (
            self.num_query_pids,
            self.num_query_imgs,
            self.num_query_cams,
            self.num_query_vids,
        ) = self.get_imagedata_info(self.query)
        (
            self.num_gallery_pids,
            self.num_gallery_imgs,
            self.num_gallery_cams,
            self.num_gallery_vids,
        ) = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError(f"'{self.dataset_dir}' is not available")
        if not osp.exists(self.image_dir):
            raise RuntimeError(f"'{self.image_dir}' is not available")
        if not osp.exists(self.split_dir):
            raise RuntimeError(f"'{self.split_dir}' is not available")
        if not osp.exists(self.train_list_path):
            raise RuntimeError(f"'{self.train_list_path}' is not available")

    def _process_list(self, list_path: str, relabel: bool = False) -> list:
        """
        Process VehicleID text file format.

        Format: <image_id> <vehicle_id>
        Example: 0265099 22551

        Args:
            list_path: Path to the list file
            relabel: Whether to relabel IDs to be contiguous (0 to N-1)

        Returns list of tuples: (img_path, pid, camid, trackid)
        Note: camid and trackid are always 0 for VehicleID
        """
        if not osp.exists(list_path):
            raise RuntimeError(f"'{list_path}' is not available")

        with open(list_path, "r") as f:
            lines = f.readlines()

        # First pass: collect all unique vehicle IDs if relabeling
        if relabel:
            vehicle_ids = set()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 2:
                    continue
                _, vehicle_id = parts
                vehicle_ids.add(int(vehicle_id))

            # Create mapping from original ID to new contiguous ID
            pid2label = {pid: label for label, pid in enumerate(sorted(vehicle_ids))}

        # Second pass: create dataset with relabeled IDs if needed
        dataset = []
        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 2:
                continue

            img_name, vehicle_id = parts
            img_path = osp.join(self.image_dir, f"{img_name}.jpg")

            vehicle_id = int(vehicle_id)
            if relabel:
                vehicle_id = pid2label[vehicle_id]

            # VehicleID has no camera information
            # Format: (img_path, pid, camid, trackid)
            dataset.append((img_path, vehicle_id, 0, 0))

        return dataset

    def _print_dataset_statistics(self, train: list, test_variants: list[str]):
        """Print dataset statistics for all test variants"""
        num_train_pids, num_train_imgs, num_train_cams, _ = self.get_imagedata_info(
            train
        )

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print(
            f"  train    | {num_train_pids:5d} | {num_train_imgs:8d} | {num_train_cams:9d}"
        )

        for variant in test_variants:
            test_data = self.test_sets[variant]["query"]
            num_test_pids, num_test_imgs, num_test_cams, _ = self.get_imagedata_info(
                test_data
            )

            # Show variant name and size
            variant_label = f"test({variant})"
            print(
                f"  {variant_label:8s} | {num_test_pids:5d} | {num_test_imgs:8d} | {num_test_cams:9d}"
            )

        print("  ----------------------------------------")
