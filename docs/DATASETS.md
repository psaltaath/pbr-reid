# Datasets

All loaders expect the datasets under one root folder (`data.root_dir`, default
`data/`; override with `--data.root_dir /path` or `export PBR_DATA_ROOT=/path`).
Folder names are fixed by the loaders in `datasets/`:

```
$PBR_DATA_ROOT/
├── Market-1501-v15.09.15/          # bounding_box_train/ query/ bounding_box_test/
├── dukemtmcreid/                   # bounding_box_train/ query/ bounding_box_test/
├── MSMT17_V2/                      # mask_train_v2/ mask_test_v2/ list_{train,val,query,gallery}.txt
├── cuhk03/                         # images_detected/ images_labeled/ splits_new_{detected,labeled}.json
├── IUSTPersonReID/                 # bounding_box_train/ query/ bounding_box_test/
├── VeRi/                           # image_train/ image_query/ image_test/ (VeRi-776)
└── VehicleID_V1.0/                 # image/ train_test_split/ (train_list.txt, test_list_800.txt, ...)
```

| `dataset_name`  | Config            | Train IDs | Geometry (H×W) | Protocol notes |
|-----------------|-------------------|----------:|----------------|----------------|
| `market1501`    | `market1501`      | 751       | 384×192        | junk/same-camera removal |
| `dukemtmcreid`  | `dukemtmc`        | 702       | 384×192        | |
| `msmt17v2`      | `msmt17`          | 1041      | 384×192        | MSMT17 **V2** release (masked crops) |
| `cuhk03`        | `cuhk03`, `cuhk03_labeled` | 767 | 384×192 | new protocol (767/700 split), detected or labeled boxes; the `splits_new_*.json` files follow the TransReID / torchreid preprocessing |
| `iustpersonreid`| `iust`            | 1193      | 384×192        | |
| `veri`          | `veri`            | 576       | 192×384        | VeRi-776 |
| `vehicleid`     | `vehicleid`       | 13164     | 192×384        | no camera labels; S/M/L test splits (`data.vehicleid_test_variants`), gallery = one random image per identity per the standard protocol |

Each loader prints its statistics on construction, so a wrong layout fails fast.

Datasets are not redistributed here; obtain them from their original sources:
Market-1501 (Zheng et al. 2015), DukeMTMC-reID (Ristani et al. 2016 / Zheng et al. 2017),
MSMT17 (Wei et al. 2018), CUHK03 (Li et al. 2014; new protocol Zhong et al. 2017),
IUST PersonReID (Moghaddam et al. 2024), VeRi-776 (Liu et al. 2016), VehicleID (Liu et al. 2016).
