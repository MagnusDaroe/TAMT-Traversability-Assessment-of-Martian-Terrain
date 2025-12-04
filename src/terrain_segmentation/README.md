ros2 service call /yolo_trainer/start_training std_srvs/srv/Trigger
# AI4Mars Dataset Structure

## Overview

The AI4Mars dataset is a crowdsourced and expert-labeled dataset of Martian terrain imagery from three NASA Mars rover missions. It contains 60K+ images with merged labels from multiple annotators, providing high-quality semantic segmentation masks for terrain classification.

### Missions Included
- **M2020** - Mars 2020 Rover (Perseverance)
- **MSL** - Mars Science Laboratory (Curiosity Rover)
- **MER** - Mars Exploration Rovers (Spirit and Opportunity)

### Label Types

The dataset provides two distinct label configurations:

#### 1. NAV (Navigation/Traversability)
Used for rover navigation and path planning. Available for **all missions** (M2020, MSL, MER).

**Classes (4):**
- `0` - soil (fine-grained material, generally traversable)
- `1` - bedrock (solid rock surface)
- `2` - sand (loose granular material, potentially hazardous)
- `3` - big rock (large obstacles)
- `255` - NULL (unlabeled/masked regions)

**Use cases:** Autonomous navigation, traversability analysis, path planning

#### 2. GEO (Geology)
Used for scientific geological analysis. Available for **M2020 only**.

**Classes (22):**
- **Bedrock** (0-6): massive, layered (angled/flat/unsure), conglomerate, holey, unsure
- **Float Rocks** (10-17): massive, layered (angled/flat/unsure), conglomerate, holey, mixed, unsure
- **Sand** (20-22): dune, ripples, sand
- **Other** (30-50): pebbles, vein, hill/peak
- `255` - NULL (unlabeled/masked regions)

**Use cases:** Geological mapping, rock classification, scientific analysis

### Data Quality

- **Train set**: Crowdsourced labels with minimum 3 labelers and 2/3 agreement per pixel
- **Test set**: Expert labels with 100% agreement required (using `min3-100agree` version)
- **Masking**: Distances >30m and rover body are masked to ensure quality
- **Format**: Grayscale PNG images with pixel values representing class IDs

---

## Directory Structure

```
ai4mars-dataset-merged-0.6/
│
├── m2020/                              # Mars 2020 (Perseverance) mission
│   ├── images/
│   │   ├── HAFIQ/                      # High-resolution color images
│   │   │   └── *.JPG
│   │   ├── mcam/                       # MastCam color images
│   │   │   └── *.JPG
│   │   └── ncam/                       # Navigation camera images
│   │       └── *.JPG
│   │
│   └── labels/
│       ├── M2020_GEO/                  # Geology labels (M2020 only)
│       │   ├── mcam/
│       │   │   ├── train/
│       │   │   │   └── *.png           # Training labels (22 classes)
│       │   │   └── test/
│       │   │       └── *.png           # Test labels (expert-annotated)
│       │   └── ncam/
│       │       ├── train/
│       │       └── test/
│       │
│       └── NAV/                        # Navigation labels
│           ├── train/
│           │   └── *.png               # Training labels (4 classes)
│           └── test/
│               └── *.png               # Test labels (expert-annotated)
│
├── msl/                                # Mars Science Laboratory (Curiosity)
│   ├── mcam/                           # MastCam
│   │   ├── images/
│   │   │   └── *.JPG                   # Color images
│   │   └── labels/
│   │       ├── train/
│   │       │   └── *.png               # NAV labels only (4 classes)
│   │       └── test/
│   │           └── masked-gold-min3-100agree/
│   │               └── *.png           # Expert test labels
│   │
│   └── ncam/                           # Navigation camera
│       ├── images/
│       │   ├── edr/                    # Raw EDR images
│       │   │   └── *.JPG
│       │   ├── mxy/                    # Rover masks (binary)
│       │   │   └── *.png
│       │   └── rng-30m/                # Range masks (30m limit)
│       │       └── *.png
│       └── labels/
│           ├── train/
│           │   └── *.png               # NAV labels only (4 classes)
│           └── test/
│               └── masked-gold-min3-100agree/
│                   └── *.png           # Expert test labels
│
└── mer/                                # Mars Exploration Rovers (Spirit/Opportunity)
    ├── images/
    │   ├── eff/                        # Front camera images
    │   │   └── *.JPG
    │   └── test/
    │       └── *.JPG
    │
    └── labels/
        ├── train/
        │   └── merged-unmasked/
        │       └── *.png               # NAV labels only (4 classes)
        └── test/
            └── masked-gold-min3-100agree/
                └── *.png               # Expert test labels
```

---

### Image Types by Camera
- **NavCam** (ncam): Grayscale navigation imagery
- **MastCam** (mcam): Color science imagery  
- **HAFIQ** (M2020): High-resolution color imagery
- **EFF** (MER): Front-facing grayscale imagery

### Pixel Value Format
Labels are grayscale PNG files where pixel intensity = class ID:
- NAV: Values are 0, 1, 2, 3, or 255
- GEO: Values are 0-6, 10-17, 20-22, 30, 40, 50, or 255
- These values appear black in standard image viewers (zoom in or use code to verify)

---

## Conversion Tool

Use `convert_ai4mars_to_yolo.py` to convert this dataset to YOLO format:

```bash
# Create both NAV and GEO datasets
python3 convert_ai4mars_to_yolo.py /path/to/ai4mars-dataset-merged-0.6

# Create only navigation dataset (all missions)
python3 convert_ai4mars_to_yolo.py /path/to/dataset --config nav

# Create only geology dataset (M2020 only)
python3 convert_ai4mars_to_yolo.py /path/to/dataset --config geo

# Example command
python3 src/terrain_segmentation/scripts/convert_ai4mars_to_yolo.py /home/spacerob/Documents/ai4mars-dataset-merged-0.6

```

**Output structure:**
```
AI4MARS_YOLO/
├── ai4mars_nav/          # 4-class navigation dataset
│   ├── images/
│   ├── labels/           # PNG masks + YOLO TXT files
│   └── data.yaml
└── ai4mars_geo/          # 22-class geology dataset
    ├── images/
    ├── labels/           # PNG masks + YOLO TXT files
    └── data.yaml
```

**View labels:**

```bash
# Auto-detect configuration from directory name
python view_yolo_txt_labels.py /path/to/ai4mars_nav

# Force a specific configuration
python view_yolo_txt_labels.py /path/to/yolo_format --config nav
python view_yolo_txt_labels.py /path/to/yolo_format --config geo

# Example command
python3 src/terrain_segmentation/scripts/view_yolo_txt_labels.py /home/spacerob/Documents/ai4mars-dataset-merged-0.6/AI4MARS_YOLO/ai4mars_nav

```



```bash
python3 src/terrain_segmentation/scripts/reconfigure_model_output.py \
  --source ~/tamt/src/terrain_segmentation/models/terrain_segmentation/exp2/weights/best.pt \
  --target-classes 6 \
  --output ~/tamt/src/terrain_segmentation/models/adapted_models/exp3_adapted_6class.pt
  ```









  https://y-t-g.github.io/tutorials/yolov8n-add-classes/





    python3 /home/spacerob/tamt/src/terrain_segmentation/scripts/split_dataset.py /home/spacerob/tamt/dataset/tamt_simulation_4_3500 --train 0.8 --val 0.15 --test 0.05
 

  ros2 run terrain_segmentation Image_publisher_node.py \
    --ros-args \
    -p image_folder:=/home/daroe/tamt/dataset/AI4MARS_NAV_GOOD/images \
    -p publish_rate:=2.0 \
    -p loop:=true


    m2020_test_NLF_0034_0669963392_071ECM_N0031392NCAM03034_07_195J_merged12.jpeg
