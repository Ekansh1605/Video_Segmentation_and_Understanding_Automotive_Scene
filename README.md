# Video Segmentation and Understanding of Automotive Scenes

This project implements a temporal deep‑learning pipeline for understanding automotive driving scenes from video, including semantic segmentation, monocular depth estimation and ego‑motion estimation, trained on CARLA sequences. It provides full training scripts, evaluation utilities and 3D visualisation tools (camera poses and point clouds) to analyse model performance over time.

***

## Overview

This repository contains a modular PyTorch implementation for video‑based scene understanding in autonomous driving scenarios.  
Given short sequences from CARLA (RGB images, segmentation labels, depth maps and camera extrinsics), the system predicts:

- Per‑pixel semantic segmentation for each frame  
- Per‑pixel depth maps  
- Ego‑motion between consecutive frames as 4×4 camera pose transforms  

The architecture is split into three main parts:

- **Segmentation network** (DeepLabv3‑ResNet50)  
- **Depth estimation network** (ResNet50 + custom decoder)  
- **Ego‑motion network** (feature‑based motion estimator + GRU + camera head)  

A **Geometry Filter** (with a ConvGRU) maintains temporal consistency for segmentation and depth, while an **Ego Motion Filter** models camera motion over time.

***

## Main Features

- CARLA dataset loaders for images, segmentations, depth and camera extrinsics  
- Temporal data augmentations (noise, clutter, lighting changes)  
- Semantic segmentation training (DeepLabv3‑ResNet50)  
- Monocular depth estimation with a custom decoder  
- Ego‑motion estimation from features, producing 4×4 camera transforms  
- Sequence‑level training of a ConvGRU‑based geometry filter with joint segmentation + depth  
- Sequence inference over full videos (segmentation, depth, transforms)  
- Visualisations:
  - Qualitative segmentation comparisons (image / ground truth / prediction)  
  - Camera trajectory plots in 3D  
  - Point‑cloud reconstruction from predicted depth + camera poses  
  - GIF generation of predictions over time  

***

## Project Structure

Core data and models:

- `datasets.py`  
  - `CarlaDataset`: base dataset for single images + segmentation labels (and optional depth)  
  - `MovementDataset`: returns `(image1, image2, p1, p2)` for ego‑motion training  
  - `DepthDataset`: returns `(image, depth)` for depth training  
  - `SequenceDataset`: returns sequences `(images, segmentations, depths, extrinsics)`  

- `augmentations.py`  
  - `AddGaussianNoise`: adds clipped Gaussian noise  
  - `RandomAddClutter`: adds occluding patches across a sequence  
  - `change_lighting`: simulates temporal lighting changes  

- `segmentation_model.py`  
  - DeepLabv3‑ResNet50 backbone + classifier, with an optional feature filter  

- `identity.py`  
  - Identity module used to bypass filters / heads when needed  

- `geometry_filter.py`  
  - Deeplab backbone  
  - `ConvGRUCell` from `conv_gru.py` for temporal feature memory  
  - Segmentation head (classifier)  
  - Depth head (`DepthDecoder`)  

- `depth_estimation_model.py`  
  - ResNet50 backbone + `DepthDecoder` for monocular depth  

- `motion_estimator.py`  
  - CNN that encodes motion from two feature maps into a 128‑dimensional motion vector  

- `camera_head.py`  
  - MLP that maps the motion vector to translation and rotation parameters (roll, pitch, yaw via sinus)  

- `ego_motion_filter.py`  
  - Main ego‑motion model (MotionEstimator + GRUCell + CameraHead) producing a 4×4 camera transform per frame pair  
  - Variants: `ego_motion_filter_no_rnn.py`, `ego_motion_filter_old.py`  

- `sequence_segmenter.py`  
  - `process_video`: runs GeometryFilter + EgoMotionFilter over a clip, returning:
    - Predicted segmentations
    - Predicted depth maps
    - Predicted camera transforms  
  - `process_video_framewise`: frame‑wise segmentation without temporal state  

- `util.py`  
  - Random seed setting  
  - Parameter counting  
  - Saving/loading checkpoints for geometry + ego‑motion and segmenter/depth models  
  - IoU computation  
  - `get_pretrained_resnet` to extract a pretrained DeepLab backbone  
  - `add_visualization` for logging qualitative segmentation examples  

Training loops and scripts:

- **Segmentation**
  - `trainer.py`: generic training/evaluation loop for segmentation  
  - `train_segmenter.py`: entry point to train DeepLabv3‑ResNet50 on CARLA  

- **Depth**
  - `depth_trainer.py`: depth training / evaluation loops  
  - `depth_train_script.py`: entry point to train the depth model  

- **Ego‑motion**
  - `motion_trainer.py`: training loop for ego‑motion, using a frozen segmentation backbone as feature extractor  
  - `train_script_egomotion.py`: entry point to train the Ego Motion Filter  

- **Sequence (joint temporal)**
  - `sequence_trainer.py`: training loops for GeometryFilter + EgoMotionFilter on sequences  
  - `train_script_sequence.py`: entry point to train the full temporal model  

Visualisation and 3D tools:

- `visualizations.py`: plotting utilities (qualitative segmentation, etc.)  
- `save_image.py`: save tensors as image files  
- `save_gif.py`: generate animated GIFs from sequences of frames  
- `camera_parameter_loader.py`: load camera intrinsics/extrinsics  
- `camera_pose_visualizer.py`: 3D plotting of camera poses  
- `visualize_camera_poses.py`: script to visualise camera trajectories  
- `pointcloud.py`: reconstruction of 3D point clouds from depth + camera poses  

Analysis notebooks:

- `notebook.ipynb`, `playground.ipynb`, `demonstation.ipynb`: experiments and demos  
- `depth_notebook.ipynb`: depth‑specific analysis  
- `ego_motion_notebook.ipynb`: ego‑motion analysis  
- `pointcloud.ipynb`: point‑cloud visualisation  

Helper scripts:

- `find_weights.py`: computes per‑class weights for segmentation loss  

***

## Installation

```bash
git clone https://github.com/<your-username>/Video_Segmentation_and_Understanding_Automotive_Scene.git
cd Video_Segmentation_and_Understanding_Automotive_Scene

# (Optional) create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

If `requirements.txt` is not present yet, typical dependencies are:

- Python 3.9+  
- PyTorch and torchvision (with CUDA support recommended)  
- numpy  
- matplotlib  
- tqdm  
- tensorboard  

***

## Dataset

The code expects a CARLA dataset structured approximately as:

```text
root/
  Town01/
    seq_0000/
      img_000.png
      segmentation_000.png
      depth_000.png
      ...
      meta.pkl
    seq_0001/
    ...
  Town02/
  ...
```

Assumptions:

- Images: `img_XXX.png`  
- Segmentation labels: `segmentation_XXX.png` (class indices encoded in RGB, mapped to 0…21)  
- Depth maps: `depth_XXX.png` (RGB encoding decoded in `read_depth`)  
- Camera parameters: `meta.pkl` with per‑frame extrinsics  

Adjust `root_dir` in the scripts and notebooks to point to your dataset.

***

## How to Train

### 1. Train the segmentation model

```bash
python train_segmenter.py
```

This will:

- Build training and validation splits using `CarlaDataset`  
- Instantiate `SegmentationModel` (DeepLabv3‑ResNet50)  
- Train for the configured number of epochs  
- Save checkpoints under `models/` (backbone + classifier)  

### 2. Train the depth estimation model

```bash
python depth_train_script.py
```

This will:

- Use `DepthDataset` for training/validation  
- Instantiate `DepthEstimiationModel` (ResNet50 + DepthDecoder)  
- Train the depth head and optionally the backbone  
- Save depth model checkpoints in `models/`  

### 3. Train the ego‑motion model

Ensure a trained segmentation backbone checkpoint exists (e.g. `models/segmenter_epoch_8.pth`), then run:

```bash
python train_script_egomotion.py
```

This will:

- Use `MovementDataset` (frame pairs + extrinsic matrices)  
- Freeze the pretrained segmentation backbone and extract features  
- Train `EgoMotionFilter` to predict camera transforms between frames  
- Save ego‑motion checkpoints to `models/`  

### 4. Train the full temporal sequence model

```bash
python train_script_sequence.py
```

This will:

- Use `SequenceDataset` (short video sequences)  
- Train `GeometryFilter` (segmentation + depth + ConvGRU) and `EgoMotionFilter` jointly on sequences  
- Save combined checkpoints (geometry + ego‑motion) to `models/`  

***

## Inference and Visualisation

### Sequence inference

Use `sequence_segmenter.py` to run the trained temporal model over a sequence:

```python
from sequence_segmenter import process_video
from geometry_filter import GeometryFilter
from ego_motion_filter import EgoMotionFilter
from datasets import SequenceDataset
import torch

# Example (pseudo-code)
dataset = SequenceDataset(root_dir=..., split='test')
sequence, segs, depths, meta = dataset[0]        # one sequence
sequence = sequence.unsqueeze(0)                 # add batch dimension

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
geometry_filter = GeometryFilter().to(device)
ego_motion_filter = EgoMotionFilter().to(device)

# load your trained weights here

pred_segs, pred_depths, pred_transforms = process_video(sequence, geometry_filter, ego_motion_filter, device)
```

### Qualitative segmentation results

During training/evaluation, `util.add_visualization` can log qualitative segmentation examples (image, ground truth, prediction).  
You can also call plotting functions from `visualizations.py` directly on your predicted outputs.

### Camera pose trajectories

```bash
python visualize_camera_poses.py
```

This script visualises predicted and/or ground‑truth camera trajectories in 3D using `camera_pose_visualizer.py`.

### Point‑cloud reconstruction

```bash
python pointcloud.py
```

This script reconstructs 3D point clouds from predicted depth and camera poses, allowing inspection of the estimated scene geometry.

### GIF generation

```bash
python save_gif.py
```

Use this script to create GIFs of predictions over time (for example, segmentation overlays or depth maps).

***

## Results

Add your own quantitative and qualitative results here, e.g.:

- Mean IoU per class on validation sequences  
- Depth metrics (RMSE, absolute relative error, etc.)  
- Pose errors if ground‑truth poses are available  
- Screenshots of segmentations, depth maps, camera trajectories and point‑clouds  

***

## Future Work

- Training on real driving datasets (KITTI, Cityscapes, etc.)  
- Multi‑task losses combining segmentation, depth and motion consistency  
- More advanced temporal architectures (ConvLSTM, transformers)  
- Uncertainty estimation for segmentation and depth  

***

## License

Specify your license here (e.g. MIT, Apache 2.0) and acknowledge any external repositories or assets you adapted (for example, ConvGRU implementation, pretrained backbones or CARLA).
