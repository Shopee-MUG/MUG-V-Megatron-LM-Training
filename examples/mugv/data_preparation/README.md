# Data Preparation for MUG-V Megatron Training

This directory provides a streamlined pipeline to prepare training data for MUG-V video generation model. All scripts use the exact same models (VideoVAE and T5 encoder) from `MUG-V` to ensure consistency with inference.

---

## 📋 Quick Links

- **[Quick Start Guide](QUICKSTART.md)** - TL;DR commands and common usage
- **[Sample Dataset](#download-sample-dataset)** - Download pre-processed sample data for testing
- **[Environment Setup](#environment-setup)** - Python environment and dependency installation
- **[Usage Examples](#step-by-step-guide)** - Detailed step-by-step instructions
- **[Validation Report](VALIDATION_REPORT.md)** - Logic correctness verification

---

## Download Sample Dataset

If you want to quickly test the training pipeline without preparing your own data, you can download our pre-processed sample dataset:

**Dataset:** [MUG-V/MUG-V-Training-Samples](https://huggingface.co/datasets/MUG-V/MUG-V-Training-Samples)

```bash
# Install Hugging Face CLI
pip install huggingface_hub

# Download the sample dataset
huggingface-cli download MUG-V/MUG-V-Training-Samples --repo-type dataset --local-dir sample_dataset

# The downloaded dataset includes:
# sample_dataset/
# ├── train.csv              # Training metadata
# ├── latents/               # Pre-encoded VideoVAE latents
# └── text_features/         # Pre-encoded T5-XXL features
```

This sample dataset is ready to use for training. You can skip the data preparation steps below if you just want to test the training pipeline.

**To prepare your own data**, continue reading the sections below.

---

## Environment Setup

```bash
uv venv --python 3.12
source .venv/bin/activate  # or: .venv\Scripts\activate on Windows
uv pip install -r examples/mugv/data_preparation/requirements.txt
```

**Disk Space**: ~20 GB (PyTorch + CUDA + T5-XXL weights)

**Note**: The requirements.txt includes `MUG-V` (installed via git) which provides the VideoVAE and T5Encoder model definitions used in both training and inference.

---

## Overview

The data preparation pipeline transforms your raw videos and captions into training-ready format in 3 simple steps:

```
Your Raw Data (videos/ + captions.csv)
    ↓
Step 1: Extract Text Features (T5-XXL embeddings)
    ↓
Step 2: Encode Videos (VideoVAE latents)
    ↓
Step 3: Generate Training CSV (match & validate)
    ↓
Ready for Training!
```

**Input (What You Provide):**
```
your_raw_data/
├── videos/
│   ├── video_001.mp4
│   ├── video_002.mp4
│   └── ...
└── captions.csv         # sample_id + text columns
```

**Output (Ready for Training):**
```
data_root/
├── train.csv                    # Training metadata
├── latents/                     # VideoVAE latents
│   ├── video_001.pt            # Shape: [24, T, H, W]
│   └── video_002.pt
└── text_features/               # T5-XXL embeddings
    ├── video_001_text.pt       # Dict: {'y': [1, 1, L, 4096], 'mask': [1, L]}
    └── video_002_text.pt
```

## Prepare Your Raw Data

Before running the data preparation scripts, you need to organize your raw data (videos and captions) in the following structure:

**Required Input Structure:**
```
your_raw_data/
├── videos/
│   ├── video_001.mp4
│   ├── video_002.mp4
│   ├── video_003.mp4
│   └── ...
└── captions.csv
```

**Important: File Naming Convention**
- Video filenames (without extension) must match the `sample_id` in captions.csv
- For example: `video_001.mp4` → `sample_id` must be `video_001`
- Supported video formats: `.mp4`, `.webm`, `.mkv`, `.mov`, `.avi`

**Captions CSV Format:**

The captions.csv file must contain two columns: `sample_id` and `text`

```csv
sample_id,text
video_001,"A woman showcasing her elegant dress in a modern studio"
video_002,"Product demonstration of a smartphone with rotating camera view"
video_003,"Cooking tutorial showing how to make pasta from scratch"
```

**Column Requirements:**
- `sample_id`: Unique identifier matching video filename (without extension)
- `text`: Natural language description of the video content

---

## Step-by-Step Guide

### Step 1: Extract Text Features

Extract T5-XXL embeddings from captions using the exact same encoder as inference.

```bash
python data_preparation/1_encode_text_features.py \
    --captions /path/to/captions.csv \
    --output-dir /path/to/text_features \
    --batch-size 32
```

**Arguments:**
- `--captions`: CSV or JSONL file with captions (required)
- `--output-dir`: Output directory for text features (required)
- `--text-column`: Column name for text (default: `text`)
- `--id-column`: Column name for sample ID (default: `sample_id`)
- `--model-name`: T5 model name (default: `DeepFloyd/t5-v1_1-xxl`)
- `--max-length`: Max sequence length (default: 300)
- `--batch-size`: Batch size (default: 32)

**Output:**
- `text_features/<sample_id>_text.pt` for each sample
- Format: `{'y': FloatTensor[1, 1, seq_len, 4096], 'mask': BoolTensor[1, seq_len]}`

---

### Step 2: Encode Videos to Latents

Encode videos using the VideoVAE from `MUG-V`.

**Download VideoVAE Model:**

```bash
# Download VideoVAE checkpoint from Hugging Face
wget https://huggingface.co/MUG-V/MUG-V-inference/resolve/main/vae.pt -O /path/to/vae.pt

# Or using huggingface-cli
pip install huggingface_hub
huggingface-cli download MUG-V/MUG-V-inference vae.pt --local-dir ./models
```

**Run Video Encoding:**

```bash
python data_preparation/2_encode_video_latents.py \
    --video-dir /path/to/videos \
    --output-dir /path/to/latents \
    --vae-checkpoint /path/to/vae.pt \
    --batch-size 1 \
    --fps 24
```

**Arguments:**
- `--video-dir`: Directory with video files (required)
- `--output-dir`: Output directory for latents (required)
- `--vae-checkpoint`: Path to VideoVAE checkpoint (required, download from [here](https://huggingface.co/MUG-V/MUG-V-inference/blob/main/vae.pt))
- `--fps`: Target FPS (default: 24)
- `--batch-size`: Batch size (default: 1, increase if you have VRAM)
- `--device`: Device to use (default: auto-detect)
- `--compile`: Use torch.compile for faster encoding

**Output:**
- `latents/<sample_id>.pt` for each video
- Format: `FloatTensor[24, T, H, W]` (24 channels, T frames after 8× temporal compression)

---

### Step 3: Generate Training CSV

Match latents and text features to create the training CSV.

```bash
python data_preparation/3_generate_training_csv.py \
    --latents /path/to/latents \
    --text-features /path/to/text_features \
    --output /path/to/train.csv \
    --source real
```

**Arguments:**
- `--latents`: Directory with latent `.pt` files (required)
- `--text-features`: Directory with text feature `.pt` files (required)
- `--output`: Output CSV path (required)
- `--captions`: Optional captions CSV for validation
- `--source`: `real` or `generated` (default: `real`)
  - `real`: Apply VAE normalization using dataset mean/std
  - `generated`: Skip normalization
- `--relative-paths`: Use relative paths in CSV

**Output CSV format:**
```csv
sample_id,source,latent_path,text_feat_path
video_001,real,latents/video_001.pt,text_features/video_001_text.pt
video_002,real,latents/video_002.pt,text_features/video_002_text.pt
```

---

## Complete Example Workflow

This example shows the complete workflow from raw data to ready-for-training dataset.

**Starting Point: Your Raw Data**
```
/path/to/my_videos/
├── videos/
│   ├── clip_001.mp4
│   ├── clip_002.mp4
│   └── clip_003.mp4
└── captions.csv
```

Where `captions.csv` contains:
```csv
sample_id,text
clip_001,"A chef preparing a gourmet meal in a professional kitchen"
clip_002,"Sunset over ocean waves crashing on the beach"
clip_003,"City traffic at night with neon lights reflecting on wet streets"
```

**Step-by-Step Processing:**

```bash
# 0. Install dependencies
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r examples/mugv/data_preparation/requirements.txt

# 1. Set up output directory
export RAW_DATA=/path/to/my_videos
export DATA_ROOT=/path/to/processed_data
mkdir -p $DATA_ROOT/{latents,text_features}

# 2. Extract text features from captions
python data_preparation/1_encode_text_features.py \
    --captions $RAW_DATA/captions.csv \
    --output-dir $DATA_ROOT/text_features \
    --batch-size 32

# 3. Encode videos to latents
python data_preparation/2_encode_video_latents.py \
    --video-dir $RAW_DATA/videos \
    --output-dir $DATA_ROOT/latents \
    --vae-checkpoint /path/to/vae.pt \
    --fps 24

# 3. Generate training CSV
python data_preparation/3_generate_training_csv.py \
    --latents $DATA_ROOT/latents \
    --text-features $DATA_ROOT/text_features \
    --output $DATA_ROOT/train.csv

# Ready for training!
# Your processed data is now at: $DATA_ROOT/
# Mount this directory to /data in container and start training
```

**After Processing, Your Data Structure:**
```
/path/to/processed_data/
├── train.csv
├── latents/
│   ├── clip_001.pt          # [24, T, H, W]
│   ├── clip_002.pt
│   └── clip_003.pt
└── text_features/
    ├── clip_001_text.pt     # {'y': [1, 1, L, 4096], 'mask': [1, L]}
    ├── clip_002_text.pt
    └── clip_003_text.pt
```

---

## Data Format Specifications

### Latent Files (`*.pt`)

**Format**: PyTorch tensor saved with `torch.save()`

**Shape**: `[C, T, H, W]`
- `C = 24`: VAE latent channels
- `T`: Temporal frames (video length / 8, due to 8× temporal compression)
- `H, W`: Spatial dimensions (video height/8, video width/8, due to 8× spatial compression)

**Example:**
```python
latent = torch.load("latents/video_001.pt")
print(latent.shape)  # torch.Size([24, 30, 64, 64])
# This represents a ~5s video at 720p (30 frames * 8 = 240 frames @ 24fps)
```

### Text Feature Files (`*_text.pt`)

**Format**: Python dict saved with `torch.save()`

**Structure:**
```python
{
    'y': torch.FloatTensor,    # Shape: [1, 1, seq_len, 4096]
    'mask': torch.BoolTensor,  # Shape: [1, seq_len]
}
```

**Example:**
```python
text_feat = torch.load("text_features/video_001_text.pt")
print(text_feat['y'].shape)    # torch.Size([1, 1, 300, 4096])
print(text_feat['mask'].shape) # torch.Size([1, 300])
print(text_feat['mask'].sum()) # tensor(87) - actual valid tokens
```

### Training CSV

**Required columns:**
- `sample_id`: Unique identifier (string)
- `source`: `generated` or `real`
- `latent_path`: Relative path to latent `.pt` (from CSV directory)
- `text_feat_path`: Relative path to text feature `.pt` (from CSV directory)

**Note**: All paths in the CSV are relative to the CSV file location.

---

## Troubleshooting

### ImportError: Cannot import MUGVAE or T5Encoder

**Solution:**
```bash
# Reinstall requirements (includes MUG-V)
uv pip install -r examples/mugv/data_preparation/requirements.txt
```

### Out of memory during video encoding

**Solution:** Reduce batch size or process videos sequentially:
```bash
python data_preparation/2_encode_video_latents.py \
    --batch-size 1 \
    ... # other args
```

### Mismatched sample counts

If you see different numbers of latents and text features:

1. Check that video files and caption sample_ids match
2. Use `3_generate_training_csv.py --captions` to filter to valid samples

### Wrong tensor shapes

Expected shapes:
- Latents: `[24, T, H, W]` (4D tensor)
- Text: `{'y': [1, 1, L, 4096], 'mask': [1, L]}` (dict with 2 tensors)

If shapes are wrong, re-run the extraction scripts with correct parameters.

You can manually verify a few samples with:
```python
import torch
latent = torch.load("latents/video_001.pt")
text_feat = torch.load("text_features/video_001_text.pt")
print(f"Latent: {latent.shape}")
print(f"Text y: {text_feat['y'].shape}")
print(f"Text mask: {text_feat['mask'].shape}")
```

---

## Advanced Usage

### Custom T5 Model

Use a different T5 variant or fine-tuned model:

```bash
python data_preparation/1_encode_text_features.py \
    --model-name /path/to/custom-t5 \
    --max-length 512 \
    ...
```

### Processing Subset of Data

Use `--captions` filter in `3_generate_training_csv.py`:

```bash
python data_preparation/3_generate_training_csv.py \
    --latents /path/to/latents \
    --text-features /path/to/text_features \
    --captions /path/to/subset.csv \
    --output /path/to/train_subset.csv
```

Only samples present in `subset.csv` will be included.

### Multiple Datasets

Combine multiple datasets by concatenating CSVs:

```bash
cat dataset1/train.csv dataset2/train.csv | grep -v "^sample_id" > combined.csv
# Add header back
echo "sample_id,source,latent_path,text_feat_path" | cat - combined.csv > train.csv
```

Make sure paths are correctly set up relative to the final CSV location.

---

After running these steps, your dataset is ready for Megatron training. Simply mount the `data_root` directory and point the training script to `train.csv`.

For training instructions, see the main [README](../README.md#quick-start).

