# Data Preparation Quick Reference

## 🚀 Option 1: Download Sample Dataset (Fastest)

**For quick testing without preparing your own data:**

```bash
# Install Hugging Face CLI
pip install huggingface_hub

# Download pre-processed sample dataset
huggingface-cli download MUG-V/MUG-V-Training-Samples --repo-type dataset --local-dir sample_dataset

# Ready to use! The dataset includes:
# - train.csv (training metadata)
# - latents/ (pre-encoded VideoVAE latents)
# - text_features/ (pre-encoded T5-XXL features)
```

**Dataset:** [MUG-V/MUG-V-Training-Samples](https://huggingface.co/datasets/MUG-V/MUG-V-Training-Samples)

---

## 🚀 Option 2: Prepare Your Own Data

**For production training with your own videos:**

```bash
# 1. Install dependencies
uv venv --python 3.10 && source .venv/bin/activate
uv pip install -r examples/mugv/data_preparation/requirements.txt

# 2. Download VideoVAE model
wget https://huggingface.co/MUG-V/MUG-V-inference/resolve/main/vae.pt -O vae.pt

# 3. Prepare your data (put videos in videos/, captions in captions.csv)
# Then run the pipeline:
python data_preparation/1_encode_text_features.py \
    --captions captions.csv --output-dir text_features

python data_preparation/2_encode_video_latents.py \
    --video-dir videos --output-dir latents --vae-checkpoint vae.pt

python data_preparation/3_generate_training_csv.py \
    --latents latents --text-features text_features --output train.csv
```

---

## 📁 File Organization

```
data_preparation/
├── README.md                   # Complete usage documentation
├── VALIDATION_REPORT.md        # Logic correctness verification
├── requirements.txt            # Python dependencies
│
├── 1_encode_text_features.py  # T5-XXL text encoding
├── 2_encode_video_latents.py  # VideoVAE encoding
└── 3_generate_training_csv.py # CSV generation
```

---

## 🔧 Common Commands

### Download Models

```bash
# Download VideoVAE for video encoding
wget https://huggingface.co/MUG-V/MUG-V-inference/resolve/main/vae.pt -O vae.pt

# Or use huggingface-cli
pip install huggingface_hub
huggingface-cli download MUG-V/MUG-V-inference vae.pt --local-dir ./models
```

**Model Links:**
- VideoVAE: [vae.pt](https://huggingface.co/MUG-V/MUG-V-inference/blob/main/vae.pt)
- MUGDiT-10B: [dit.pt](https://huggingface.co/MUG-V/MUG-V-inference/blob/main/dit.pt)

### Text Encoding
```bash
python data_preparation/1_encode_text_features.py \
    --captions /path/to/captions.csv \
    --output-dir /path/to/text_features \
    --batch-size 32 \
    --max-length 300
```

### Video Encoding
```bash
python data_preparation/2_encode_video_latents.py \
    --video-dir /path/to/videos \
    --output-dir /path/to/latents \
    --vae-checkpoint /path/to/vae.pt \
    --batch-size 1 \
    --compile  # Optional: faster encoding
```

### CSV Generation
```bash
python data_preparation/3_generate_training_csv.py \
    --latents /path/to/latents \
    --text-features /path/to/text_features \
    --output /path/to/train.csv \
    --source real  # or "generated"
```

---

## 📊 Data Format Reference

### Input: Captions CSV
```csv
sample_id,text
video_001,A person walking in the park
video_002,A car driving on the highway
```

### Output: Training CSV
```csv
sample_id,source,latent_path,text_feat_path
video_001,real,latents/video_001.pt,text_features/video_001_text.pt
video_002,real,latents/video_002.pt,text_features/video_002_text.pt
```

### Text Features Format
```python
# text_features/video_001_text.pt
{
    'y': FloatTensor[1, 1, seq_len, 4096],  # T5-XXL embeddings
    'mask': BoolTensor[1, seq_len]          # Attention mask
}
```

### Video Latents Format
```python
# latents/video_001.pt
FloatTensor[24, T, H, W]  # 24 channels, T frames, H×W resolution
```

---

## 🎯 Key Parameters

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| `--batch-size` (text) | 32 | Increase if you have VRAM |
| `--batch-size` (video) | 1 | VAE encoding is memory-intensive |
| `--max-length` | 300 | T5-XXL max sequence length |
| `--fps` | 24 | Target video FPS |
| `--source` | `real` | Use `real` for actual videos |
| `--compile` | Optional | Use for faster VAE encoding |

---

## ⚠️ Common Issues

### ImportError: No module named 'mug_v'
```bash
# Reinstall requirements (includes MUG-V)
uv pip install -r examples/mugv/data_preparation/requirements.txt
```

### No video reading backend available
```bash
uv pip install av  # or: uv pip install decord
```

### CUDA out of memory (VAE encoding)
```bash
# Ensure batch-size=1
python data_preparation/2_encode_video_latents.py ... --batch-size 1

# Or split videos into smaller clips first
```

### Text encoder downloads model every time
```bash
# Pre-download T5-XXL model
python -c "from transformers import T5EncoderModel; T5EncoderModel.from_pretrained('DeepFloyd/t5-v1_1-xxl')"
```

---

## 🔗 Links

- **Full Documentation**: [README.md](README.md)
- **Environment Setup**: [README.md#environment-setup](README.md#environment-setup)
- **Validation Report**: [VALIDATION_REPORT.md](VALIDATION_REPORT.md)
- **Main Repository README**: [../README.md](../README.md)

---

## 💡 Tips

1. **Pre-download models** to avoid network issues during processing
2. **Use separate environment** from training (different dependencies)
3. **Process in parallel** for large datasets (split by GPU)
4. **Verify dataset** before starting training (save time)
5. **Check [README.md#environment-setup](README.md#environment-setup)** for troubleshooting guide

---

**Quick Help:**
```bash
# Show help for any script
python data_preparation/1_encode_text_features.py --help
python data_preparation/2_encode_video_latents.py --help
python data_preparation/3_generate_training_csv.py --help
```

---

**Last Updated:** 2025-10-16
