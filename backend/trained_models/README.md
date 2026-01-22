# Trained Models Directory

This directory contains pre-trained models for deep learning fusion methods.

## Models

- **dnn_fusion_model.pth**: DNN-based fusion model
- **cnn_fusion_model.pth**: CNN-based fusion model
- **densefuse_model.pth**: DenseFuse fusion model

## Training

To train models on TNO dataset (70-30 split):

```bash
cd backend
python train_models.py --model all
```

### Training Options

```bash
# Train only CNN
python train_models.py --model cnn --epochs-cnn 30

# Train only DenseFuse
python train_models.py --model densefuse --epochs-densefuse 25

# Quick test with limited samples
python train_models.py --model all --max-samples 50

# Custom train-test split
python train_models.py --model all --test-size 0.2  # 80-20 split
```

## Model Info

Models are trained on the TNO Image Fusion Dataset with thermal-visual image pairs.

- **Training Set**: 70% of dataset
- **Test Set**: 30% of dataset
- **Image Size**: 256x256
- **Normalization**: [0, 1] range

The models are automatically loaded by the Flask API on startup if they exist in this directory.
