"""
Colorized Training Script
=========================
Bu script, YCbCr color space separation yaparak modelleri eğitir.

Pipeline:
1. LLVIP dataset'i YCbCr formatında yükle
2. IR + Y (luminance) kanallarını ağa besle
3. Network fused luminance çıktısını ver
4. Post-processing için Cb/Cr kanallarını sakla

Kullanım:
    python colorized_train.py --model cnn --epochs 30 --batch_size 16
    python colorized_train.py --model dnn --learning_rate 0.0005
    python colorized_train.py --model densefuse --target_size 256 256
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Backend modüllerini import et
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from utils.llvip_ycbcr_dataset_loader import LLVIPYCbCrDatasetLoader, LLVIPYCbCrTorchDataset
from models.cnn_fusion import CNNFusionNet
from models.dnn_fusion import DNNFusionNet
from models.densefuse_fusion import DenseFuseNet


# Dizinleri ayarla
BACKEND_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.dirname(BACKEND_DIR)
DATASET_DIR = os.path.join(PROJECT_DIR, 'LLVIP')
MODELS_DIR = os.path.join(BACKEND_DIR, 'trained_models')

os.makedirs(MODELS_DIR, exist_ok=True)


class ColorizedFusionTrainer:
    """
    YCbCr color space separation ile modelleri eğitir
    """
    
    def __init__(self, model, model_type='cnn', device=None, learning_rate=0.001):
        """
        Parametreler:
        ------------
        model : nn.Module
            Füzyon modeli
        model_type : str
            'cnn', 'dnn', veya 'densefuse'
        device : torch.device
            'cuda' veya 'cpu'
        learning_rate : float
            Learning rate
        """
        self.model = model
        self.model_type = model_type
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lr = learning_rate
        
        self.model.to(self.device)
        
        # Loss ve optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        print(f"[Trainer] Model: {model_type}")
        print(f"[Trainer] Device: {self.device}")
        print(f"[Trainer] Learning rate: {self.lr}")
    
    
    def train_epoch(self, train_loader):
        """
        Bir epoch eğitim yap
        """
        self.model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc='Training')
        
        for batch_idx, (ir_tensor, y_tensor, cb_tensor, cr_tensor) in enumerate(progress_bar):
            # Tensors to device
            ir_tensor = ir_tensor.to(self.device)
            y_tensor = y_tensor.to(self.device)
            
            # Forward pass
            # Network IR + Y kanallarını alır ve fused luminance üretir
            fused_luminance = self.model(ir_tensor, y_tensor)
            
            # Loss hesapla
            # Target: IR ve Y'nin ortalaması (self-supervised)
            target = (ir_tensor + y_tensor) / 2.0
            loss = self.criterion(fused_luminance, target)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Progress bar güncelle
            progress_bar.set_postfix({'Loss': f'{loss.item():.6f}'})
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    
    def validate(self, val_loader):
        """
        Validation set üzerinde değerlendirme yap
        """
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for ir_tensor, y_tensor, cb_tensor, cr_tensor in val_loader:
                ir_tensor = ir_tensor.to(self.device)
                y_tensor = y_tensor.to(self.device)
                
                fused_luminance = self.model(ir_tensor, y_tensor)
                target = (ir_tensor + y_tensor) / 2.0
                loss = self.criterion(fused_luminance, target)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        return avg_loss
    
    
    def train(self, train_loader, val_loader=None, epochs=20, save_path=None):
        """
        Modeli eğit
        
        Parametreler:
        ------------
        train_loader : DataLoader
            Training veri loader
        val_loader : DataLoader, optional
            Validation veri loader
        epochs : int
            Epoch sayısı
        save_path : str, optional
            Eğitilmiş modeli kaydet
        """
        print(f"\n[Training] Starting training for {epochs} epochs...")
        
        best_loss = float('inf')
        
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            
            val_loss = None
            if val_loader:
                val_loss = self.validate(val_loader)
            
            # Print epoch info
            log_msg = f"Epoch [{epoch}/{epochs}] - Train Loss: {train_loss:.6f}"
            if val_loss:
                log_msg += f" | Val Loss: {val_loss:.6f}"
            print(log_msg)
            
            # Save best model
            if val_loader:
                if val_loss < best_loss:
                    best_loss = val_loss
                    if save_path:
                        torch.save(self.model.state_dict(), save_path)
                        print(f"  ✅ Best model saved to {save_path}")
        
        print("[Training] Completed!")
    
    
    def save(self, path):
        """
        Modeli kaydet
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"[Model saved to {path}]")


def create_model(model_type, device):
    """
    Modeli oluştur
    """
    if model_type == 'cnn':
        model = CNNFusionNet(num_filters=[16, 32, 64], kernel_size=3)
    elif model_type == 'dnn':
        model = DNNFusionNet(hidden_sizes=[256, 128, 64])
    elif model_type == 'densefuse':
        model = DenseFuseNet(growth_rate=16, num_blocks=3, num_layers_per_block=4)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model.to(device)


def main():
    """
    Ana fonksiyon
    """
    parser = argparse.ArgumentParser(
        description='YCbCr color space separation ile füzyon modeli eğit'
    )
    
    parser.add_argument('--dataset_dir', type=str, default=DATASET_DIR,
                       help=f'LLVIP dataset dizini (default: {DATASET_DIR})')
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'dnn', 'densefuse'],
                       help='Eğitilecek model (default: cnn)')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Epoch sayısı (default: 30)')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size (default: 8)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--target_size', type=int, nargs=2, default=[256, 256],
                       help='Hedef görüntü boyutu (height width) (default: 256 256)')
    parser.add_argument('--device', type=str, default=None, choices=['cpu', 'cuda'],
                       help='PyTorch device')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maksimum örnek sayısı (debug için)')
    parser.add_argument('--validation_split', type=float, default=0.1,
                       help='Validation split (default: 0.1)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("COLORIZED FUSION TRAINING WITH YCBCR COLOR SPACE SEPARATION")
    print("=" * 70)
    
    # Device ayarla
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    
    # Dataset yükle
    print(f"\n[Step 1] Loading LLVIP dataset from {args.dataset_dir}...")
    dataset_loader = LLVIPYCbCrDatasetLoader(args.dataset_dir)
    
    target_size = tuple(args.target_size)
    ir_train, y_train, cb_train, cr_train = dataset_loader.get_train_data_ycbcr(
        target_size=target_size,
        normalize=True,
        max_samples=args.max_samples
    )
    
    print(f"  ✅ Training data shape: IR={ir_train.shape}, Y={y_train.shape}")
    
    # Validation split
    num_train = len(ir_train)
    num_val = int(num_train * args.validation_split)
    num_train = num_train - num_val
    
    # Shuffle indices
    indices = np.random.permutation(num_train + num_val)
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]
    
    # Create torch datasets
    train_dataset = LLVIPYCbCrTorchDataset(
        ir_train[train_indices],
        y_train[train_indices],
        cb_train[train_indices],
        cr_train[train_indices]
    )
    
    val_dataset = LLVIPYCbCrTorchDataset(
        ir_train[val_indices],
        y_train[val_indices],
        cb_train[val_indices],
        cr_train[val_indices]
    )
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    print(f"  ✅ Train samples: {len(train_dataset)}")
    print(f"  ✅ Validation samples: {len(val_dataset)}")
    
    # Model oluştur
    print(f"\n[Step 2] Creating {args.model.upper()} model...")
    model = create_model(args.model, device)
    print("  ✅ Model created!")
    
    # Trainer oluştur ve eğit
    print(f"\n[Step 3] Training...")
    trainer = ColorizedFusionTrainer(
        model,
        model_type=args.model,
        device=device,
        learning_rate=args.learning_rate
    )
    
    save_path = os.path.join(MODELS_DIR, f'{args.model}_fusion_colorized.pth')
    trainer.train(
        train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        save_path=save_path
    )
    
    # Modeli kaydet
    print(f"\n[Step 4] Saving final model...")
    trainer.save(save_path)
    
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Target size: {target_size}")
    print(f"Device: {device}")
    print(f"Model saved to: {save_path}")
    print("\nYCbCr Color Space Separation Benefits:")
    print("  • Trains on luminance channel (Y) + IR thermal")
    print("  • Preserves original RGB color information")
    print("  • Produces colorized fusion output")
    print("=" * 70)


if __name__ == '__main__':
    main()
