"""
Model Training Script
=====================
TNO dataset ile CNN ve DenseFuse modellerini eğitir.
70-30 train-test split kullanır.
Eğitilmiş modelleri trained_models/ klasörüne kaydeder.

Kullanım:
    python train_models.py --model cnn
    python train_models.py --model densefuse
    python train_models.py --model all
"""

import os
import sys
import argparse
import numpy as np
import torch

# Backend modüllerini import et
sys.path.append(os.path.dirname(__file__))
from utils.tno_dataset_loader import TNODatasetLoader
from models.dnn_fusion import DNNFusionTrainer
from models.cnn_fusion import CNNFusionTrainer
from models.densefuse_fusion import DenseFuseTrainer


# Dizinleri ayarla
BACKEND_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.dirname(BACKEND_DIR)
DATASET_DIR = os.path.join(PROJECT_DIR, 'TNO_Image_Fusion_Dataset', 'TNO_Image_Fusion_Dataset')
MODELS_DIR = os.path.join(BACKEND_DIR, 'trained_models')

# Models dizinini oluştur
os.makedirs(MODELS_DIR, exist_ok=True)


def train_dnn(dataset_loader, config):
    """
    DNN modelini eğitir ve kaydeder
    
    Parametreler:
    ------------
    dataset_loader : TNODatasetLoader
        Dataset loader
        
    config : dict
        Training konfigürasyonu
    """
    print("\n" + "="*60)
    print("DNN MODEL TRAINING")
    print("="*60)
    
    # Training data yükle
    print("\n[1/4] Loading training data...")
    train_ir, train_vis = dataset_loader.get_train_data(
        target_size=config.get('img_size', (256, 256)),
        normalize=True,
        max_samples=config.get('max_train_samples', None)
    )
    
    # Trainer oluştur
    print("\n[2/4] Creating DNN trainer...")
    trainer = DNNFusionTrainer(
        hidden_sizes=config.get('hidden_sizes', [256, 128, 64]),
        epochs=config.get('epochs', 15),
        batch_size=config.get('batch_size', 1024),
        lr=config.get('lr', 0.001)
    )
    
    # Modeli eğit
    print("\n[3/4] Training model...")
    trainer.train(train_ir, train_vis)
    
    # Modeli kaydet
    print("\n[4/4] Saving model...")
    model_path = os.path.join(MODELS_DIR, 'dnn_fusion_model.pth')
    torch.save({
        'model_state_dict': trainer.model.state_dict(),
        'hidden_sizes': config.get('hidden_sizes', [256, 128, 64]),
        'config': config,
        'train_samples': len(train_ir),
        'dataset_split': f"{int((1-dataset_loader.test_size)*100)}-{int(dataset_loader.test_size*100)}"
    }, model_path)
    
    print(f"\n✅ DNN model saved to: {model_path}")
    print(f"   Trained on {len(train_ir)} image pairs")
    
    return trainer


def train_cnn(dataset_loader, config):
    """
    CNN modelini eğitir ve kaydeder
    
    Parametreler:
    ------------
    dataset_loader : TNODatasetLoader
        Dataset loader
        
    config : dict
        Training konfigürasyonu
    """
    print("\n" + "="*60)
    print("CNN MODEL TRAINING")
    print("="*60)
    
    # Training data yükle
    print("\n[1/4] Loading training data...")
    train_ir, train_vis = dataset_loader.get_train_data(
        target_size=config.get('img_size', (256, 256)),
        normalize=True,
        max_samples=config.get('max_train_samples', None)
    )
    
    # Trainer oluştur
    print("\n[2/4] Creating CNN trainer...")
    trainer = CNNFusionTrainer(
        num_filters=config.get('num_filters', [16, 32, 64]),
        kernel_size=config.get('kernel_size', 3),
        epochs=config.get('epochs', 30),
        batch_size=config.get('batch_size', 16),
        lr=config.get('lr', 0.001),
        patch_size=config.get('patch_size', 64)
    )
    
    # Modeli eğit
    print("\n[3/4] Training model...")
    trainer.train(train_ir, train_vis)
    
    # Modeli kaydet
    print("\n[4/4] Saving model...")
    model_path = os.path.join(MODELS_DIR, 'cnn_fusion_model.pth')
    torch.save({
        'model_state_dict': trainer.model.state_dict(),
        'num_filters': config.get('num_filters', [16, 32, 64]),
        'kernel_size': config.get('kernel_size', 3),
        'config': config,
        'train_samples': len(train_ir),
        'dataset_split': f"{int((1-dataset_loader.test_size)*100)}-{int(dataset_loader.test_size*100)}"
    }, model_path)
    
    print(f"\n✅ CNN model saved to: {model_path}")
    print(f"   Trained on {len(train_ir)} image pairs")
    
    return trainer


def train_densefuse(dataset_loader, config):
    """
    DenseFuse modelini eğitir ve kaydeder
    
    Parametreler:
    ------------
    dataset_loader : TNODatasetLoader
        Dataset loader
        
    config : dict
        Training konfigürasyonu
    """
    print("\n" + "="*60)
    print("DENSEFUSE MODEL TRAINING")
    print("="*60)
    
    # Training data yükle
    print("\n[1/4] Loading training data...")
    train_ir, train_vis = dataset_loader.get_train_data(
        target_size=config.get('img_size', (256, 256)),
        normalize=True,
        max_samples=config.get('max_train_samples', None)
    )
    
    # Trainer oluştur
    print("\n[2/4] Creating DenseFuse trainer...")
    trainer = DenseFuseTrainer(
        growth_rate=config.get('growth_rate', 16),
        num_blocks=config.get('num_blocks', 3),
        num_layers_per_block=config.get('num_layers_per_block', 4),
        epochs=config.get('epochs', 25),
        batch_size=config.get('batch_size', 16),
        lr=config.get('lr', 0.0001),
        patch_size=config.get('patch_size', 64)
    )
    
    # Modeli eğit
    print("\n[3/4] Training model...")
    trainer.train(train_ir, train_vis)
    
    # Modeli kaydet
    print("\n[4/4] Saving model...")
    model_path = os.path.join(MODELS_DIR, 'densefuse_model.pth')
    torch.save({
        'model_state_dict': trainer.model.state_dict(),
        'growth_rate': config.get('growth_rate', 16),
        'num_blocks': config.get('num_blocks', 3),
        'num_layers_per_block': config.get('num_layers_per_block', 4),
        'config': config,
        'train_samples': len(train_ir),
        'dataset_split': f"{int((1-dataset_loader.test_size)*100)}-{int(dataset_loader.test_size*100)}"
    }, model_path)
    
    print(f"\n✅ DenseFuse model saved to: {model_path}")
    print(f"   Trained on {len(train_ir)} image pairs")
    
    return trainer


def evaluate_model(trainer, dataset_loader, model_name):
    """
    Modeli test setinde değerlendirir
    
    Parametreler:
    ------------
    trainer : Trainer object
        Eğitilmiş trainer
        
    dataset_loader : TNODatasetLoader
        Dataset loader
        
    model_name : str
        Model adı (logging için)
    """
    print(f"\n{'='*60}")
    print(f"EVALUATING {model_name.upper()}")
    print(f"{'='*60}")
    
    # Test data yükle
    test_ir, test_vis = dataset_loader.get_test_data(
        target_size=(256, 256),
        normalize=True,
        max_samples=10  # Hızlı test için ilk 10 çift
    )
    
    print(f"\nTesting on {len(test_ir)} image pairs...")
    
    # İlk birkaç görüntüde test et
    for i in range(min(3, len(test_ir))):
        fused = trainer.predict(test_ir[i], test_vis[i])
        print(f"  Sample {i+1}: fused shape = {fused.shape}, range = [{fused.min():.3f}, {fused.max():.3f}]")
    
    print(f"✅ {model_name} evaluation complete!")


def main():
    parser = argparse.ArgumentParser(description='Train image fusion models on TNO dataset')
    parser.add_argument('--model', type=str, default='all', 
                       choices=['dnn', 'cnn', 'densefuse', 'all'],
                       help='Model to train (dnn/cnn/densefuse/all)')
    parser.add_argument('--epochs-dnn', type=int, default=15,
                       help='Number of epochs for DNN')
    parser.add_argument('--epochs-cnn', type=int, default=30,
                       help='Number of epochs for CNN')
    parser.add_argument('--epochs-densefuse', type=int, default=25,
                       help='Number of epochs for DenseFuse')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--test-size', type=float, default=0.3,
                       help='Test set ratio (0.3 = 30%%)')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Max training samples (for quick testing)')
    parser.add_argument('--no-eval', action='store_true',
                       help='Skip evaluation on test set')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("TNO DATASET - MODEL TRAINING")
    print("="*60)
    print(f"Dataset: {DATASET_DIR}")
    print(f"Models will be saved to: {MODELS_DIR}")
    print(f"Train-Test Split: {int((1-args.test_size)*100)}-{int(args.test_size*100)}")
    
    # Dataset yükle
    print("\n[SETUP] Loading TNO dataset...")
    dataset_loader = TNODatasetLoader(
        DATASET_DIR, 
        test_size=args.test_size,
        random_state=42
    )
    
    # DNN Training
    if args.model in ['dnn', 'all']:
        dnn_config = {
            'hidden_sizes': [256, 128, 64],
            'epochs': args.epochs_dnn,
            'batch_size': 1024,
            'lr': 0.001,
            'img_size': (256, 256),
            'max_train_samples': args.max_samples
        }
        
        dnn_trainer = train_dnn(dataset_loader, dnn_config)
        
        if not args.no_eval:
            evaluate_model(dnn_trainer, dataset_loader, 'DNN')
    
    # CNN Training
    if args.model in ['cnn', 'all']:
        cnn_config = {
            'num_filters': [16, 32, 64],
            'kernel_size': 3,
            'epochs': args.epochs_cnn,
            'batch_size': args.batch_size,
            'lr': 0.001,
            'patch_size': 64,
            'img_size': (256, 256),
            'max_train_samples': args.max_samples
        }
        
        cnn_trainer = train_cnn(dataset_loader, cnn_config)
        
        if not args.no_eval:
            evaluate_model(cnn_trainer, dataset_loader, 'CNN')
    
    # DenseFuse Training
    if args.model in ['densefuse', 'all']:
        densefuse_config = {
            'growth_rate': 16,
            'num_blocks': 3,
            'num_layers_per_block': 4,
            'epochs': args.epochs_densefuse,
            'batch_size': args.batch_size,
            'lr': 0.0001,
            'patch_size': 64,
            'img_size': (256, 256),
            'max_train_samples': args.max_samples
        }
        
        densefuse_trainer = train_densefuse(dataset_loader, densefuse_config)
        
        if not args.no_eval:
            evaluate_model(densefuse_trainer, dataset_loader, 'DenseFuse')
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"Models saved to: {MODELS_DIR}")
    print("\nYou can now use these models in the web app!")
    print("The models will be loaded automatically on startup.")


if __name__ == '__main__':
    main()
