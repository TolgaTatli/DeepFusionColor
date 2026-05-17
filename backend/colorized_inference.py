"""
Colorized Fusion Inference Script
==================================
Bu script, YCbCr color space separation kullanarak renkli füzyon sonuçları ürütür.

Pipeline:
1. IR (grayscale) ve Visible RGB (renkli) görüntülerini yükle
2. Visible RGB'yi YCbCr'a çevir
3. IR + Y (luminance) kanalını ağa besle
4. Fused luminance çıktısını al
5. Fused luminance + orijinal Cb, Cr → RGB'ye geri dönüştür
6. Renklendirilmiş füzyon sonucunu kaydet

Kullanım:
    python colorized_inference.py --ir_path path/to/ir.jpg --vis_path path/to/visible.jpg --model cnn
"""

import os
import sys
import argparse
import numpy as np
import torch
import cv2
from pathlib import Path

# Backend modüllerini import et
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.image_utils import (
    load_image, save_image, normalize_image, resize_images, convert_to_uint8,
    rgb_to_ycbcr, ycbcr_to_rgb, fuse_ycbcr,
    tensor_rgb_to_ycbcr, tensor_ycbcr_to_rgb, tensor_fuse_ycbcr
)
from models.cnn_fusion import CNNFusionNet
from models.dnn_fusion import DNNFusionNet
from models.densefuse_fusion import DenseFuseNet


class ColorizedFusionInference:
    """
    YCbCr color space separation ile renkli füzyon yapan inference sınıfı
    """
    
    def __init__(self, model_path=None, model_type='cnn', device=None):
        """
        Parametreler:
        ------------
        model_path : str, optional
            Pre-trained model dosya yolu (.pth)
        model_type : str
            'cnn', 'dnn', veya 'densefuse'
        device : torch.device, optional
            'cuda' veya 'cpu'
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        
        # Model yükle
        self.model = self._load_model(model_type, model_path)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"[Colorized Fusion] Model: {model_type}")
        print(f"[Colorized Fusion] Device: {self.device}")
        if model_path:
            print(f"[Colorized Fusion] Pretrained: {model_path}")
    
    
    def _load_model(self, model_type, model_path):
        """
        Modeli yükle (pre-trained veya yeni)
        """
        if model_type == 'cnn':
            model = CNNFusionNet(num_filters=[16, 32, 64], kernel_size=3)
        elif model_type == 'dnn':
            model = DNNFusionNet(hidden_sizes=[256, 128, 64])
        elif model_type == 'densefuse':
            model = DenseFuseNet(growth_rate=16, num_blocks=3, num_layers_per_block=4)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Pre-trained model yükle
        if model_path and os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            print(f"[Colorized Fusion] Loaded pretrained model from: {model_path}")
        
        return model
    
    
    def preprocess_ycbcr(self, ir_image, rgb_image, target_size=(256, 256)):
        """
        Görüntüleri YCbCr formatında hazırla
        
        Parametreler:
        ------------
        ir_image : numpy.ndarray [H, W]
            Grayscale IR görüntü (uint8 veya float32)
        rgb_image : numpy.ndarray [H, W, 3]
            RGB görüntü (uint8 veya float32)
        target_size : tuple
            Hedef boyut (H, W)
        
        Returns:
        -------
        dict : Hazırlanmış görüntüler ve metadata
            {
                'ir_tensor': [1, 1, H, W],
                'y_tensor': [1, 1, H, W],
                'rgb_original': [H, W, 3],
                'target_size': (H, W)
            }
        """
        # Aynı boyuta getir
        ir_resized, rgb_resized = resize_images(ir_image, rgb_image, target_size)
        
        # float32'ye çevir ve normalize et
        ir_float = ir_resized.astype(np.float32) / 255.0 if ir_resized.max() > 1 else ir_resized.astype(np.float32)
        rgb_float = rgb_resized.astype(np.float32) / 255.0 if rgb_resized.max() > 1 else rgb_resized.astype(np.float32)
        
        # RGB'den Y kanalını çıkar
        Y, _, _ = rgb_to_ycbcr(rgb_float)
        
        # Tensor'e çevir [1, 1, H, W] (batch_size=1, channels=1)
        ir_tensor = torch.tensor(ir_float, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        y_tensor = torch.tensor(Y, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        
        return {
            'ir_tensor': ir_tensor,
            'y_tensor': y_tensor,
            'rgb_original': rgb_float,  # float32, [0-1]
            'target_size': target_size
        }
    
    
    def fuse(self, ir_image, rgb_image, target_size=(256, 256)):
        """
        IR ve Visible görüntülerini füsyonla (renkli çıktı)
        
        Parametreler:
        ------------
        ir_image : numpy.ndarray
            IR görüntü [H, W]
        rgb_image : numpy.ndarray
            RGB görüntü [H, W, 3]
        target_size : tuple
            İşlem boyutu
        
        Returns:
        -------
        dict : Füzyon sonuçları
            {
                'fused_luminance': [H, W],
                'colorized_fusion': [H, W, 3],
                'grayscale_fusion': [H, W]
            }
        """
        # Ön işleme
        processed = self.preprocess_ycbcr(ir_image, rgb_image, target_size)
        
        ir_tensor = processed['ir_tensor']  # [1, 1, H, W]
        y_tensor = processed['y_tensor']    # [1, 1, H, W]
        rgb_original = processed['rgb_original']  # [H, W, 3]
        
        # Forward pass
        with torch.no_grad():
            fused_luminance_tensor = self.model(ir_tensor, y_tensor)  # [1, 1, H, W]
        
        # Tensor'i numpy'e çevir
        fused_luminance = fused_luminance_tensor.squeeze().cpu().numpy()  # [H, W]
        
        # Fused luminance'ı [0, 1] aralığında kıstla
        fused_luminance = np.clip(fused_luminance, 0, 1)
        
        # Renklendir (fused luminance + original Cb, Cr)
        colorized_rgb = fuse_ycbcr(fused_luminance, rgb_original)
        
        # Grayscale füzyon (sadece luminance)
        grayscale_fusion = fused_luminance
        
        return {
            'fused_luminance': fused_luminance,      # [H, W], [0, 1]
            'colorized_fusion': colorized_rgb,       # [H, W, 3], [0, 1]
            'grayscale_fusion': grayscale_fusion,    # [H, W], [0, 1]
            'ir_input': ir_tensor.squeeze().cpu().numpy(),
            'y_input': y_tensor.squeeze().cpu().numpy()
        }
    
    
    def save_results(self, results, output_dir='results/colorized'):
        """
        Füzyon sonuçlarını kaydet
        
        Parametreler:
        ------------
        results : dict
            Fuse() fonksiyonunun çıktısı
        output_dir : str
            Çıktı dizini
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Renkli füzyon
        save_image(results['colorized_fusion'], 
                  os.path.join(output_dir, 'colorized_fusion.png'),
                  denormalize=True)
        
        # Grayscale füzyon
        save_image(results['grayscale_fusion'], 
                  os.path.join(output_dir, 'grayscale_fusion.png'),
                  denormalize=True)
        
        # Fused luminance
        save_image(results['fused_luminance'], 
                  os.path.join(output_dir, 'fused_luminance.png'),
                  denormalize=True)
        
        print(f"\n[Results saved to: {output_dir}]")
        print(f"  - colorized_fusion.png: Renkli füzyon sonucu")
        print(f"  - grayscale_fusion.png: Siyah-beyaz füzyon (luminance)")
        print(f"  - fused_luminance.png: Fused luminance channel")


def main():
    """
    Ana fonksiyon
    """
    parser = argparse.ArgumentParser(
        description='YCbCr color space separation ile renkli füzyon'
    )
    
    parser.add_argument('--ir_path', type=str, required=True,
                       help='IR (thermal) görüntü yolu')
    parser.add_argument('--vis_path', type=str, required=True,
                       help='Visible (RGB) görüntü yolu')
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'dnn', 'densefuse'],
                       help='Füzyon modeli (default: cnn)')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Pre-trained model dosya yolu')
    parser.add_argument('--target_size', type=int, nargs=2, default=[256, 256],
                       help='İşlem boyutu (height, width)')
    parser.add_argument('--output_dir', type=str, default='results/colorized',
                       help='Çıktı dizini')
    parser.add_argument('--device', type=str, default=None, choices=['cpu', 'cuda'],
                       help='PyTorch device')
    
    args = parser.parse_args()
    
    # Görüntü yollarını kontrol et
    if not os.path.exists(args.ir_path):
        print(f"❌ IR görüntü bulunamadı: {args.ir_path}")
        return
    
    if not os.path.exists(args.vis_path):
        print(f"❌ Visible görüntü bulunamadı: {args.vis_path}")
        return
    
    print("=" * 70)
    print("COLORIZED FUSION WITH YCBCR COLOR SPACE SEPARATION")
    print("=" * 70)
    
    # Device ayarla
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    
    # Inference nesnesi oluştur
    inference = ColorizedFusionInference(
        model_path=args.model_path,
        model_type=args.model,
        device=device
    )
    
    # Görüntüleri yükle
    print("\n[Step 1] Loading images...")
    ir_image = load_image(args.ir_path, color_mode='grayscale')
    rgb_image = load_image(args.vis_path, color_mode='color')
    
    print(f"  ✅ IR shape: {ir_image.shape}")
    print(f"  ✅ Visible shape: {rgb_image.shape}")
    
    # Füzyon yap
    print("\n[Step 2] Performing fusion with YCbCr color space separation...")
    target_size = tuple(args.target_size)
    results = inference.fuse(ir_image, rgb_image, target_size=target_size)
    print("  ✅ Fusion completed!")
    
    # Sonuçları kaydet
    print("\n[Step 3] Saving results...")
    inference.save_results(results, args.output_dir)
    
    # Özet
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Target size: {target_size}")
    print(f"Output directory: {args.output_dir}")
    print(f"\nColorized Fusion Benefits:")
    print("  • Retains colors from the original visible image")
    print("  • Improved thermal-visual information fusion")
    print("  • Better perceptual quality of fused results")
    print("=" * 70)


if __name__ == '__main__':
    main()
