"""
LLVIP Dataset Loader with YCbCr Color Space Support
===================================================
Bu versiyon, RGB görünür görüntülerini YCbCr uzayına çevirerek,
yalnızca luminance (Y) kanalını füzyon ağına besler.

Bu, renkli çıktı elde etmek için önerilir:
1. IR (grayscale) + Y (visible luminance) → Fused Luminance
2. Fused Luminance + Original Cb/Cr → Colorized RGB Output

Dataset Yapısı:
- infrared/train/, infrared/test/
- visible/train/, visible/test/
- Thermal görseller grayscale, Visible görseller RGB (3-channel)
"""

import os
import numpy as np
from PIL import Image
import glob
from pathlib import Path
from typing import Tuple, List
import torch
from torch.utils.data import Dataset, DataLoader


class LLVIPYCbCrDatasetLoader:
    """
    LLVIP dataset'i YCbCr formatında yükler ve train-test split kullanır
    """
    
    def __init__(self, dataset_root, train_size=0.7, random_state=42):
        """
        Parametreler:
        ------------
        dataset_root : str
            LLVIP dataset ana dizini (infrared/ ve visible/ içermelidir)
            
        train_size : float
            Training seti oranı (0.7 = %70)
            
        random_state : int
            Random seed (tekrarlanabilirlik için)
        """
        self.dataset_root = dataset_root
        self.train_size = train_size
        self.random_state = random_state
        
        # Klasörleri kur
        self.ir_train_path = os.path.join(dataset_root, 'infrared', 'train')
        self.ir_test_path = os.path.join(dataset_root, 'infrared', 'test')
        self.vis_train_path = os.path.join(dataset_root, 'visible', 'train')
        self.vis_test_path = os.path.join(dataset_root, 'visible', 'test')
        
        # Klasörlerin var olduğunu kontrol et
        for path in [self.ir_train_path, self.ir_test_path, self.vis_train_path, self.vis_test_path]:
            if not os.path.exists(path):
                raise ValueError(f"Dataset klasörü bulunamadı: {path}")
        
        # Görüntü çiftlerini bul
        self.train_pairs = self._find_image_pairs('train')
        self.test_pairs = self._find_image_pairs('test')
        
        print(f"\n[LLVIP YCbCr Dataset]")
        print(f"  Train pairs: {len(self.train_pairs)}")
        print(f"  Test pairs: {len(self.test_pairs)}")
        print(f"  Total pairs: {len(self.train_pairs) + len(self.test_pairs)}")
    
    
    def _find_image_pairs(self, split: str) -> List[Tuple[str, str]]:
        """
        Dataset'teki IR-VIS görüntü çiftlerini bulur
        
        Args:
            split: 'train' veya 'test'
        
        Returns:
            [(ir_path, vis_path), ...] listesi
        """
        pairs = []
        
        if split == 'train':
            ir_dir = self.ir_train_path
            vis_dir = self.vis_train_path
        else:
            ir_dir = self.ir_test_path
            vis_dir = self.vis_test_path
        
        # IR görüntülerini al
        ir_images = sorted(os.listdir(ir_dir))
        
        for ir_file in ir_images:
            # Desteklenen formatlara bak
            if not ir_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif')):
                continue
            
            # VIS görüntüsünü bul (aynı isimde)
            vis_file = ir_file
            vis_path = os.path.join(vis_dir, vis_file)
            
            if os.path.exists(vis_path):
                ir_path = os.path.join(ir_dir, ir_file)
                pairs.append((ir_path, vis_path))
        
        return pairs
    
    
    def _rgb_to_ycbcr(self, rgb_image):
        """
        RGB görüntüyü YCbCr'a çevirir
        
        Returns:
            Y: [H, W] luminance
            Cb: [H, W] chrominance blue
            Cr: [H, W] chrominance red
        """
        # uint8 formatında ise float32'ye çevir
        if rgb_image.dtype == np.uint8:
            rgb_float = rgb_image.astype(np.float32) / 255.0
        else:
            rgb_float = rgb_image.astype(np.float32)
            if rgb_float.max() > 1.0:
                rgb_float = rgb_float / 255.0
        
        R = rgb_float[:, :, 0]
        G = rgb_float[:, :, 1]
        B = rgb_float[:, :, 2]
        
        # ITU-R BT.601 standardı
        Y = 0.299 * R + 0.587 * G + 0.114 * B
        Cb = -0.169 * R - 0.331 * G + 0.5 * B + 0.5
        Cr = 0.5 * R - 0.419 * G - 0.081 * B + 0.5
        
        return Y.astype(np.float32), Cb.astype(np.float32), Cr.astype(np.float32)
    
    
    def get_train_data_ycbcr(self, target_size=(256, 256), normalize=True, max_samples=None):
        """
        Training veri yükle (YCbCr formatında)
        
        Parametreler:
        ------------
        target_size : tuple
            Görüntü boyutlandırma
        
        normalize : bool
            Normalizasyon (0-1 aralığı)
        
        max_samples : int
            Maksimum örnek sayısı
        
        Returns:
        -------
        (ir_images, y_images, cb_images, cr_images) : numpy arrays
            ir_images: [N, H, W] grayscale IR
            y_images: [N, H, W] luminance channels from visible
            cb_images: [N, H, W] Cb channels (stored for colorization)
            cr_images: [N, H, W] Cr channels (stored for colorization)
        """
        return self._load_image_pairs_ycbcr(
            self.train_pairs,
            target_size=target_size,
            normalize=normalize,
            max_samples=max_samples
        )
    
    
    def get_test_data_ycbcr(self, target_size=(256, 256), normalize=True, max_samples=None):
        """
        Test veri yükle (YCbCr formatında)
        
        Returns:
        -------
        (ir_images, y_images, cb_images, cr_images) : numpy arrays
        """
        return self._load_image_pairs_ycbcr(
            self.test_pairs,
            target_size=target_size,
            normalize=normalize,
            max_samples=max_samples
        )
    
    
    def _load_image_pairs_ycbcr(self, pairs, target_size, normalize, max_samples):
        """
        Görüntü çiftlerini YCbCr formatında yükler
        """
        import cv2
        
        ir_images = []
        y_images = []
        cb_images = []
        cr_images = []
        
        count = 0
        for ir_path, vis_path in pairs:
            if max_samples and count >= max_samples:
                break
            
            try:
                # IR görüntüsünü grayscale yükle
                ir_img = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
                
                # Visible görüntüsünü RGB yükle
                vis_img = cv2.imread(vis_path, cv2.IMREAD_COLOR)
                vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
                
                if ir_img is None or vis_img is None:
                    continue
                
                # Aynı boyuta getir
                h1, w1 = ir_img.shape[:2]
                h2, w2 = vis_img.shape[:2]
                target_h, target_w = target_size
                
                ir_img = cv2.resize(ir_img, (target_w, target_h), interpolation=cv2.INTER_AREA)
                vis_img = cv2.resize(vis_img, (target_w, target_h), interpolation=cv2.INTER_AREA)
                
                # RGB'yi YCbCr'a çevir
                Y, Cb, Cr = self._rgb_to_ycbcr(vis_img)
                
                # Normalizasyon
                if normalize:
                    ir_img = ir_img.astype(np.float32) / 255.0
                    # Y, Cb, Cr zaten [0, 1] aralığında
                else:
                    ir_img = ir_img.astype(np.float32)
                
                ir_images.append(ir_img)
                y_images.append(Y)
                cb_images.append(Cb)
                cr_images.append(Cr)
                
                count += 1
                
            except Exception as e:
                print(f"  ⚠️ Hata yükleme: {ir_path} - {str(e)}")
                continue
        
        return (
            np.array(ir_images),
            np.array(y_images),
            np.array(cb_images),
            np.array(cr_images)
        )


class LLVIPYCbCrTorchDataset(Dataset):
    """
    PyTorch Dataset sınıfı - YCbCr formatında
    DataLoader ile kullanılır
    """
    
    def __init__(self, ir_images, y_images, cb_images=None, cr_images=None, 
                 patch_size=None, stride=None):
        """
        Parametreler:
        ------------
        ir_images : numpy.ndarray [N, H, W]
            Infrared görüntüler
        y_images : numpy.ndarray [N, H, W]
            Y (luminance) channels
        cb_images, cr_images : numpy.ndarray [N, H, W], optional
            Cb ve Cr channels (post-processing için)
        patch_size : int, optional
            Patch boyutu (None ise full image)
        stride : int, optional
            Patch stride
        """
        self.ir_images = ir_images
        self.y_images = y_images
        self.cb_images = cb_images
        self.cr_images = cr_images
        
        if patch_size is not None:
            # Patch'lere böl
            self.patches = self._create_patches(patch_size, stride)
        else:
            # Full images
            self.patches = [(i, None, None, None, None) 
                           for i in range(len(ir_images))]
    
    
    def _create_patches(self, patch_size, stride):
        """
        Görüntüleri patch'lere böler
        """
        patches = []
        
        for img_idx in range(len(self.ir_images)):
            h, w = self.ir_images[img_idx].shape
            
            for i in range(0, h - patch_size + 1, stride):
                for j in range(0, w - patch_size + 1, stride):
                    patches.append((img_idx, i, j, patch_size, patch_size))
        
        return patches
    
    
    def __len__(self):
        return len(self.patches)
    
    
    def __getitem__(self, idx):
        """
        Returns:
            ir_tensor: [1, H, W] tensor
            y_tensor: [1, H, W] tensor
            cb_tensor: [1, H, W] tensor (veya None)
            cr_tensor: [1, H, W] tensor (veya None)
        """
        img_idx, i, j, h, w = self.patches[idx]
        
        if i is None:
            # Full image
            ir = self.ir_images[img_idx]
            y = self.y_images[img_idx]
            cb = self.cb_images[img_idx] if self.cb_images is not None else None
            cr = self.cr_images[img_idx] if self.cr_images is not None else None
        else:
            # Patch
            ir = self.ir_images[img_idx][i:i+h, j:j+w]
            y = self.y_images[img_idx][i:i+h, j:j+w]
            cb = self.cb_images[img_idx][i:i+h, j:j+w] if self.cb_images is not None else None
            cr = self.cr_images[img_idx][i:i+h, j:j+w] if self.cr_images is not None else None
        
        # Tensor'e çevir [1, H, W]
        ir_tensor = torch.tensor(ir, dtype=torch.float32).unsqueeze(0)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
        cb_tensor = torch.tensor(cb, dtype=torch.float32).unsqueeze(0) if cb is not None else None
        cr_tensor = torch.tensor(cr, dtype=torch.float32).unsqueeze(0) if cr is not None else None
        
        return ir_tensor, y_tensor, cb_tensor, cr_tensor


# Backward compatibility - istemerse eski API ile de kullanabilir
class LLVIPDatasetLoaderYCbCr(LLVIPYCbCrDatasetLoader):
    """
    Eski isim ile uyumluluk için alias
    """
    pass
