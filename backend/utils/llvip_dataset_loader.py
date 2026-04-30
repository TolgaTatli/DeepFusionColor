"""
LLVIP Dataset Loader
====================
LLVIP dataset'inden thermal-visual (infrared-visible) görüntü çiftlerini yükler.
Train-Test split yapısını kullanır ve PyTorch Dataset sağlar.

Dataset Yapısı:
- infrared/train/, infrared/test/
- visible/train/, visible/test/
- Thermal ve Visible görüntüleri ayrı klasörlerde
- Eşleşen görüntüler aynı isimde
"""

import os
import numpy as np
from PIL import Image
import glob
from pathlib import Path
from typing import Tuple, List


class LLVIPDatasetLoader:
    """
    LLVIP dataset'i yükler ve train-test split kullanır
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
        
        print(f"\n[LLVIP Dataset]")
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
    
    
    def get_train_data(self, target_size=(256, 256), normalize=True, max_samples=None):
        """
        Training veri yükle
        
        Parametreler:
        ------------
        target_size : tuple
            Görüntü boyutlandırma ("
        
        normalize : bool
            Normalizasyon (0-1 aralığı)
        
        max_samples : int
            Maksimum örnek sayısı
        
        Returns:
        -------
        (ir_images, vis_images) : numpy arrays
        """
        return self._load_image_pairs(
            self.train_pairs,
            target_size=target_size,
            normalize=normalize,
            max_samples=max_samples
        )
    
    
    def get_test_data(self, target_size=(256, 256), normalize=True, max_samples=None):
        """
        Test veri yükle
        
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
        (ir_images, vis_images) : numpy arrays
        """
        return self._load_image_pairs(
            self.test_pairs,
            target_size=target_size,
            normalize=normalize,
            max_samples=max_samples
        )
    
    
    def _load_image_pairs(self, pairs, target_size=(256, 256), normalize=True, max_samples=None):
        """
        IR-VIS görüntü çiftlerini yükle ve işle
        """
        ir_images = []
        vis_images = []
        
        # Örnek sayısını sınırla
        if max_samples:
            pairs = pairs[:max_samples]
        
        print(f"  Loading {len(pairs)} image pairs...")
        
        for i, (ir_path, vis_path) in enumerate(pairs):
            if (i + 1) % max(1, len(pairs) // 5) == 0:
                print(f"    {i+1}/{len(pairs)}")
            
            try:
                # Görüntüleri yükle
                ir_img = Image.open(ir_path).convert('L')  # Grayscale
                vis_img = Image.open(vis_path).convert('L')  # Grayscale
                
                # Boyutlandır
                ir_img = ir_img.resize(target_size, Image.Resampling.LANCZOS)
                vis_img = vis_img.resize(target_size, Image.Resampling.LANCZOS)
                
                # Numpy array'e dönüştür
                ir_array = np.array(ir_img, dtype=np.float32)
                vis_array = np.array(vis_img, dtype=np.float32)
                
                # Normalize et
                if normalize:
                    ir_array = ir_array / 255.0
                    vis_array = vis_array / 255.0
                
                ir_images.append(ir_array)
                vis_images.append(vis_array)
                
            except Exception as e:
                print(f"    Error: {ir_path} failed to load: {e}")
        
        ir_images = np.array(ir_images)
        vis_images = np.array(vis_images)
        
        print(f"  OK: {len(ir_images)} pairs loaded")
        print(f"    Shape: {ir_images.shape}")
        
        return ir_images, vis_images
    
    
    def get_data_generators(self, target_size=(256, 256), batch_size=32):
        """
        Train ve test veri oluşturucularını (generators) döndür
        
        Returns:
        -------
        (train_ir, train_vis, test_ir, test_vis) : arrays
        """
        train_ir, train_vis = self.get_train_data(target_size=target_size)
        test_ir, test_vis = self.get_test_data(target_size=target_size)
        
        return train_ir, train_vis, test_ir, test_vis
