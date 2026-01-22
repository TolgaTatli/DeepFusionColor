"""
TNO Image Fusion Dataset Loader
================================
TNO dataset'inden thermal-visual görüntü çiftlerini yükler.
Train-Test split yapar ve PyTorch Dataset sağlar.

Dataset Yapısı:
- IR (Infrared/Thermal) ve VIS (Visual) görüntü çiftleri
- Farklı senaryolar (Athena, Triclobs, DHV, FEL, tank)
- Her klasörde IR ve VIS görüntü çiftleri var
"""

import os
import numpy as np
from PIL import Image
import glob
from sklearn.model_selection import train_test_split


class TNODatasetLoader:
    """
    TNO dataset'i yükler ve train-test split yapar
    """
    
    def __init__(self, dataset_root, test_size=0.3, random_state=42):
        """
        Parametreler:
        ------------
        dataset_root : str
            TNO dataset ana dizini
            
        test_size : float
            Test seti oranı (0.3 = %30)
            
        random_state : int
            Random seed (tekrarlanabilirlik için)
        """
        self.dataset_root = dataset_root
        self.test_size = test_size
        self.random_state = random_state
        
        # Görüntü çiftlerini bul
        self.image_pairs = self._find_image_pairs()
        
        # Train-test split
        self.train_pairs, self.test_pairs = train_test_split(
            self.image_pairs, 
            test_size=test_size, 
            random_state=random_state
        )
        
        print(f"\n[TNO Dataset]")
        print(f"  Total pairs: {len(self.image_pairs)}")
        print(f"  Train pairs: {len(self.train_pairs)} ({100*(1-test_size):.0f}%)")
        print(f"  Test pairs: {len(self.test_pairs)} ({100*test_size:.0f}%)")
    
    
    def _find_image_pairs(self):
        """
        Dataset'teki tüm IR-VIS görüntü çiftlerini bulur
        
        Returns:
        -------
        list : [(ir_path, vis_path), ...]
        """
        pairs = []
        
        # Ana kategorileri gez
        categories = ['Athena_images', 'Triclobs_images', 'DHV_images', 'FEL_images', 'tank']
        
        for category in categories:
            category_path = os.path.join(self.dataset_root, category)
            
            if not os.path.exists(category_path):
                continue
            
            # Alt klasörleri gez
            if category == 'tank':
                # tank klasörü direkt görüntü içerebilir
                subfolders = [category_path]
            else:
                subfolders = [os.path.join(category_path, d) 
                             for d in os.listdir(category_path)
                             if os.path.isdir(os.path.join(category_path, d))]
            
            for subfolder in subfolders:
                # IR ve VIS görüntüleri bul
                ir_images = []
                vis_images = []
                
                # Tüm dosyaları kontrol et
                for file in os.listdir(subfolder):
                    if not file.endswith('.bmp'):
                        continue
                    
                    file_lower = file.lower()
                    file_path = os.path.join(subfolder, file)
                    
                    # IR/Thermal görüntüleri
                    if any(x in file_lower for x in ['_ir.bmp', '_ii.bmp', 'ir_']):
                        ir_images.append(file_path)
                    # Visual görüntüleri
                    elif any(x in file_lower for x in ['_vis.bmp', 'vis_', '_r.bmp', '_rg.bmp']):
                        vis_images.append(file_path)
                
                # IR-VIS çiftlerini eşleştir
                # Basit yaklaşım: alfabetik sırayla eşleştir
                ir_images.sort()
                vis_images.sort()
                
                for ir_img in ir_images:
                    # Her IR için en uygun VIS'i bul
                    # Aynı base name'e sahip olanı tercih et
                    ir_base = os.path.basename(ir_img).replace('_IR', '').replace('_II', '').replace('IR_', '')
                    
                    best_vis = None
                    for vis_img in vis_images:
                        vis_base = os.path.basename(vis_img).replace('_Vis', '').replace('_VIS', '').replace('VIS_', '').replace('_r', '').replace('_rg', '')
                        
                        # Base name benzerliği kontrolü
                        if vis_base in ir_base or ir_base in vis_base:
                            best_vis = vis_img
                            break
                    
                    # Bulamazsan ilk VIS'i al
                    if best_vis is None and len(vis_images) > 0:
                        best_vis = vis_images[0]
                    
                    if best_vis:
                        pairs.append((ir_img, vis_img))
        
        return pairs
    
    
    def load_image(self, path, target_size=(256, 256), normalize=True):
        """
        Görüntü yükler ve preprocess yapar
        
        Parametreler:
        ------------
        path : str
            Görüntü yolu
            
        target_size : tuple
            Hedef boyut (height, width)
            
        normalize : bool
            [0, 1] aralığına normalize et
            
        Returns:
        -------
        numpy.ndarray : Yüklenmiş görüntü
        """
        # Görüntüyü yükle
        img = Image.open(path)
        
        # Resize
        if target_size:
            img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)
        
        # NumPy array'e çevir
        img_array = np.array(img)
        
        # Grayscale'e çevir (eğer RGB ise)
        if len(img_array.shape) == 3:
            img_array = np.mean(img_array, axis=2)
        
        # Normalize
        if normalize:
            img_array = img_array.astype(np.float32) / 255.0
        
        return img_array
    
    
    def get_train_data(self, target_size=(256, 256), normalize=True, max_samples=None):
        """
        Training setini yükler
        
        Parametreler:
        ------------
        target_size : tuple
            Görüntü boyutu
            
        normalize : bool
            Normalize et
            
        max_samples : int or None
            Maksimum örnek sayısı (hızlı test için)
            
        Returns:
        -------
        (ir_images, vis_images) : (numpy.ndarray, numpy.ndarray)
            Shape: (N, H, W)
        """
        pairs = self.train_pairs[:max_samples] if max_samples else self.train_pairs
        
        ir_images = []
        vis_images = []
        
        print(f"[TNO Loader] Loading {len(pairs)} training pairs...")
        
        for i, (ir_path, vis_path) in enumerate(pairs):
            if (i + 1) % 20 == 0:
                print(f"  Loaded {i+1}/{len(pairs)} pairs...")
            
            ir_img = self.load_image(ir_path, target_size, normalize)
            vis_img = self.load_image(vis_path, target_size, normalize)
            
            ir_images.append(ir_img)
            vis_images.append(vis_img)
        
        return np.array(ir_images), np.array(vis_images)
    
    
    def get_test_data(self, target_size=(256, 256), normalize=True, max_samples=None):
        """
        Test setini yükler
        
        Returns:
        -------
        (ir_images, vis_images) : (numpy.ndarray, numpy.ndarray)
        """
        pairs = self.test_pairs[:max_samples] if max_samples else self.test_pairs
        
        ir_images = []
        vis_images = []
        
        print(f"[TNO Loader] Loading {len(pairs)} test pairs...")
        
        for i, (ir_path, vis_path) in enumerate(pairs):
            ir_img = self.load_image(ir_path, target_size, normalize)
            vis_img = self.load_image(vis_path, target_size, normalize)
            
            ir_images.append(ir_img)
            vis_images.append(vis_img)
        
        return np.array(ir_images), np.array(vis_images)


def test_loader():
    """
    Dataset loader'ı test et
    """
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    dataset_root = os.path.join(
        os.path.dirname(__file__), 
        '..', '..', 
        'TNO_Image_Fusion_Dataset', 
        'TNO_Image_Fusion_Dataset'
    )
    
    loader = TNODatasetLoader(dataset_root, test_size=0.3)
    
    # Birkaç örnek yükle
    train_ir, train_vis = loader.get_train_data(max_samples=5)
    test_ir, test_vis = loader.get_test_data(max_samples=2)
    
    print(f"\nTrain shapes: IR={train_ir.shape}, VIS={train_vis.shape}")
    print(f"Test shapes: IR={test_ir.shape}, VIS={test_vis.shape}")
    print(f"Value range: [{train_ir.min():.3f}, {train_ir.max():.3f}]")


if __name__ == '__main__':
    test_loader()
