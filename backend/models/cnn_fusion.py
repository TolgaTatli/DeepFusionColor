"""
Convolutional Neural Network (CNN) Tabanlı Görüntü Fusion
==========================================================
CNN kullanarak spatial features (mekansal özellikler) çıkarıp fusion yapar.
DNN'den daha güçlü çünkü komşu pikselleri de dikkate alır!

Nasıl Çalışır:
- Convolutional katmanlar ile her görüntüden features çıkarır
- Features'ları birleştirir
- Decoder ile füzyon edilmiş görüntüyü oluşturur
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class CNNFusionNet(nn.Module):
    """
    CNN mimarisi - Encoder-Decoder yapısı
    """
    
    def __init__(self, num_filters=[16, 32, 64], kernel_size=3):
        """
        Parametreler:
        ------------
        num_filters : list
            Her katmandaki filtre sayısı
            Az filtre = hızlı, basit pattern
            Çok filtre = yavaş, karmaşık pattern
            
            Etki: [8, 16, 32] = hızlı, basit
                  [32, 64, 128] = yavaş, detaylı
                  
        kernel_size : int
            Convolution kernel boyutu (genelde 3 veya 5)
            3 = küçük alan, hızlı
            5 = büyük alan, daha fazla context
            7 = çok büyük alan, yavaş
            
            Etki: Kernel büyüdükçe daha geniş alan görür ama işlem artar
        """
        super(CNNFusionNet, self).__init__()
        
        # Padding hesapla (görüntü boyutunu korumak için)
        # padding = (kernel_size - 1) // 2
        padding = kernel_size // 2
        
        # ENCODER (Feature Extraction)
        # Her görüntü için ayrı encoder (ağırlıkları paylaşır)
        
        encoder_layers = []
        in_channels = 1  # Grayscale görüntü
        
        for num_filter in num_filters:
            # Convolutional layer
            # in_channels: giriş kanal sayısı
            # num_filter: çıkış kanal sayısı (öğrenilen filtre sayısı)
            encoder_layers.append(
                nn.Conv2d(in_channels, num_filter, kernel_size=kernel_size, 
                         padding=padding, stride=1)
            )
            # Batch Normalization (eğitimi hızlandırır)
            encoder_layers.append(nn.BatchNorm2d(num_filter))
            # ReLU aktivasyon
            encoder_layers.append(nn.ReLU())
            in_channels = num_filter
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # FUSION LAYER
        # İki encoder çıktısını birleştirir
        # Concatenation: iki feature map'i birleştirir (channel boyutunda)
        fusion_in = num_filters[-1] * 2  # İki görüntüden gelen features
        
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_in, num_filters[-1], kernel_size=1),  # 1x1 conv (channel fusion)
            nn.BatchNorm2d(num_filters[-1]),
            nn.ReLU()
        )
        
        # DECODER (Reconstruction)
        # Feature'ları tekrar görüntüye çevirir
        decoder_layers = []
        
        for i in range(len(num_filters) - 1, 0, -1):
            # Transpose convolution veya upsampling
            decoder_layers.append(
                nn.Conv2d(num_filters[i], num_filters[i-1], 
                         kernel_size=kernel_size, padding=padding)
            )
            decoder_layers.append(nn.BatchNorm2d(num_filters[i-1]))
            decoder_layers.append(nn.ReLU())
        
        # Son katman: tek kanala dönüştür (fused image)
        decoder_layers.append(
            nn.Conv2d(num_filters[0], 1, kernel_size=kernel_size, padding=padding)
        )
        decoder_layers.append(nn.Sigmoid())  # [0, 1] aralığına çek
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, img1, img2):
        """
        İleri geçiş
        
        Parametreler:
        ------------
        img1, img2 : torch.Tensor
            [batch, 1, H, W] şeklinde tensor
        """
        # Her görüntüden feature çıkar
        feat1 = self.encoder(img1)
        feat2 = self.encoder(img2)
        
        # Feature'ları birleştir (concatenate)
        # Channel dimension boyunca birleştir
        fused_feat = torch.cat([feat1, feat2], dim=1)
        
        # Fusion layer
        fused_feat = self.fusion(fused_feat)
        
        # Decode et (görüntüye geri çevir)
        output = self.decoder(fused_feat)
        
        return output


class ImagePatchDataset(Dataset):
    """
    Görüntüleri patch'lere (parçalara) böler - CNN eğitimi için
    """
    
    def __init__(self, img1, img2, patch_size=64, stride=32):
        """
        Parametreler:
        ------------
        patch_size : int
            Patch boyutu (kare). Büyük patch = az örnek ama fazla context
            Küçük patch = çok örnek ama az context
            
        stride : int
            Patch'ler arası adım. Küçük stride = overlapping patches = daha iyi
        """
        self.img1 = img1
        self.img2 = img2
        self.patch_size = patch_size
        self.stride = stride
        
        # Patch koordinatlarını hesapla
        h, w = img1.shape
        self.patches = []
        
        for i in range(0, h - patch_size + 1, stride):
            for j in range(0, w - patch_size + 1, stride):
                self.patches.append((i, j))
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        i, j = self.patches[idx]
        ps = self.patch_size
        
        # Patch'leri çıkar
        patch1 = self.img1[i:i+ps, j:j+ps]
        patch2 = self.img2[i:i+ps, j:j+ps]
        
        # Tensor'e çevir [1, H, W] formatında
        patch1 = torch.tensor(patch1, dtype=torch.float32).unsqueeze(0)
        patch2 = torch.tensor(patch2, dtype=torch.float32).unsqueeze(0)
        
        # Hedef: basit ortalama (self-supervised)
        target = (patch1 + patch2) / 2.0
        
        return patch1, patch2, target


class CNNFusion:
    """
    CNN tabanlı görüntü füzyon sınıfı
    """
    
    def __init__(self, num_filters=[16, 32, 64], kernel_size=3, 
                 epochs=20, batch_size=16, lr=0.001, patch_size=64):
        """
        Parametreler:
        ------------
        num_filters : list
            Filtre sayıları
            
        kernel_size : int
            Convolution kernel boyutu
            
        epochs : int
            Eğitim epoch sayısı
            CNN için genelde 20-50 epoch iyi
            
        batch_size : int
            Patch batch sayısı
            CNN için genelde 8-32 arası
            
        lr : float
            Öğrenme oranı
            
        patch_size : int
            Patch boyutu
            64 = orta boyut (önerilen)
            128 = büyük, daha fazla context ama yavaş
            32 = küçük, hızlı ama az context
        """
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patch_size = patch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        
        print(f"[CNN Fusion] Device: {self.device}")
        print(f"[CNN Fusion] Config: filters={num_filters}, kernel={kernel_size}, "
              f"epochs={epochs}, patch_size={patch_size}")
    
    def fuse(self, img1, img2):
        """
        İki görüntüyü CNN ile birleştirir
        
        Parametreler:
        ------------
        img1, img2 : numpy.ndarray
            Birleştirilecek görüntüler
            
        Returns:
        -------
        numpy.ndarray : Füzyon edilmiş görüntü
        """
        h, w = img1.shape
        
        # Model oluştur
        self.model = CNNFusionNet(num_filters=self.num_filters, 
                                 kernel_size=self.kernel_size).to(self.device)
        
        # Loss ve optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Dataset ve DataLoader
        dataset = ImagePatchDataset(img1, img2, patch_size=self.patch_size, stride=self.patch_size//2)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        print(f"[CNN Fusion] Training on {len(dataset)} patches...")
        
        # Eğitim döngüsü
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for patch1, patch2, target in dataloader:
                patch1 = patch1.to(self.device)
                patch2 = patch2.to(self.device)
                target = target.to(self.device)
                
                # Forward pass
                outputs = self.model(patch1, patch2)
                loss = criterion(outputs, target)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.6f}")
        
        # Inference
        print("[CNN Fusion] Generating fused image...")
        self.model.eval()
        with torch.no_grad():
            # Tüm görüntüyü işle (batch halinde)
            img1_tensor = torch.tensor(img1, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
            img2_tensor = torch.tensor(img2, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
            
            fused_tensor = self.model(img1_tensor, img2_tensor)
            fused_image = fused_tensor.squeeze().cpu().numpy()
        
        return fused_image
    
    
    def load_pretrained(self, model_path):
        """
        Pre-trained model yükler
        
        Parametreler:
        ------------
        model_path : str
            Model dosya yolu (.pth)
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"[CNN] Loaded pre-trained model (trained on {checkpoint.get('train_samples', 'unknown')} samples)")
    
    
    def predict(self, img1, img2):
        """
        Inference-only (eğitim yapmadan füzyon)
        Pre-trained model ile direkt füzyon yapar
        
        Parametreler:
        ------------
        img1, img2 : numpy.ndarray
            Birleştirilecek görüntüler
            
        Returns:
        -------
        numpy.ndarray : Füzyon edilmiş görüntü
        """
        self.model.eval()
        with torch.no_grad():
            img1_tensor = torch.tensor(img1, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
            img2_tensor = torch.tensor(img2, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
            
            fused_tensor = self.model(img1_tensor, img2_tensor)
            fused_image = fused_tensor.squeeze().cpu().numpy()
        
        return fused_image


def cnn_fusion(img1, img2, num_filters=[16, 32, 64], kernel_size=3, 
               epochs=20, batch_size=16, lr=0.001, patch_size=64):
    """
    Hızlı CNN fusion fonksiyonu
    
    Parametreler:
    ------------
    img1, img2 : numpy.ndarray
        Birleştirilecek görüntüler
    num_filters : list
        Filtre sayıları
    kernel_size : int
        Kernel boyutu
    epochs : int
        Epoch sayısı
    batch_size : int
        Batch boyutu
    lr : float
        Öğrenme oranı
    patch_size : int
        Patch boyutu
        
    Returns:
    -------
    numpy.ndarray : Füzyon edilmiş görüntü
    
    Örnek Kullanım:
    --------------
    # Hızlı test
    result = cnn_fusion(thermal_img, visible_img, num_filters=[8, 16, 32], epochs=10)
    
    # Yüksek kalite
    result = cnn_fusion(thermal_img, visible_img, num_filters=[32, 64, 128], 
                       epochs=30, patch_size=128)
    """
    fusion_model = CNNFusion(num_filters=num_filters, kernel_size=kernel_size,
                            epochs=epochs, batch_size=batch_size, 
                            lr=lr, patch_size=patch_size)
    return fusion_model.fuse(img1, img2)


class CNNFusionTrainer:
    """
    CNN Fusion Model Trainer (Pre-trained model desteği ile)
    """
    
    def __init__(self, num_filters=[16, 32, 64], kernel_size=3, 
                 epochs=20, batch_size=16, lr=0.001, patch_size=64,
                 pretrained_path=None):
        """
        Parametreler:
        ------------
        num_filters : list
            Her katmandaki filtre sayısı
        kernel_size : int
            Kernel boyutu
        epochs : int
            Eğitim epoch sayısı
        batch_size : int
            Batch boyutu
        lr : float
            Öğrenme oranı
        patch_size : int
            Patch boyutu
        pretrained_path : str or None
            Pre-trained model yolu (.pth file)
        """
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patch_size = patch_size
        
        # Model oluştur
        self.model = CNNFusionNet(num_filters, kernel_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Pre-trained model yükle (varsa)
        if pretrained_path and os.path.exists(pretrained_path):
            self.load_pretrained(pretrained_path)
            print(f"[CNN Fusion] Pre-trained model loaded from: {pretrained_path}")
        else:
            print(f"[CNN Fusion] Model created: filters={num_filters}, kernel={kernel_size}, epochs={epochs}")
    
    def train(self, train_images_ir, train_images_vis):
        """
        Modeli TNO dataset ile eğitir
        
        Parametreler:
        ------------
        train_images_ir : numpy.ndarray
            Thermal görüntüler (N, H, W)
        train_images_vis : numpy.ndarray
            Visual görüntüler (N, H, W)
        """
        print(f"[CNN Fusion] Training on {len(train_images_ir)} image pairs...")
        
        # Tüm görüntülerden patch'ler çıkar
        all_patches_ir = []
        all_patches_vis = []
        all_targets = []
        
        for ir_img, vis_img in zip(train_images_ir, train_images_vis):
            dataset = ImagePatchDataset(ir_img, vis_img, self.patch_size, self.patch_size//2)
            for i in range(len(dataset)):
                p1, p2, target = dataset[i]
                all_patches_ir.append(p1.numpy())
                all_patches_vis.append(p2.numpy())
                all_targets.append(target.numpy())
        
        all_patches_ir = torch.tensor(np.array(all_patches_ir), dtype=torch.float32)
        all_patches_vis = torch.tensor(np.array(all_patches_vis), dtype=torch.float32)
        all_targets = torch.tensor(np.array(all_targets), dtype=torch.float32)
        
        # DataLoader oluştur
        dataset = torch.utils.data.TensorDataset(all_patches_ir, all_patches_vis, all_targets)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Loss ve optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Training
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for patch1, patch2, target in dataloader:
                patch1 = patch1.to(self.device)
                patch2 = patch2.to(self.device)
                target = target.to(self.device)
                
                fused = self.model(patch1, patch2)
                loss = criterion(fused, target)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.6f}")
        
        print("[CNN Fusion] Training complete!")
    
    def load_pretrained(self, model_path):
        """
        Pre-trained model yükler
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"[CNN] Loaded pre-trained model (trained on {checkpoint.get('train_samples', 'unknown')} samples)")
    
    def predict(self, img1, img2):
        """
        Inference-only (eğitim yapmadan füzyon)
        """
        self.model.eval()
        with torch.no_grad():
            img1_tensor = torch.tensor(img1, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
            img2_tensor = torch.tensor(img2, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
            
            fused_tensor = self.model(img1_tensor, img2_tensor)
            fused_image = fused_tensor.squeeze().cpu().numpy()
        
        return fused_image

