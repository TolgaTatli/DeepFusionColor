"""
DenseFuse - Dense Block Tabanlı Görüntü Fusion
===============================================
EKSTRA YÖNTEM! State-of-the-art deep learning fusion yaklaşımı.
Dense connection kullanarak bilgi akışını maksimize eder.

Nasıl Çalışır:
- DenseNet'ten esinlenmiş mimari (dense blocks)
- Her katman önceki tüm katmanlardan input alır
- Gradient flow ve feature reuse'u iyileştirir
- Encoder-Decoder yapısı ile fusion yapar

Neden DenseFuse:
- CNN'den daha güçlü (dense connections sayesinde)
- Literatürde çok başarılı sonuçlar (SOTA)
- Gradient vanishing problemini çözer
- Feature reuse ile parametre verimliliği
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class DenseBlock(nn.Module):
    """
    Dense Block: Her katman önceki tüm katmanlardan input alır
    """
    
    def __init__(self, in_channels, growth_rate, num_layers):
        """
        Parametreler:
        ------------
        in_channels : int
            Giriş kanal sayısı
            
        growth_rate : int
            Her katmanın ürettiği feature map sayısı
            Küçük = hafif model, hızlı
            Büyük = ağır model, güçlü
            
            Etki: 12 = hafif, hızlı
                  32 = dengeli (önerilen)
                  48 = ağır, güçlü
                  
        num_layers : int
            Dense block içindeki katman sayısı
            Az katman = basit
            Çok katman = karmaşık
            
            Etki: 3 = basit
                  4-5 = dengeli (önerilen)
                  6+ = çok karmaşık
        """
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            # Her katman önceki tüm katmanlardan input alır
            # Input channels = in_channels + i * growth_rate
            layer_input_channels = in_channels + i * growth_rate
            
            layer = nn.Sequential(
                nn.BatchNorm2d(layer_input_channels),
                nn.ReLU(),
                # 3x3 convolution
                nn.Conv2d(layer_input_channels, growth_rate, 
                         kernel_size=3, padding=1, bias=False)
            )
            self.layers.append(layer)
    
    def forward(self, x):
        """
        Dense connection: her katmanın çıktısı sonraki tüm katmanlara gider
        """
        features = [x]
        
        for layer in self.layers:
            # Önceki tüm feature'ları concat et
            concat_features = torch.cat(features, dim=1)
            # Yeni feature üret
            new_feature = layer(concat_features)
            # Feature listesine ekle
            features.append(new_feature)
        
        # Tüm feature'ları birleştir
        return torch.cat(features, dim=1)


class DenseFuseNet(nn.Module):
    """
    DenseFuse Network: Encoder-DenseBlock-Decoder yapısı
    """
    
    def __init__(self, growth_rate=32, num_blocks=3, num_layers_per_block=4):
        """
        Parametreler:
        ------------
        growth_rate : int
            Dense block growth rate
            
        num_blocks : int
            Encoder/decoder'daki dense block sayısı
            Az block = hızlı, basit
            Çok block = yavaş, güçlü
            
            Etki: 2 = basit
                  3 = dengeli (önerilen)
                  4+ = çok karmaşık
                  
        num_layers_per_block : int
            Her block içindeki katman sayısı
        """
        super(DenseFuseNet, self).__init__()
        
        self.growth_rate = growth_rate
        
        # İlk convolution (input processing)
        self.input_conv = nn.Conv2d(1, growth_rate, kernel_size=3, padding=1)
        
        # ENCODER Dense Blocks
        encoder_blocks = []
        in_channels = growth_rate
        
        for i in range(num_blocks):
            block = DenseBlock(in_channels, growth_rate, num_layers_per_block)
            encoder_blocks.append(block)
            # Dense block sonrası channel sayısı artar
            in_channels = in_channels + num_layers_per_block * growth_rate
        
        self.encoder = nn.ModuleList(encoder_blocks)
        
        # FUSION LAYER
        # İki encoder çıktısını birleştirir
        fusion_in_channels = in_channels * 2  # İki görüntüden
        
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        
        # DECODER
        # Encoder'ın tersine çalışır
        decoder_blocks = []
        
        for i in range(num_blocks):
            # Transition layer (channel reduction)
            transition = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            )
            decoder_blocks.append(transition)
            in_channels = in_channels // 2
        
        self.decoder = nn.ModuleList(decoder_blocks)
        
        # Output layer
        self.output_conv = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # [0, 1] aralığına çek
        )
    
    def forward(self, img1, img2):
        """
        İleri geçiş
        
        Parametreler:
        ------------
        img1, img2 : torch.Tensor
            [batch, 1, H, W] şeklinde
        """
        # İlk convolution
        x1 = self.input_conv(img1)
        x2 = self.input_conv(img2)
        
        # Encoder
        for block in self.encoder:
            x1 = block(x1)
            x2 = block(x2)
        
        # Fusion
        fused = torch.cat([x1, x2], dim=1)
        fused = self.fusion(fused)
        
        # Decoder
        for transition in self.decoder:
            fused = transition(fused)
        
        # Output
        output = self.output_conv(fused)
        
        return output


class ImagePairDatasetDense(Dataset):
    """
    DenseFuse için dataset
    """
    
    def __init__(self, img1, img2, patch_size=64, stride=32):
        self.img1 = img1
        self.img2 = img2
        self.patch_size = patch_size
        self.stride = stride
        
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
        
        patch1 = self.img1[i:i+ps, j:j+ps]
        patch2 = self.img2[i:i+ps, j:j+ps]
        
        patch1 = torch.tensor(patch1, dtype=torch.float32).unsqueeze(0)
        patch2 = torch.tensor(patch2, dtype=torch.float32).unsqueeze(0)
        
        # Target: weighted average (biraz img1'e daha fazla ağırlık)
        target = 0.6 * patch1 + 0.4 * patch2
        
        return patch1, patch2, target


class DenseFusion:
    """
    DenseFuse tabanlı görüntü füzyon sınıfı
    """
    
    def __init__(self, growth_rate=32, num_blocks=3, num_layers_per_block=4,
                 epochs=25, batch_size=16, lr=0.0001, patch_size=64):
        """
        Parametreler:
        ------------
        growth_rate : int
            Dense block growth rate (12, 32, 48 gibi)
            
        num_blocks : int
            Dense block sayısı (2-4 arası önerilir)
            
        num_layers_per_block : int
            Her block'taki katman sayısı (3-5 arası)
            
        epochs : int
            Eğitim epoch sayısı
            DenseFuse için 20-30 epoch önerilir
            
        batch_size : int
            Batch boyutu (8-32 arası)
            
        lr : float
            Öğrenme oranı
            DenseFuse için küçük lr önerilir (0.0001-0.001)
            
        patch_size : int
            Patch boyutu (64 veya 128 önerilir)
        """
        self.growth_rate = growth_rate
        self.num_blocks = num_blocks
        self.num_layers_per_block = num_layers_per_block
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patch_size = patch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        
        print(f"[DenseFuse] Device: {self.device}")
        print(f"[DenseFuse] Config: growth_rate={growth_rate}, blocks={num_blocks}, "
              f"layers/block={num_layers_per_block}, epochs={epochs}")
        print(f"[DenseFuse] This is a STATE-OF-THE-ART deep learning fusion method!")
    
    def fuse(self, img1, img2):
        """
        İki görüntüyü DenseFuse ile birleştirir
        
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
        self.model = DenseFuseNet(growth_rate=self.growth_rate,
                                 num_blocks=self.num_blocks,
                                 num_layers_per_block=self.num_layers_per_block).to(self.device)
        
        # Loss ve optimizer
        criterion = nn.MSELoss()
        # Adam optimizer, küçük lr
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Dataset ve DataLoader
        dataset = ImagePairDatasetDense(img1, img2, 
                                       patch_size=self.patch_size, 
                                       stride=self.patch_size//2)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        print(f"[DenseFuse] Training on {len(dataset)} patches...")
        
        # Eğitim
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for patch1, patch2, target in dataloader:
                patch1 = patch1.to(self.device)
                patch2 = patch2.to(self.device)
                target = target.to(self.device)
                
                outputs = self.model(patch1, patch2)
                loss = criterion(outputs, target)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.6f}")
        
        # Inference
        print("[DenseFuse] Generating fused image...")
        self.model.eval()
        with torch.no_grad():
            img1_tensor = torch.tensor(img1, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
            img2_tensor = torch.tensor(img2, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
            
            fused_tensor = self.model(img1_tensor, img2_tensor)
            fused_image = fused_tensor.squeeze().cpu().numpy()
        
        return fused_image


def densefuse_fusion(img1, img2, growth_rate=32, num_blocks=3, 
                    num_layers_per_block=4, epochs=25, batch_size=16, 
                    lr=0.0001, patch_size=64):
    """
    Hızlı DenseFuse fusion fonksiyonu
    
    Parametreler:
    ------------
    img1, img2 : numpy.ndarray
        Birleştirilecek görüntüler
    growth_rate : int
        Dense block growth rate
    num_blocks : int
        Dense block sayısı
    num_layers_per_block : int
        Her block'taki katman sayısı
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
    # Hızlı test (lightweight)
    result = densefuse_fusion(thermal_img, visible_img, 
                             growth_rate=16, num_blocks=2, epochs=15)
    
    # Yüksek kalite (SOTA)
    result = densefuse_fusion(thermal_img, visible_img, 
                             growth_rate=48, num_blocks=4, 
                             num_layers_per_block=5, epochs=30)
    """
    fusion_model = DenseFusion(growth_rate=growth_rate, num_blocks=num_blocks,
                              num_layers_per_block=num_layers_per_block,
                              epochs=epochs, batch_size=batch_size, 
                              lr=lr, patch_size=patch_size)
    return fusion_model.fuse(img1, img2)
