"""
Deep Neural Network (DNN) Tabanlı Görüntü Fusion
=================================================
Basit ama etkili fully connected neural network kullanarak fusion yapar.

Nasıl Çalışır:
- İki görüntünün piksellerini feature olarak alır
- Fully connected katmanlardan geçirir
- Füzyon edilmiş pikseli tahmin eder
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class DNNFusionNet(nn.Module):
    """
    Basit DNN mimarisi görüntü füzyonu için
    """
    
    def __init__(self, hidden_sizes=[256, 128, 64]):
        """
        Parametreler:
        ------------
        hidden_sizes : list
            Gizli katman boyutları
            Küçük değerler = hızlı ama basit
            Büyük değerler = yavaş ama karmaşık pattern öğrenir
            
            Etki: [128, 64, 32] = hızlı, basit
                  [512, 256, 128] = yavaş, detaylı
        """
        super(DNNFusionNet, self).__init__()
        
        # Giriş: 2 piksel (her görüntüden 1'er)
        layers = []
        input_size = 2
        
        # Gizli katmanlar oluştur
        for hidden_size in hidden_sizes:
            # Linear (fully connected) katman
            layers.append(nn.Linear(input_size, hidden_size))
            # Batch normalization (eğitimi stabilize eder)
            layers.append(nn.BatchNorm1d(hidden_size))
            # ReLU aktivasyon (non-linearity ekler)
            layers.append(nn.ReLU())
            # Dropout (overfitting'i önler)
            # 0.3 = %30 nöron rastgele kapatılır
            # Etki: Düşük değer = overfit riski, Yüksek değer = underfit riski
            layers.append(nn.Dropout(0.3))
            input_size = hidden_size
        
        # Çıkış katmanı: tek piksel değeri
        layers.append(nn.Linear(input_size, 1))
        # Sigmoid: çıkışı [0, 1] aralığına sıkıştırır
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        İleri geçiş (forward pass)
        
        Parametreler:
        ------------
        x : torch.Tensor
            [batch_size, 2] şeklinde tensor (iki görüntüden piksel çiftleri)
        """
        return self.network(x)


class ImagePairDataset(Dataset):
    """
    PyTorch Dataset: görüntü çiftleri için
    """
    
    def __init__(self, img1, img2):
        """
        img1, img2'nin her pikselini eşleştirir
        """
        self.img1_flat = img1.flatten()
        self.img2_flat = img2.flatten()
    
    def __len__(self):
        return len(self.img1_flat)
    
    def __getitem__(self, idx):
        # İki görüntüden ilgili pikselleri al
        x = torch.tensor([self.img1_flat[idx], self.img2_flat[idx]], dtype=torch.float32)
        # Hedef: basit ortalama (self-supervised learning için)
        y = torch.tensor([(self.img1_flat[idx] + self.img2_flat[idx]) / 2], dtype=torch.float32)
        return x, y


class DNNFusion:
    """
    DNN tabanlı görüntü füzyon sınıfı
    """
    
    def __init__(self, hidden_sizes=[256, 128, 64], epochs=10, batch_size=1024, lr=0.001):
        """
        Parametreler:
        ------------
        hidden_sizes : list
            Gizli katman boyutları
            
        epochs : int
            Eğitim epoch sayısı
            Az epoch = hızlı ama yetersiz öğrenme
            Çok epoch = yavaş, overfit riski
            Etki: 5 = hızlı test, 20 = iyi sonuç, 50+ = overfit olabilir
            
        batch_size : int
            Her iterasyonda işlenen örnek sayısı
            Küçük = yavaş ama stabil, Büyük = hızlı ama memory kullanımı fazla
            Etki: 512 = yavaş, stabil, 2048 = hızlı, memory yoğun
            
        lr : float
            Öğrenme oranı (learning rate)
            Küçük = yavaş öğrenme ama stabil, Büyük = hızlı ama unstable
            Etki: 0.0001 = çok yavaş, 0.001 = iyi (önerilen), 0.01 = unstable olabilir
        """
        self.hidden_sizes = hidden_sizes
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        
        print(f"[DNN Fusion] Device: {self.device}")
        print(f"[DNN Fusion] Config: epochs={epochs}, batch_size={batch_size}, lr={lr}")
    
    def fuse(self, img1, img2):
        """
        İki görüntüyü DNN ile birleştirir
        
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
        self.model = DNNFusionNet(hidden_sizes=self.hidden_sizes).to(self.device)
        
        # Loss function ve optimizer
        # MSE: Mean Squared Error - piksel farkının karesi
        criterion = nn.MSELoss()
        # Adam optimizer: adaptive learning rate, genelde en iyi sonucu verir
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Dataset ve DataLoader
        dataset = ImagePairDataset(img1, img2)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Eğitim döngüsü
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                
                # Backward pass ve optimization
                optimizer.zero_grad()  # Gradientleri sıfırla
                loss.backward()  # Gradientleri hesapla
                optimizer.step()  # Ağırlıkları güncelle
                
                total_loss += loss.item()
            
            # Her epoch'ta loss'u göster
            avg_loss = total_loss / len(dataloader)
            if (epoch + 1) % 2 == 0 or epoch == 0:
                print(f"  Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.6f}")
        
        # Inference (tahmin yapma)
        self.model.eval()
        with torch.no_grad():
            # Tüm piksel çiftlerini işle
            img1_flat = torch.tensor(img1.flatten(), dtype=torch.float32)
            img2_flat = torch.tensor(img2.flatten(), dtype=torch.float32)
            inputs = torch.stack([img1_flat, img2_flat], dim=1).to(self.device)
            
            # Batch halinde işle (memory tasarrufu için)
            outputs = []
            for i in range(0, len(inputs), self.batch_size):
                batch = inputs[i:i+self.batch_size]
                output = self.model(batch)
                outputs.append(output.cpu())
            
            # Birleştir ve reshape et
            fused_flat = torch.cat(outputs, dim=0).numpy().flatten()
            fused_image = fused_flat.reshape(h, w)
        
        return fused_image


def dnn_fusion(img1, img2, hidden_sizes=[256, 128, 64], epochs=10, batch_size=1024, lr=0.001):
    """
    Hızlı DNN fusion fonksiyonu
    
    Parametreler:
    ------------
    img1, img2 : numpy.ndarray
        Birleştirilecek görüntüler
    hidden_sizes : list
        Gizli katman boyutları
    epochs : int
        Eğitim epoch sayısı
    batch_size : int
        Batch boyutu
    lr : float
        Öğrenme oranı
        
    Returns:
    -------
    numpy.ndarray : Füzyon edilmiş görüntü
    
    Örnek Kullanım:
    --------------
    # Hızlı test
    result = dnn_fusion(thermal_img, visible_img, epochs=5, batch_size=2048)
    
    # Kaliteli sonuç
    result = dnn_fusion(thermal_img, visible_img, hidden_sizes=[512, 256, 128], 
                       epochs=20, batch_size=1024, lr=0.001)
    """
    fusion_model = DNNFusion(hidden_sizes=hidden_sizes, epochs=epochs, 
                            batch_size=batch_size, lr=lr)
    return fusion_model.fuse(img1, img2)
