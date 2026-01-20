"""
VIF (Visual Information Fidelity) Tabanlı Görüntü Fusion
=========================================================
Görsel bilgi sadakati prensibine dayalı fusion yöntemi.
İnsan görsel sistemini modelleyerek çalışır!

Nasıl Çalışır:
- Görüntüleri multi-scale (çoklu ölçek) representations'a ayırır
- Her ölçekte visual saliency (görsel belirginlik) hesaplar
- Daha belirgin bölgeleri seçerek fusion yapar
"""

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter


class VIFFusion:
    """
    VIF (Visual Information Fidelity) tabanlı fusion
    """
    
    def __init__(self, scales=4, sigma=1.5, window_size=11, contrast_threshold=0.02):
        """
        Parametreler:
        ------------
        scales : int
            Piramit seviye sayısı (multi-scale analysis)
            Az seviye = hızlı, basit
            Çok seviye = yavaş, detaylı
            
            Etki: 2 = çok hızlı, basit
                  4 = dengeli (önerilen)
                  6 = çok detaylı, yavaş
                  
        sigma : float
            Gaussian blur standart sapması
            Küçük sigma = keskin, gürültülü
            Büyük sigma = yumuşak, az gürültü
            
            Etki: 0.5 = çok keskin
                  1.5 = dengeli (önerilen)
                  3.0 = çok yumuşak
                  
        window_size : int
            Local variance hesabı için pencere boyutu
            Küçük = hassas, lokal
            Büyük = genel, global
            
            Etki: 7 = hassas detay
                  11 = dengeli (önerilen)
                  15 = genel yapı
                  
        contrast_threshold : float
            Kontrast eşik değeri
            Düşük = daha fazla detay alınır
            Yüksek = sadece yüksek kontrastlı detaylar
            
            Etki: 0.01 = çok hassas
                  0.02 = dengeli (önerilen)
                  0.05 = sadece belirgin detaylar
        """
        self.scales = scales
        self.sigma = sigma
        self.window_size = window_size
        self.contrast_threshold = contrast_threshold
        
        print(f"[VIF Fusion] Config: scales={scales}, sigma={sigma}, "
              f"window_size={window_size}")
    
    def _gaussian_pyramid(self, image, scales):
        """
        Gaussian piramit oluşturur (multi-scale representation)
        
        Gaussian Pyramid: Görüntüyü farklı ölçeklerde temsil eder
        - Level 0: Orijinal görüntü
        - Level 1: 2x küçültülmüş
        - Level 2: 4x küçültülmüş
        - ...
        
        Parametreler:
        ------------
        image : numpy.ndarray
            Görüntü
        scales : int
            Piramit seviye sayısı
            
        Returns:
        -------
        list : Piramit seviyeleri
        """
        pyramid = [image]
        
        for i in range(1, scales):
            # Gaussian blur uygula
            blurred = gaussian_filter(pyramid[-1], sigma=self.sigma)
            # 2x küçült (downsample)
            downsampled = cv2.resize(blurred, 
                                    (blurred.shape[1]//2, blurred.shape[0]//2),
                                    interpolation=cv2.INTER_AREA)
            pyramid.append(downsampled)
        
        return pyramid
    
    def _laplacian_pyramid(self, gaussian_pyramid):
        """
        Laplacian piramit oluşturur (detail extraction)
        
        Laplacian Pyramid: Her seviyedeki detayları çıkarır
        L[i] = G[i] - upsample(G[i+1])
        
        Parametreler:
        ------------
        gaussian_pyramid : list
            Gaussian piramit
            
        Returns:
        -------
        list : Laplacian piramit
        """
        laplacian = []
        
        for i in range(len(gaussian_pyramid) - 1):
            # Bir sonraki seviyeyi büyüt
            upsampled = cv2.resize(gaussian_pyramid[i+1],
                                  (gaussian_pyramid[i].shape[1], 
                                   gaussian_pyramid[i].shape[0]),
                                  interpolation=cv2.INTER_LINEAR)
            
            # Fark al (detaylar)
            lap = gaussian_pyramid[i] - upsampled
            laplacian.append(lap)
        
        # En üst seviyeyi ekle (base)
        laplacian.append(gaussian_pyramid[-1])
        
        return laplacian
    
    def _local_variance(self, image):
        """
        Local variance (yerel varyans) hesaplar
        Yüksek varyans = detaylı/textured bölge
        Düşük varyans = düz/smooth bölge
        
        Parametreler:
        ------------
        image : numpy.ndarray
            Görüntü
            
        Returns:
        -------
        numpy.ndarray : Variance map
        """
        # Mean filtreleme (local ortalama)
        mean = cv2.blur(image, (self.window_size, self.window_size))
        
        # Variance = E[X^2] - E[X]^2
        mean_sq = cv2.blur(image**2, (self.window_size, self.window_size))
        variance = mean_sq - mean**2
        
        return variance
    
    def _saliency_map(self, image):
        """
        Visual saliency (görsel belirginlik) haritası oluşturur
        
        Saliency: Görüntüde dikkat çeken bölgeler
        Yüksek kontrast, kenarlar, texture → yüksek saliency
        
        Parametrele:
        ------------
        image : numpy.ndarray
            Görüntü
            
        Returns:
        -------
        numpy.ndarray : Saliency map
        """
        # Local variance kullan (texture/detail göstergesi)
        variance = self._local_variance(image)
        
        # Gradient magnitude (kenar göstergesi)
        gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(gx**2 + gy**2)
        
        # Saliency = variance + gradient
        # İkisini normalize edip topla
        variance_norm = (variance - variance.min()) / (variance.max() - variance.min() + 1e-10)
        gradient_norm = (gradient_mag - gradient_mag.min()) / (gradient_mag.max() - gradient_mag.min() + 1e-10)
        
        saliency = variance_norm + gradient_norm
        
        return saliency
    
    def fuse(self, img1, img2):
        """
        İki görüntüyü VIF ile birleştirir
        
        Parametreler:
        ------------
        img1, img2 : numpy.ndarray
            Birleştirilecek görüntüler
            
        Returns:
        -------
        numpy.ndarray : Füzyon edilmiş görüntü
        """
        print("[VIF Fusion] Building pyramids...")
        
        # Gaussian piramitler oluştur
        gauss_pyr1 = self._gaussian_pyramid(img1, self.scales)
        gauss_pyr2 = self._gaussian_pyramid(img2, self.scales)
        
        # Laplacian piramitler oluştur
        lap_pyr1 = self._laplacian_pyramid(gauss_pyr1)
        lap_pyr2 = self._laplacian_pyramid(gauss_pyr2)
        
        print("[VIF Fusion] Fusing pyramids...")
        
        # Her seviyeyi füzyon et
        fused_pyramid = []
        
        for i in range(len(lap_pyr1)):
            # Saliency map'ler hesapla
            sal1 = self._saliency_map(np.abs(lap_pyr1[i]))
            sal2 = self._saliency_map(np.abs(lap_pyr2[i]))
            
            # Decision map: hangi görüntüden alınacak
            # Daha yüksek saliency → daha belirgin detaylar
            decision_map = sal1 >= sal2
            
            # Weighted fusion
            # Saliency oranına göre ağırlıklandır
            sal_sum = sal1 + sal2 + 1e-10
            w1 = sal1 / sal_sum
            w2 = sal2 / sal_sum
            
            # Füzyon: ağırlıklı toplam
            fused_level = w1 * lap_pyr1[i] + w2 * lap_pyr2[i]
            
            fused_pyramid.append(fused_level)
        
        print("[VIF Fusion] Reconstructing image...")
        
        # Piramidi reconstruct et (yukarıdan aşağıya)
        fused_image = fused_pyramid[-1]
        
        for i in range(len(fused_pyramid) - 2, -1, -1):
            # Büyüt (upsample)
            upsampled = cv2.resize(fused_image,
                                  (fused_pyramid[i].shape[1], 
                                   fused_pyramid[i].shape[0]),
                                  interpolation=cv2.INTER_LINEAR)
            
            # Detayları ekle
            fused_image = upsampled + fused_pyramid[i]
        
        # [0, 1] aralığına normalize et
        fused_image = np.clip(fused_image, 0, 1)
        
        return fused_image


def vif_fusion(img1, img2, scales=4, sigma=1.5, window_size=11, contrast_threshold=0.02):
    """
    Hızlı VIF fusion fonksiyonu
    
    Parametreler:
    ------------
    img1, img2 : numpy.ndarray
        Birleştirilecek görüntüler
    scales : int
        Piramit seviye sayısı
    sigma : float
        Gaussian blur sigma
    window_size : int
        Local variance pencere boyutu
    contrast_threshold : float
        Kontrast eşiği
        
    Returns:
    -------
    numpy.ndarray : Füzyon edilmiş görüntü
    
    Örnek Kullanım:
    --------------
    # Hızlı test
    result = vif_fusion(thermal_img, visible_img, scales=3, sigma=1.0)
    
    # Yüksek kalite
    result = vif_fusion(thermal_img, visible_img, scales=5, sigma=2.0, window_size=15)
    """
    fusion_model = VIFFusion(scales=scales, sigma=sigma, 
                            window_size=window_size, 
                            contrast_threshold=contrast_threshold)
    return fusion_model.fuse(img1, img2)
