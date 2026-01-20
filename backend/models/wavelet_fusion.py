"""
Wavelet Tabanlı Görüntü Fusion
===============================
Wavelet dönüşümü kullanarak görüntü füzyonu yapar.
Geleneksel ama çok etkili bir yöntem!

Nasıl Çalışır:
- Görüntüleri wavelet alt bantlarına ayrıştırır (frekans bileşenleri)
- Her alt bant için farklı füzyon kuralı uygular
- Ters wavelet ile birleştirilmiş görüntüyü oluşturur
"""

import numpy as np
import pywt
from typing import Literal


class WaveletFusion:
    """
    Wavelet Transform tabanlı görüntü füzyon sınıfı
    """
    
    def __init__(self, wavelet='db4', level=3, fusion_rule='max'):
        """
        Parametreler:
        ------------
        wavelet : str
            Wavelet tipi. Seçenekler:
            - 'db4': Daubechies 4 (varsayılan, çoğu durumda iyi)
            - 'haar': Haar wavelet (en basit, hızlı)
            - 'sym4': Symlet 4 (simetrik, kenar koruması iyi)
            - 'coif2': Coiflet 2 (düzgün yeniden yapılandırma)
            
            Etki: haar en hızlı ama düşük kalite, db4 dengeli, sym4 kenarlar için iyi
            
        level : int
            Ayrıştırma seviyesi (1-5 arası önerilir)
            Küçük level = az detay, hızlı işlem
            Büyük level = çok detay, yavaş işlem
            Etki: 1 = temel fusion, 3 = orta detay (önerilen), 5 = maksimum detay
            
        fusion_rule : str
            Füzyon kuralı:
            - 'max': Maksimum değeri al (kontrast artar, gürültü de artabilir)
            - 'mean': Ortalama al (yumuşak geçiş, gürültü azalır)
            - 'weighted': Ağırlıklı ortalama (dengeli sonuç)
            
            Etki: max = keskin ama gürültülü, mean = yumuşak, weighted = dengeli
        """
        self.wavelet = wavelet
        self.level = level
        self.fusion_rule = fusion_rule
        
        print(f"[Wavelet Fusion] Initialized with {wavelet}, level={level}, rule={fusion_rule}")
    
    def fuse(self, img1, img2):
        """
        İki görüntüyü wavelet fusion ile birleştirir
        
        Parametreler:
        ------------
        img1, img2 : numpy.ndarray
            Birleştirilecek görüntüler (aynı boyutta olmalı!)
            
        Returns:
        -------
        numpy.ndarray : Füzyon edilmiş görüntü
        """
        # Görüntüleri float32'ye çevir
        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)
        
        # Wavelet ayrıştırma (decomposition)
        # Her görüntüyü alt bantlara ayırır: (cA, (cH, cV, cD))
        # cA: Approximation (düşük frekans, genel yapı)
        # cH: Horizontal details (yatay detaylar)
        # cV: Vertical details (dikey detaylar)
        # cD: Diagonal details (çapraz detaylar)
        coeffs1 = pywt.wavedec2(img1, self.wavelet, level=self.level)
        coeffs2 = pywt.wavedec2(img2, self.wavelet, level=self.level)
        
        # Füzyon katsayıları
        fused_coeffs = []
        
        # Her seviye için füzyon uygula
        for i in range(len(coeffs1)):
            if i == 0:
                # Approximation coefficients (en düşük frekans)
                # Genelde ortalama alınır çünkü genel yapıyı temsil eder
                fused_coeffs.append(self._fuse_coefficients(coeffs1[i], coeffs2[i], rule='mean'))
            else:
                # Detail coefficients (yüksek frekans - detaylar)
                # Tuple: (cH, cV, cD)
                cH1, cV1, cD1 = coeffs1[i]
                cH2, cV2, cD2 = coeffs2[i]
                
                # Her detay bandı için fusion rule uygula
                fused_cH = self._fuse_coefficients(cH1, cH2, rule=self.fusion_rule)
                fused_cV = self._fuse_coefficients(cV1, cV2, rule=self.fusion_rule)
                fused_cD = self._fuse_coefficients(cD1, cD2, rule=self.fusion_rule)
                
                fused_coeffs.append((fused_cH, fused_cV, fused_cD))
        
        # Ters wavelet dönüşümü (reconstruction)
        # Alt bantları birleştirerek orijinal görüntüyü oluşturur
        fused_image = pywt.waverec2(fused_coeffs, self.wavelet)
        
        # Boyut uyumsuzluğu varsa düzelt (wavelet bazen 1-2 piksel fark yaratır)
        if fused_image.shape != img1.shape:
            fused_image = fused_image[:img1.shape[0], :img1.shape[1]]
        
        # [0, 1] aralığına normalize et
        fused_image = np.clip(fused_image, 0, 1)
        
        return fused_image
    
    def _fuse_coefficients(self, coeff1, coeff2, rule='max'):
        """
        İki katsayı setini birleştirir
        
        Parametreler:
        ------------
        coeff1, coeff2 : numpy.ndarray
            Birleştirilecek wavelet katsayıları
        rule : str
            Füzyon kuralı
            
        Returns:
        -------
        numpy.ndarray : Birleştirilmiş katsayılar
        """
        if rule == 'max':
            # Maksimum seçme kuralı
            # Her pozisyon için daha büyük katsayıyı al
            # Avantaj: Keskin detaylar, kontrast yüksek
            # Dezavantaj: Gürültü artabilir
            return np.maximum(np.abs(coeff1), np.abs(coeff2)) * np.sign(
                np.where(np.abs(coeff1) >= np.abs(coeff2), coeff1, coeff2)
            )
        
        elif rule == 'mean':
            # Ortalama alma kuralı
            # İki katsayının ortalamasını al
            # Avantaj: Yumuşak geçiş, gürültü azaltma
            # Dezavantaj: Detay kaybı olabilir
            return (coeff1 + coeff2) / 2.0
        
        elif rule == 'weighted':
            # Ağırlıklı ortalama
            # Büyük katsayıya daha fazla ağırlık ver
            # Avantaj: Dengeli sonuç
            abs_sum = np.abs(coeff1) + np.abs(coeff2) + 1e-10  # Sıfıra bölmeyi önle
            w1 = np.abs(coeff1) / abs_sum
            w2 = np.abs(coeff2) / abs_sum
            return w1 * coeff1 + w2 * coeff2
        
        else:
            return (coeff1 + coeff2) / 2.0


def wavelet_fusion(img1, img2, wavelet='db4', level=3, fusion_rule='max'):
    """
    Hızlı wavelet fusion fonksiyonu (sınıf kullanmadan)
    
    Parametreler:
    ------------
    img1, img2 : numpy.ndarray
        Birleştirilecek görüntüler
    wavelet : str
        Wavelet tipi ('db4', 'haar', 'sym4', 'coif2')
    level : int
        Ayrıştırma seviyesi (1-5)
    fusion_rule : str
        'max', 'mean', veya 'weighted'
        
    Returns:
    -------
    numpy.ndarray : Füzyon edilmiş görüntü
    
    Örnek Kullanım:
    --------------
    result = wavelet_fusion(thermal_img, visible_img, wavelet='db4', level=3, fusion_rule='max')
    """
    fusion_model = WaveletFusion(wavelet=wavelet, level=level, fusion_rule=fusion_rule)
    return fusion_model.fuse(img1, img2)
