"""
Görüntü Füzyon Kalite Metrikleri
=================================
Füzyon sonuçlarını değerlendirmek için kullanılan metrikler.

Metrikler:
1. PSNR (Peak Signal-to-Noise Ratio): Sinyal-gürültü oranı
2. SSIM (Structural Similarity Index): Yapısal benzerlik
3. MSE (Mean Squared Error): Ortalama kare hata
4. MIS (Mutual Information Score): Karşılıklı bilgi skoru
5. EN (Entropy): Bilgi içeriği
6. SF (Spatial Frequency): Uzaysal frekans
"""

import numpy as np
from skimage.metrics import structural_similarity as ssim_calc
from skimage.metrics import peak_signal_noise_ratio as psnr_calc
from skimage.metrics import mean_squared_error as mse_calc
from scipy.stats import entropy


def psnr(img_reference, img_fused, data_range=1.0):
    """
    PSNR - Peak Signal-to-Noise Ratio
    
    Sinyal-gürültü oranını ölçer. Yüksek değer = iyi kalite
    
    Formül: PSNR = 10 * log10(MAX^2 / MSE)
    - MAX: Maksimum piksel değeri
    - MSE: Mean Squared Error
    
    Parametreler:
    ------------
    img_reference : numpy.ndarray
        Referans görüntü (genelde kaynak görüntülerden biri)
    img_fused : numpy.ndarray
        Füzyon edilmiş görüntü
    data_range : float
        Veri aralığı (1.0 = [0,1], 255 = [0,255])
        
    Returns:
    -------
    float : PSNR değeri (dB cinsinden)
            20+ = iyi
            30+ = çok iyi
            40+ = mükemmel
            
    Etki: Piksel değeri farkına çok hassas
          Küçük değişiklikler bile PSNR'yi düşürür
          
    Örnek:
    ------
    psnr_value = psnr(thermal_img, fused_img)
    print(f"PSNR: {psnr_value:.2f} dB")
    """
    try:
        value = psnr_calc(img_reference, img_fused, data_range=data_range)
        return value
    except Exception as e:
        print(f"[PSNR Error] {e}")
        return 0.0


def ssim(img_reference, img_fused, data_range=1.0):
    """
    SSIM - Structural Similarity Index
    
    Yapısal benzerliği ölçer (insan görsel algısına yakın).
    Luminance, contrast ve structure'ı dikkate alır.
    
    Formül: SSIM(x,y) = [l(x,y)]^α * [c(x,y)]^β * [s(x,y)]^γ
    - l: luminance comparison
    - c: contrast comparison
    - s: structure comparison
    
    Parametreler:
    ------------
    img_reference : numpy.ndarray
        Referans görüntü
    img_fused : numpy.ndarray
        Füzyon edilmiş görüntü
    data_range : float
        Veri aralığı
        
    Returns:
    -------
    float : SSIM değeri [-1, 1] aralığında
            0.8+ = iyi
            0.9+ = çok iyi
            0.95+ = mükemmel
            
    Etki: İnsan görsel algısına yakın
          Yapısal değişikliklere hassas
          PSNR'den daha güvenilir (literatürde)
          
    Örnek:
    ------
    ssim_value = ssim(thermal_img, fused_img)
    print(f"SSIM: {ssim_value:.4f}")
    """
    try:
        value = ssim_calc(img_reference, img_fused, data_range=data_range)
        return value
    except Exception as e:
        print(f"[SSIM Error] {e}")
        return 0.0


def mse(img_reference, img_fused):
    """
    MSE - Mean Squared Error
    
    Ortalama kare hatayı ölçer. Düşük değer = iyi
    
    Formül: MSE = (1/N) * Σ(reference - fused)^2
    
    Parametreler:
    ------------
    img_reference : numpy.ndarray
        Referans görüntü
    img_fused : numpy.ndarray
        Füzyon edilmiş görüntü
        
    Returns:
    -------
    float : MSE değeri
            0'a yakın = iyi (mükemmel = 0)
            0.001 altı = mükemmel
            0.01 altı = çok iyi
            0.1 altı = iyi
            
    Etki: Piksel seviyesinde farkı ölçer
          Yapısal bilgiyi dikkate almaz
          
    Örnek:
    ------
    mse_value = mse(thermal_img, fused_img)
    print(f"MSE: {mse_value:.6f}")
    """
    try:
        value = mse_calc(img_reference, img_fused)
        return value
    except Exception as e:
        print(f"[MSE Error] {e}")
        return float('inf')


def mutual_information(img1, img2, bins=256):
    """
    MIS - Mutual Information Score
    
    İki görüntü arasındaki karşılıklı bilgi miktarını ölçer.
    Yüksek değer = daha fazla ortak bilgi
    
    Formül: MI(X,Y) = H(X) + H(Y) - H(X,Y)
    - H(X): X'in entropisi
    - H(Y): Y'nin entropisi
    - H(X,Y): Joint entropy
    
    Parametreler:
    ------------
    img1, img2 : numpy.ndarray
        Karşılaştırılacak görüntüler
    bins : int
        Histogram bins sayısı
        Düşük bins = hızlı ama kaba
        Yüksek bins = yavaş ama detaylı
        
        Etki: 128 = hızlı
              256 = dengeli (önerilen)
              512 = detaylı
        
    Returns:
    -------
    float : Mutual information değeri
            Yüksek = iyi (daha fazla bilgi aktarımı)
            1+ = iyi
            2+ = çok iyi
            
    Etki: Füzyonun ne kadar bilgi koruduğunu gösterir
          Registration kalitesi için de kullanılır
          
    Örnek:
    ------
    mi_value = mutual_information(thermal_img, fused_img)
    print(f"MI: {mi_value:.4f}")
    """
    try:
        # [0,1] aralığındaysa [0,255]'e çevir
        if img1.max() <= 1.0:
            img1 = img1 * 255.0
        if img2.max() <= 1.0:
            img2 = img2 * 255.0
        
        # Görüntüleri [0, bins-1] aralığına getir
        img1_int = np.clip(img1, 0, 255).astype(np.int32)
        img2_int = np.clip(img2, 0, 255).astype(np.int32)
        
        # Histogram hesapla
        hist_1d_1 = np.histogram(img1_int, bins=bins, range=(0, 256))[0]
        hist_1d_2 = np.histogram(img2_int, bins=bins, range=(0, 256))[0]
        
        # Joint histogram (2D)
        hist_2d, _, _ = np.histogram2d(img1_int.flatten(), img2_int.flatten(), 
                                       bins=bins, range=[[0, 256], [0, 256]])
        
        # Normalize et (probability distribution)
        p1 = hist_1d_1 / np.sum(hist_1d_1)
        p2 = hist_1d_2 / np.sum(hist_1d_2)
        p12 = hist_2d / np.sum(hist_2d)
        
        # Entropy hesapla
        # entropy fonksiyonu 0'ları handle eder
        h1 = entropy(p1 + 1e-10)
        h2 = entropy(p2 + 1e-10)
        h12 = entropy(p12.flatten() + 1e-10)
        
        # Mutual information
        mi = h1 + h2 - h12
        
        return mi
    except Exception as e:
        print(f"[MI Error] {e}")
        return 0.0


def image_entropy(image, bins=256):
    """
    EN - Entropy (Bilgi İçeriği)
    
    Görüntünün bilgi miktarını ölçer.
    Yüksek entropy = daha fazla bilgi/detay
    
    Formül: H = -Σ p(i) * log2(p(i))
    - p(i): i'nci piksel değerinin olasılığı
    
    Parametreler:
    ------------
    image : numpy.ndarray
        Görüntü
    bins : int
        Histogram bins
        
    Returns:
    -------
    float : Entropy değeri (bits cinsinden)
            5+ = orta bilgi
            6+ = iyi bilgi
            7+ = yüksek bilgi/detay
            
    Etki: Füzyonun ne kadar detay eklediğini gösterir
          Yüksek entropy genelde daha iyi (daha fazla bilgi)
          
    Örnek:
    ------
    en_fused = image_entropy(fused_img)
    en_thermal = image_entropy(thermal_img)
    print(f"Entropy increase: {en_fused - en_thermal:.4f} bits")
    """
    try:
        # Görüntüyü bins'e böl
        img_int = (image * (bins - 1)).astype(np.int32)
        
        # Histogram
        hist = np.histogram(img_int, bins=bins, range=(0, bins))[0]
        
        # Normalize (probability)
        prob = hist / np.sum(hist)
        
        # Entropy hesapla
        ent = entropy(prob + 1e-10, base=2)  # base=2 → bits cinsinden
        
        return ent
    except Exception as e:
        print(f"[Entropy Error] {e}")
        return 0.0


def spatial_frequency(image):
    """
    SF - Spatial Frequency (Uzaysal Frekans)
    
    Görüntünün keskinliğini ve detay seviyesini ölçer.
    Yüksek SF = keskin kenarlar, bol detay
    
    Formül: SF = sqrt(RF^2 + CF^2)
    - RF: Row Frequency
    - CF: Column Frequency
    
    Parametreler:
    ------------
    image : numpy.ndarray
        Görüntü
        
    Returns:
    -------
    float : SF değeri
            Yüksek = keskin, detaylı görüntü
            10+ = orta
            20+ = iyi
            30+ = çok keskin
            
    Etki: Füzyonun keskinliği artırıp artırmadığını gösterir
          Blur/smooth görüntülerde düşük olur
          
    Örnek:
    ------
    sf_fused = spatial_frequency(fused_img)
    sf_thermal = spatial_frequency(thermal_img)
    sf_visible = spatial_frequency(visible_img)
    print(f"SF improvement: {sf_fused - max(sf_thermal, sf_visible):.2f}")
    """
    try:
        # [0,1] aralığındaysa [0,255]'e çevir (SF için gerekli)
        if image.max() <= 1.0:
            image = image * 255.0
        
        # Row frequency (yatay frekans)
        # Komşu satırlar arası fark
        rf = np.sqrt(np.mean((image[1:, :] - image[:-1, :])**2))
        
        # Column frequency (dikey frekans)
        # Komşu sütunlar arası fark
        cf = np.sqrt(np.mean((image[:, 1:] - image[:, :-1])**2))
        
        # Spatial frequency
        sf = np.sqrt(rf**2 + cf**2)
        
        return sf
    except Exception as e:
        print(f"[SF Error] {e}")
        return 0.0


def calculate_all_metrics(img_reference, img_fused, img_reference2=None, verbose=True):
    """
    Tüm metrikleri hesaplar ve sonuçları döndürür
    
    Parametreler:
    ------------
    img_reference : numpy.ndarray
        Birinci referans görüntü (örn: thermal)
    img_fused : numpy.ndarray
        Füzyon edilmiş görüntü
    img_reference2 : numpy.ndarray veya None
        İkinci referans görüntü (örn: visible)
        Varsa iki referansla da karşılaştırma yapar
    verbose : bool
        True ise sonuçları yazdır
        
    Returns:
    -------
    dict : Tüm metrik sonuçları
    
    Örnek Kullanım:
    --------------
    metrics = calculate_all_metrics(thermal_img, fused_img, visible_img)
    print(f"PSNR: {metrics['psnr_avg']:.2f} dB")
    print(f"SSIM: {metrics['ssim_avg']:.4f}")
    """
    results = {}
    
    # PSNR
    results['psnr_ref1'] = psnr(img_reference, img_fused)
    if img_reference2 is not None:
        results['psnr_ref2'] = psnr(img_reference2, img_fused)
        results['psnr_avg'] = (results['psnr_ref1'] + results['psnr_ref2']) / 2.0
    else:
        results['psnr_avg'] = results['psnr_ref1']
    
    # SSIM
    results['ssim_ref1'] = ssim(img_reference, img_fused)
    if img_reference2 is not None:
        results['ssim_ref2'] = ssim(img_reference2, img_fused)
        results['ssim_avg'] = (results['ssim_ref1'] + results['ssim_ref2']) / 2.0
    else:
        results['ssim_avg'] = results['ssim_ref1']
    
    # MSE
    results['mse_ref1'] = mse(img_reference, img_fused)
    if img_reference2 is not None:
        results['mse_ref2'] = mse(img_reference2, img_fused)
        results['mse_avg'] = (results['mse_ref1'] + results['mse_ref2']) / 2.0
    else:
        results['mse_avg'] = results['mse_ref1']
    
    # Mutual Information
    results['mi_ref1'] = mutual_information(img_reference, img_fused)
    if img_reference2 is not None:
        results['mi_ref2'] = mutual_information(img_reference2, img_fused)
        results['mi_avg'] = (results['mi_ref1'] + results['mi_ref2']) / 2.0
    else:
        results['mi_avg'] = results['mi_ref1']
    
    # Entropy (sadece fused için)
    results['entropy'] = image_entropy(img_fused)
    
    # Spatial Frequency (sadece fused için)
    results['sf'] = spatial_frequency(img_fused)
    
    if verbose:
        print("\n" + "="*60)
        print("GÖRÜNTÜ FÜZYON METRİK SONUÇLARI")
        print("="*60)
        print(f"PSNR (Peak Signal-to-Noise Ratio): {results['psnr_avg']:.2f} dB")
        print(f"  → Yüksek değer iyi (30+ çok iyi, 40+ mükemmel)")
        print(f"\nSSIM (Structural Similarity):      {results['ssim_avg']:.4f}")
        print(f"  → Yüksek değer iyi (0.9+ çok iyi, 0.95+ mükemmel)")
        print(f"\nMSE (Mean Squared Error):           {results['mse_avg']:.6f}")
        print(f"  → Düşük değer iyi (0.01 altı çok iyi)")
        print(f"\nMI (Mutual Information):            {results['mi_avg']:.4f}")
        print(f"  → Yüksek değer iyi (2+ çok iyi)")
        print(f"\nEN (Entropy):                       {results['entropy']:.4f} bits")
        print(f"  → Yüksek değer iyi (7+ yüksek bilgi)")
        print(f"\nSF (Spatial Frequency):             {results['sf']:.4f}")
        print(f"  → Yüksek değer iyi (20+ keskin)")
        print("="*60 + "\n")
    
    return results


# Ek olarak CC (Correlation Coefficient) ekleyelim
def correlation_coefficient(img1, img2):
    """
    CC - Correlation Coefficient (Korelasyon Katsayısı)
    
    İki görüntü arasındaki linear ilişkiyi ölçer.
    
    Parametreler:
    ------------
    img1, img2 : numpy.ndarray
        Karşılaştırılacak görüntüler
        
    Returns:
    -------
    float : CC değeri [-1, 1] aralığında
            1'e yakın = yüksek pozitif korelasyon (iyi)
            0'a yakın = korelasyon yok
            -1'e yakın = negatif korelasyon
    """
    try:
        # Flatten
        img1_flat = img1.flatten()
        img2_flat = img2.flatten()
        
        # Correlation coefficient
        cc = np.corrcoef(img1_flat, img2_flat)[0, 1]
        
        return cc
    except Exception as e:
        print(f"[CC Error] {e}")
        return 0.0
