"""
Görüntü İşleme Yardımcı Fonksiyonlar
====================================
Bu modül, görüntü yükleme, kaydetme ve ön işleme fonksiyonlarını içerir.
"""

import cv2
import numpy as np
from PIL import Image
import os


def load_image(image_path, color_mode='grayscale'):
    """
    Görüntü yükleme fonksiyonu
    
    Parametreler:
    ------------
    image_path : str
        Yüklenecek görüntünün dosya yolu
    color_mode : str
        'grayscale' = Gri tonlamalı yükleme (IR/thermal görüntüler için)
        'color' = Renkli yükleme (RGB görüntüler için)
        
    Etki: color_mode değiştirirsen görüntü formatı değişir. Grayscale fusion için daha hızlı
    
    Returns:
    -------
    numpy.ndarray : Yüklenmiş görüntü matrisi
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Görüntü bulunamadı: {image_path}")
    
    if color_mode == 'grayscale':
        # Gri tonlamalı yükleme - termal/IR görüntüler için
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        # Renkli yükleme - görünür spektrum için
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR'den RGB'ye çevir
    
    if img is None:
        raise ValueError(f"Görüntü yüklenemedi: {image_path}")
    
    return img


def normalize_image(image, method='minmax'):
    """
    Görüntü normalizasyonu - değerleri belirli bir aralığa getirir
    
    Parametreler:
    ------------
    image : numpy.ndarray
        Normalize edilecek görüntü
    method : str
        'minmax' = [0, 1] aralığına normalize eder (varsayılan, en yaygın)
        'standard' = Ortalama 0, std 1 olacak şekilde normalize eder
        
    Etki: minmax kullanırsan görüntü daha parlak, standard kullanırsan kontrast artar
    
    Returns:
    -------
    numpy.ndarray : Normalize edilmiş görüntü
    """
    if method == 'minmax':
        # Min-Max normalizasyonu: (x - min) / (max - min)
        # Tüm piksel değerlerini [0, 1] aralığına getirir
        img_min = np.min(image)
        img_max = np.max(image)
        if img_max - img_min == 0:
            return np.zeros_like(image, dtype=np.float32)
        normalized = (image - img_min) / (img_max - img_min)
    elif method == 'standard':
        # Z-score normalizasyonu: (x - mean) / std
        # Ortalamayı 0, standart sapmayı 1 yapar
        mean = np.mean(image)
        std = np.std(image)
        if std == 0:
            return np.zeros_like(image, dtype=np.float32)
        normalized = (image - mean) / std
    else:
        normalized = image
    
    return normalized.astype(np.float32)


def resize_images(img1, img2, target_size=None):
    """
    İki görüntüyü aynı boyuta getirir - fusion için zorunlu
    
    Parametreler:
    ------------
    img1, img2 : numpy.ndarray
        Boyutlandırılacak görüntüler
    target_size : tuple veya None
        Hedef boyut (height, width). None ise daha küçük görüntüye göre ayarlar
        
    Etki: target_size küçültürsen işlem hızlanır ama detay kaybı olur,
          büyütürsen işlem yavaşlar ama detay artar
    
    Returns:
    -------
    tuple : (resized_img1, resized_img2)
    """
    if target_size is None:
        # Her iki görüntünün minimum boyutunu al
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        target_size = (min(h1, h2), min(w1, w2))
    
    # cv2.INTER_AREA: Küçültme için en iyi kalite
    # cv2.INTER_CUBIC: Büyütme için en iyi kalite
    img1_resized = cv2.resize(img1, (target_size[1], target_size[0]), 
                              interpolation=cv2.INTER_AREA)
    img2_resized = cv2.resize(img2, (target_size[1], target_size[0]), 
                              interpolation=cv2.INTER_AREA)
    
    return img1_resized, img2_resized


def save_image(image, output_path, denormalize=True):
    """
    Görüntü kaydetme fonksiyonu
    
    Parametreler:
    ------------
    image : numpy.ndarray
        Kaydedilecek görüntü
    output_path : str
        Kaydedilecek dosya yolu
    denormalize : bool
        True ise [0,1] aralığındaki değerleri [0,255]'e çevirir
        
    Etki: denormalize=False yapıp manuel kontrol edebilirsin
    """
    # Çıktı dizinini oluştur
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if denormalize and image.max() <= 1.0:
        # [0, 1] aralığından [0, 255] aralığına çevir
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    
    # RGB ise BGR'ye çevir (OpenCV formatı)
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(output_path, image)


def preprocess_for_fusion(img1, img2, target_size=(256, 256), normalize=True):
    """
    İki görüntüyü fusion için hazırlar (ön işleme)
    
    Parametreler:
    ------------
    img1, img2 : numpy.ndarray veya str
        Fusion yapılacak görüntüler veya dosya yolları
    target_size : tuple
        Hedef boyut. (256, 256) genelde iyi bir başlangıç
        Küçük boyut = hızlı işlem, Büyük boyut = yüksek kalite
    normalize : bool
        True ise normalizasyon yap
        
    Etki: target_size'ı (128,128) yapsan çok hızlı ama düşük kalite,
          (512,512) yapsan yavaş ama yüksek kalite olur
    
    Returns:
    -------
    tuple : (preprocessed_img1, preprocessed_img2)
    """
    # Eğer string ise yükle
    if isinstance(img1, str):
        img1 = load_image(img1)
    if isinstance(img2, str):
        img2 = load_image(img2)
    
    # Aynı boyuta getir
    img1, img2 = resize_images(img1, img2, target_size)
    
    # Normalize et
    if normalize:
        img1 = normalize_image(img1, method='minmax')
        img2 = normalize_image(img2, method='minmax')
    
    return img1, img2


def convert_to_uint8(image):
    """
    Görüntüyü uint8 formatına çevirir (gösterim için gerekli)
    
    Parametreler:
    ------------
    image : numpy.ndarray
        Çevrilecek görüntü
        
    Returns:
    -------
    numpy.ndarray : uint8 formatında görüntü
    """
    if image.dtype == np.uint8:
        return image
    
    # [0, 1] aralığında ise
    if image.max() <= 1.0:
        return (image * 255).astype(np.uint8)
    
    # Diğer durumlarda normalize et
    img_min = image.min()
    img_max = image.max()
    if img_max - img_min > 0:
        image = (image - img_min) / (img_max - img_min) * 255
    
    return image.astype(np.uint8)


# ============================================================================
# YCbCr Color Space Conversion Functions for Colorized Fusion
# ============================================================================

def rgb_to_ycbcr(rgb_image):
    """
    RGB görüntüyü YCbCr renk uzayına çevirir
    
    YCbCr Açıklaması:
    - Y: Luminance (parlaklık) - siyah-beyaz bilgisi
    - Cb: Chrominance Blue (mavi renk farkı)
    - Cr: Chrominance Red (kırmızı renk farkı)
    
    Parametreler:
    ------------
    rgb_image : numpy.ndarray
        RGB format görüntü [H, W, 3] veya [0-1] veya [0-255] aralığında
        
    Returns:
    -------
    tuple : (Y, Cb, Cr)
        Y: Luminance channel [H, W], float32, [0-1]
        Cb: Chrominance Blue [H, W], float32, [0-1]
        Cr: Chrominance Red [H, W], float32, [0-1]
    """
    # uint8 formatında ise float32'ye çevir
    if rgb_image.dtype == np.uint8:
        rgb_float = rgb_image.astype(np.float32) / 255.0
    else:
        rgb_float = rgb_image.astype(np.float32)
        # Eğer [0, 1] değilse normalize et
        if rgb_float.max() > 1.0:
            rgb_float = rgb_float / 255.0
    
    # RGB to YCbCr dönüşümü (ITU-R BT.601 standardı)
    R = rgb_float[:, :, 0]
    G = rgb_float[:, :, 1]
    B = rgb_float[:, :, 2]
    
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.169 * R - 0.331 * G + 0.5 * B + 0.5
    Cr = 0.5 * R - 0.419 * G - 0.081 * B + 0.5
    
    return Y.astype(np.float32), Cb.astype(np.float32), Cr.astype(np.float32)


def ycbcr_to_rgb(Y, Cb, Cr):
    """
    YCbCr kanallarını RGB görüntüye çevirir
    
    Parametreler:
    ------------
    Y : numpy.ndarray
        Luminance channel [H, W], float32, [0-1]
    Cb : numpy.ndarray
        Chrominance Blue [H, W], float32, [0-1]
    Cr : numpy.ndarray
        Chrominance Red [H, W], float32, [0-1]
        
    Returns:
    -------
    numpy.ndarray : RGB format görüntü [H, W, 3], float32, [0-1]
    """
    # Float32'ye çevir
    Y = Y.astype(np.float32)
    Cb = Cb.astype(np.float32)
    Cr = Cr.astype(np.float32)
    
    # YCbCr to RGB dönüşümü (ITU-R BT.601 standardı)
    R = Y + 1.402 * (Cr - 0.5)
    G = Y - 0.344136 * (Cb - 0.5) - 0.714136 * (Cr - 0.5)
    B = Y + 1.772 * (Cb - 0.5)
    
    # [0, 1] aralığına sınırla (clamp)
    R = np.clip(R, 0, 1)
    G = np.clip(G, 0, 1)
    B = np.clip(B, 0, 1)
    
    # RGB görüntüsü oluştur [H, W, 3]
    rgb_image = np.stack([R, G, B], axis=2)
    
    return rgb_image.astype(np.float32)


def fuse_ycbcr(fused_luminance, original_rgb):
    """
    Füzyon edilmiş luminance ile orijinal RGB görüntünün renk kanallarını birleştirir
    
    Bu fonksiyon, füzyon ağının çıktısı olan luminance kanalını (siyah-beyaz),
    orijinal RGB görüntüsünün renk bilgileriyle birleştirerek renkli çıktı oluşturur.
    
    Parametreler:
    ------------
    fused_luminance : numpy.ndarray
        Füzyon ağının çıktısı - yalnız luminance [H, W], float32, [0-1]
    original_rgb : numpy.ndarray
        Orijinal RGB görüntü [H, W, 3], float32, [0-1]
        
    Returns:
    -------
    numpy.ndarray : Renklendirilmiş füzyon sonucu [H, W, 3], float32, [0-1]
    """
    # Orijinal RGB'den Cb ve Cr kanallarını çıkar
    _, Cb, Cr = rgb_to_ycbcr(original_rgb)
    
    # Füzyon edilmiş luminance ile orijinal renk kanallarını birleştir
    colorized_rgb = ycbcr_to_rgb(fused_luminance, Cb, Cr)
    
    return colorized_rgb


# ============================================================================
# PyTorch Tensor Utilities for YCbCr Conversion
# ============================================================================

def tensor_rgb_to_ycbcr(rgb_tensor):
    """
    PyTorch tensor formatında RGB'den YCbCr'a dönüşüm
    
    Parametreler:
    ------------
    rgb_tensor : torch.Tensor
        Shape: [B, 3, H, W] (batch_size, channels, height, width)
        Values: float32, [0-1] or [0-255]
        
    Returns:
    -------
    tuple : (Y, Cb, Cr)
        Y: [B, 1, H, W] torch.Tensor
        Cb: [B, 1, H, W] torch.Tensor
        Cr: [B, 1, H, W] torch.Tensor
    """
    import torch
    
    # uint8 ise float32'ye çevir
    if rgb_tensor.dtype == torch.uint8:
        rgb_tensor = rgb_tensor.float() / 255.0
    elif rgb_tensor.max() > 1.0:
        rgb_tensor = rgb_tensor / 255.0
    
    # [B, 3, H, W] formatında olduğunu varsay
    R = rgb_tensor[:, 0:1, :, :]  # [B, 1, H, W]
    G = rgb_tensor[:, 1:2, :, :]
    B = rgb_tensor[:, 2:3, :, :]
    
    # ITU-R BT.601 standardı
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.169 * R - 0.331 * G + 0.5 * B + 0.5
    Cr = 0.5 * R - 0.419 * G - 0.081 * B + 0.5
    
    return Y, Cb, Cr


def tensor_ycbcr_to_rgb(Y, Cb, Cr):
    """
    PyTorch tensor formatında YCbCr'dan RGB'ye dönüşüm
    
    Parametreler:
    ------------
    Y : torch.Tensor
        Luminance [B, 1, H, W] float32, [0-1]
    Cb : torch.Tensor
        Chrominance Blue [B, 1, H, W] float32, [0-1]
    Cr : torch.Tensor
        Chrominance Red [B, 1, H, W] float32, [0-1]
        
    Returns:
    -------
    torch.Tensor : RGB görüntü [B, 3, H, W] float32, [0-1]
    """
    import torch
    
    # YCbCr to RGB conversion (ITU-R BT.601)
    R = Y + 1.402 * (Cr - 0.5)
    G = Y - 0.344136 * (Cb - 0.5) - 0.714136 * (Cr - 0.5)
    B = Y + 1.772 * (Cb - 0.5)
    
    # [0, 1] aralığına sınırla
    R = torch.clamp(R, 0, 1)
    G = torch.clamp(G, 0, 1)
    B = torch.clamp(B, 0, 1)
    
    # [B, 3, H, W] formatında birleştir
    rgb_tensor = torch.cat([R, G, B], dim=1)
    
    return rgb_tensor


def tensor_fuse_ycbcr(fused_luminance_tensor, original_rgb_tensor):
    """
    PyTorch tensor formatında füzyon edilmiş luminance ile renk birleştirmesi
    
    Bu fonksiyon, füzyon ağının çıktısı olan luminance tensörünü,
    orijinal RGB tensörünün renk kanallarıyla birleştirerek renkli sonuç oluşturur.
    
    Parametreler:
    ------------
    fused_luminance_tensor : torch.Tensor
        Füzyon ağının çıktısı [B, 1, H, W] float32, [0-1]
    original_rgb_tensor : torch.Tensor
        Orijinal RGB görüntü [B, 3, H, W] float32, [0-1]
        
    Returns:
    -------
    torch.Tensor : Renklendirilmiş sonuç [B, 3, H, W] float32, [0-1]
    """
    # Orijinal RGB'den Cb ve Cr çıkar
    _, Cb, Cr = tensor_rgb_to_ycbcr(original_rgb_tensor)
    
    # Füzyon edilmiş luminance ile birleştir
    colorized_tensor = tensor_ycbcr_to_rgb(fused_luminance_tensor, Cb, Cr)
    
    return colorized_tensor
