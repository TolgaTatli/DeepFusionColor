"""
Basit Örnek Script - Tek Füzyon
================================
Bu script tek bir görüntü çifti ile hızlı test için kullanılır.
"""

import os
import sys

# Backend modüllerini import et
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from models.wavelet_fusion import wavelet_fusion
from models.densefuse_fusion import densefuse_fusion
from utils.image_utils import load_image, save_image, preprocess_for_fusion
from metrics.evaluation_metrics import calculate_all_metrics


def simple_fusion_example():
    """
    Basit füzyon örneği - Wavelet kullanarak
    """
    print("=" * 60)
    print("BASIT FÜZYON ÖRNEĞİ")
    print("=" * 60)
    
    # Görüntü yolları (kendi görüntülerinin yolunu buraya yaz)
    thermal_path = "TNO_Image_Fusion_Dataset/TNO_Image_Fusion_Dataset/Athena_images/bunker/IR_Bunker.bmp"
    visible_path = "TNO_Image_Fusion_Dataset/TNO_Image_Fusion_Dataset/Athena_images/bunker/EO_Bunker.bmp"
    
    # Dosyalar var mı kontrol et
    if not os.path.exists(thermal_path):
        print(f"❌ Thermal görüntü bulunamadı: {thermal_path}")
        print("Lütfen görüntü yollarını düzenleyin!")
        return
    
    if not os.path.exists(visible_path):
        print(f"❌ Visible görüntü bulunamadı: {visible_path}")
        print("Lütfen görüntü yollarını düzenleyin!")
        return
    
    # Görüntüleri yükle
    print("\n1. Görüntüler yükleniyor...")
    thermal = load_image(thermal_path)
    visible = load_image(visible_path)
    print(f"   ✅ Thermal: {thermal.shape}")
    print(f"   ✅ Visible: {visible.shape}")
    
    # Ön işleme
    print("\n2. Ön işleme yapılıyor...")
    thermal, visible = preprocess_for_fusion(thermal, visible, target_size=(256, 256))
    print(f"   ✅ İşlenmiş boyut: {thermal.shape}")
    
    # Wavelet Fusion
    print("\n3. Wavelet Fusion uygulanıyor...")
    fused_wavelet = wavelet_fusion(thermal, visible, wavelet='db4', level=3)
    print("   ✅ Füzyon tamamlandı!")
    
    # Metrikleri hesapla
    print("\n4. Metrikler hesaplanıyor...")
    metrics = calculate_all_metrics(thermal, fused_wavelet, visible, verbose=True)
    
    # Kaydet
    print("\n5. Sonuçlar kaydediliyor...")
    os.makedirs('results/example', exist_ok=True)
    
    save_image(thermal, 'results/example/thermal.png')
    save_image(visible, 'results/example/visible.png')
    save_image(fused_wavelet, 'results/example/fused_wavelet.png')
    
    print("   ✅ Thermal kaydedildi: results/example/thermal.png")
    print("   ✅ Visible kaydedildi: results/example/visible.png")
    print("   ✅ Fused kaydedildi: results/example/fused_wavelet.png")
    
    print("\n" + "=" * 60)
    print("BAŞARILI! Sonuçları results/example/ klasöründe görebilirsin!")
    print("=" * 60)


def compare_methods_example():
    """
    İki yöntemi karşılaştırma - Wavelet vs DenseFuse
    """
    print("\n\n" + "=" * 60)
    print("YÖNTEM KARŞILAŞTIRMA ÖRNEĞİ")
    print("=" * 60)
    
    # Görüntü yolları
    thermal_path = "TNO_Image_Fusion_Dataset/TNO_Image_Fusion_Dataset/Athena_images/bunker/IR_Bunker.bmp"
    visible_path = "TNO_Image_Fusion_Dataset/TNO_Image_Fusion_Dataset/Athena_images/bunker/EO_Bunker.bmp"
    
    if not os.path.exists(thermal_path) or not os.path.exists(visible_path):
        print("❌ Görüntüler bulunamadı! simple_fusion_example() fonksiyonunu önce çalıştırın.")
        return
    
    # Görüntüleri yükle ve işle
    print("\n1. Görüntüler hazırlanıyor...")
    thermal = load_image(thermal_path)
    visible = load_image(visible_path)
    thermal, visible = preprocess_for_fusion(thermal, visible, target_size=(256, 256))
    
    # Wavelet Fusion (Hızlı)
    print("\n2. Wavelet Fusion (Geleneksel - Hızlı)...")
    fused_wavelet = wavelet_fusion(thermal, visible)
    metrics_wavelet = calculate_all_metrics(thermal, fused_wavelet, visible, verbose=False)
    
    # DenseFuse (SOTA - Yavaş ama iyi)
    print("\n3. DenseFuse (Deep Learning - SOTA)...")
    print("   ⚠️  Bu işlem birkaç dakika sürebilir...")
    fused_dense = densefuse_fusion(thermal, visible, epochs=15)  # Hızlı test için epoch azaltıldı
    metrics_dense = calculate_all_metrics(thermal, fused_dense, visible, verbose=False)
    
    # Karşılaştırma
    print("\n" + "=" * 60)
    print("KARŞILAŞTIRMA SONUÇLARI")
    print("=" * 60)
    
    print(f"\n{'Metrik':<15} {'Wavelet':<15} {'DenseFuse':<15} {'Kazanan':<15}")
    print("-" * 60)
    
    metrics_to_compare = [
        ('PSNR (dB)', 'psnr_avg', 'higher'),
        ('SSIM', 'ssim_avg', 'higher'),
        ('MSE', 'mse_avg', 'lower'),
        ('MI', 'mi_avg', 'higher'),
        ('Entropy', 'entropy', 'higher'),
        ('SF', 'sf', 'higher')
    ]
    
    for name, key, better in metrics_to_compare:
        val_w = metrics_wavelet[key]
        val_d = metrics_dense[key]
        
        if better == 'higher':
            winner = 'DenseFuse ✅' if val_d > val_w else 'Wavelet ✅'
        else:
            winner = 'DenseFuse ✅' if val_d < val_w else 'Wavelet ✅'
        
        print(f"{name:<15} {val_w:<15.4f} {val_d:<15.4f} {winner:<15}")
    
    # Kaydet
    os.makedirs('results/comparison', exist_ok=True)
    save_image(fused_wavelet, 'results/comparison/fused_wavelet.png')
    save_image(fused_dense, 'results/comparison/fused_densefuse.png')
    
    print("\n" + "=" * 60)
    print("Sonuçlar kaydedildi: results/comparison/")
    print("=" * 60)


if __name__ == '__main__':
    # Basit örnek
    simple_fusion_example()
    
    # İsteğe bağlı: Yöntem karşılaştırması
    user_input = input("\nYöntem karşılaştırması yapmak ister misin? (y/n): ")
    if user_input.lower() == 'y':
        compare_methods_example()
    
    print("\n\n🎉 Tüm örnekler tamamlandı!")
