"""
DeepFusionColor - Kapsamlı Test Script'i
=========================================
LLVIP dataset ile tüm füzyön yöntemlerini test eder.

Bu script:
1. LLVIP dataset'inden görüntüleri yükler
2. Tüm füzyon yöntemlerini uygular
3. Metrikleri hesaplar
4. Sonuçları kaydeder ve karşılaştırır
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# Backend modüllerini import et
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from models.wavelet_fusion import wavelet_fusion
from models.dnn_fusion import dnn_fusion
from models.cnn_fusion import cnn_fusion
from models.latentlrr_fusion import latentlrr_fusion
from models.densefuse_fusion import densefuse_fusion
from metrics.evaluation_metrics import calculate_all_metrics
from utils.image_utils import load_image, save_image, preprocess_for_fusion


# Dataset ve sonuç dizinleri
DATASET_ROOT = Path(__file__).parent.parent / 'LLVIP'
RESULTS_DIR = Path(__file__).parent.parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)


def find_image_pairs(dataset_root):
    """
    LLVIP Dataset'ten görüntü çiftlerini bulur
    
    LLVIP dataset yapısı:
    - infrared/train/, infrared/test/
    - visible/train/, visible/test/
    
    Bu fonksiyon test klasörlerinden çiftleri bulur.
    
    Returns:
    -------
    list : [(infrared_path, visible_path, name), ...]
    """
    pairs = []
    
    # Test klasörlerinden görüntüleri bul
    ir_test_dir = dataset_root / 'infrared' / 'test'
    vis_test_dir = dataset_root / 'visible' / 'test'
    
    if not ir_test_dir.exists() or not vis_test_dir.exists():
        print(f"❌ LLVIP dataset klasörleri bulunamadı!")
        print(f"   Infrared: {ir_test_dir}")
        print(f"   Visible: {vis_test_dir}")
        return pairs
    
    # IR görüntüleri listele
    ir_files = sorted(os.listdir(ir_test_dir))
    
    for ir_file in ir_files:
        # Desteklenen formatlara bak
        if not ir_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif')):
            continue
        
        # VIS görüntüsü aynı isimde olmalıdır
        vis_file = ir_file
        vis_path = vis_test_dir / vis_file
        ir_path = ir_test_dir / ir_file
        
        if vis_path.exists():
            # Scene adı dosya adından çıkarılır
            scene_name = ir_file.split('.')[0]
            pairs.append((str(ir_path), str(vis_path), scene_name))
    
    print(f"Toplam {len(pairs)} görüntü çifti bulundu")
    return pairs


def test_single_method(method_name, method_func, img1, img2, scene_name):
    """
    Tek bir füzyon yöntemini test eder
    
    Parametreler:
    ------------
    method_name : str
        Yöntem adı
    method_func : callable
        Füzyon fonksiyonu
    img1, img2 : numpy.ndarray
        Kaynak görüntüler
    scene_name : str
        Sahne adı (kaydetmek için)
        
    Returns:
    -------
    dict : Test sonuçları
    """
    print(f"\n{'='*60}")
    print(f"Test: {method_name} - {scene_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Füzyon yap
        fused = method_func(img1, img2)
        
        # Süre
        elapsed_time = time.time() - start_time
        
        # Metrikleri hesapla
        metrics = calculate_all_metrics(img1, fused, img2, verbose=True)
        
        # Sonucu kaydet
        output_dir = RESULTS_DIR / scene_name
        output_dir.mkdir(exist_ok=True)
        
        save_path = output_dir / f'{method_name}_fused.png'
        save_image(fused, str(save_path))
        
        print(f"\n✅ {method_name} tamamlandı! Süre: {elapsed_time:.2f}s")
        print(f"   Kaydedildi: {save_path}")
        
        return {
            'method': method_name,
            'scene': scene_name,
            'time': elapsed_time,
            'psnr': metrics['psnr_avg'],
            'ssim': metrics['ssim_avg'],
            'mse': metrics['mse_avg'],
            'mi': metrics['mi_avg'],
            'entropy': metrics['entropy'],
            'sf': metrics['sf'],
            'status': 'success'
        }
        
    except Exception as e:
        print(f"\n❌ {method_name} HATA: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'method': method_name,
            'scene': scene_name,
            'time': 0,
            'psnr': 0,
            'ssim': 0,
            'mse': 0,
            'mi': 0,
            'entropy': 0,
            'sf': 0,
            'status': 'failed',
            'error': str(e)
        }


def test_all_methods():
    """
    Tüm füzyon yöntemlerini dataset üzerinde test eder
    """
    print("\n" + "="*80)
    print("DEEPFUSIONCOLOR - KAPSAMLI TEST")
    print("="*80)
    
    # Görüntü çiftlerini bul
    image_pairs = find_image_pairs(DATASET_ROOT)
    
    if not image_pairs:
        print("❌ Hiç görüntü çifti bulunamadı!")
        print(f"Dataset yolu kontrol edin: {DATASET_ROOT}")
        return
    
    # Test edilecek yöntemler
    # Her yöntem için farklı parametreler test edebilirsin
    methods = [
        ('Wavelet', lambda img1, img2: wavelet_fusion(img1, img2, wavelet='db4', level=3)),
        ('DNN', lambda img1, img2: dnn_fusion(img1, img2, epochs=10, batch_size=1024)),
        ('CNN', lambda img1, img2: cnn_fusion(img1, img2, epochs=15, batch_size=16)),
        ('LatentLRR', lambda img1, img2: latentlrr_fusion(img1, img2, rank_ratio=0.9)),
        ('DenseFuse', lambda img1, img2: densefuse_fusion(img1, img2, epochs=20, batch_size=16))
    ]
    
    # Tüm sonuçları kaydet
    all_results = []
    
    # Her görüntü çifti için
    for thermal_path, visible_path, scene_name in image_pairs:
        print(f"\n\n{'#'*80}")
        print(f"SAHNE: {scene_name}")
        print(f"{'#'*80}")
        print(f"Thermal: {thermal_path}")
        print(f"Visible: {visible_path}")
        
        try:
            # Görüntüleri yükle
            img1 = load_image(thermal_path, color_mode='grayscale')
            img2 = load_image(visible_path, color_mode='grayscale')
            
            # Ön işleme
            img1, img2 = preprocess_for_fusion(img1, img2, target_size=(256, 256))
            
            # Orijinal görüntüleri kaydet
            output_dir = RESULTS_DIR / scene_name
            output_dir.mkdir(exist_ok=True)
            save_image(img1, str(output_dir / 'thermal.png'))
            save_image(img2, str(output_dir / 'visible.png'))
            
            # Her yöntemi test et
            for method_name, method_func in methods:
                result = test_single_method(method_name, method_func, img1, img2, scene_name)
                all_results.append(result)
                
        except Exception as e:
            print(f"❌ Sahne yüklenemedi: {e}")
            continue
    
    # Sonuçları analiz et ve kaydet
    analyze_results(all_results)


def analyze_results(results):
    """
    Test sonuçlarını analiz eder ve raporlar
    
    Parametreler:
    ------------
    results : list
        Test sonuçları listesi
    """
    print("\n\n" + "="*80)
    print("SONUÇLARIN ANALİZİ")
    print("="*80)
    
    # DataFrame'e çevir
    df = pd.DataFrame(results)
    
    # CSV olarak kaydet
    csv_path = RESULTS_DIR / 'test_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Sonuçlar kaydedildi: {csv_path}")
    
    # Başarılı testleri filtrele
    df_success = df[df['status'] == 'success']
    
    if len(df_success) == 0:
        print("\n❌ Hiç başarılı test yok!")
        return
    
    # Yöntemlere göre ortalama metrikler
    print("\n" + "-"*80)
    print("YÖNTEMLERE GÖRE ORTALAMA METRİKLER")
    print("-"*80)
    
    metrics_by_method = df_success.groupby('method').agg({
        'time': 'mean',
        'psnr': 'mean',
        'ssim': 'mean',
        'mse': 'mean',
        'mi': 'mean',
        'entropy': 'mean',
        'sf': 'mean'
    }).round(4)
    
    print(metrics_by_method)
    
    # En iyi yöntemler
    print("\n" + "-"*80)
    print("EN İYİ YÖNTEMLER")
    print("-"*80)
    
    best_psnr = df_success.loc[df_success['psnr'].idxmax()]
    best_ssim = df_success.loc[df_success['ssim'].idxmax()]
    best_mi = df_success.loc[df_success['mi'].idxmax()]
    fastest = df_success.loc[df_success['time'].idxmin()]
    
    print(f"🏆 En Yüksek PSNR: {best_psnr['method']} ({best_psnr['psnr']:.2f} dB)")
    print(f"🏆 En Yüksek SSIM: {best_ssim['method']} ({best_ssim['ssim']:.4f})")
    print(f"🏆 En Yüksek MI: {best_mi['method']} ({best_mi['mi']:.4f})")
    print(f"⚡ En Hızlı: {fastest['method']} ({fastest['time']:.2f}s)")
    
    # Visualization
    create_comparison_plots(metrics_by_method)


def create_comparison_plots(metrics_df):
    """
    Karşılaştırma grafikleri oluşturur
    
    Parametreler:
    ------------
    metrics_df : pandas.DataFrame
        Metrik sonuçları
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Füzyon Yöntemleri Karşılaştırması', fontsize=16)
    
    metrics = ['psnr', 'ssim', 'mi', 'entropy', 'sf', 'time']
    titles = ['PSNR (dB)', 'SSIM', 'Mutual Information', 
              'Entropy (bits)', 'Spatial Frequency', 'İşlem Süresi (s)']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 3, idx % 3]
        
        data = metrics_df[metric].sort_values(ascending=False)
        colors = plt.cm.viridis(np.linspace(0, 1, len(data)))
        
        data.plot(kind='bar', ax=ax, color=colors)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Yöntem')
        ax.set_ylabel(title)
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Kaydet
    plot_path = RESULTS_DIR / 'comparison_plots.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Karşılaştırma grafikleri kaydedildi: {plot_path}")
    
    plt.close()


if __name__ == '__main__':
    print("DeepFusionColor Test Suite başlatılıyor...\n")
    
    # Tüm testleri çalıştır
    test_all_methods()
    
    print("\n" + "="*80)
    print("TEST TAMAMLANDI!")
    print("="*80)
    print(f"Sonuçlar: {RESULTS_DIR}")
    print("="*80 + "\n")
