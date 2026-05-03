from utils.mistral_service import analyze_fusion_metrics
import json


def generate_ai_analysis(method, psnr, ssim, mse, mi, en, sf):

    metrics = {
        "method": method,
        "psnr": psnr,
        "ssim": ssim,
        "mse": mse,
        "mi": mi,
        "entropy": en,
        "spatial_frequency": sf
    }

    result = analyze_fusion_metrics(metrics)
    
    print(f"[ANALYSIS] Mistral service result: {result}")
    
    if result.get("success"):
        analysis_data = result.get("analysis")
        print(f"[ANALYSIS] Returning analysis: {analysis_data}")
        return analysis_data
    else:
        error_msg = f"AI analizi hatası: {result.get('error', 'Bilinmeyen hata')}"
        print(f"[ANALYSIS] Returning error: {error_msg}")
        return error_msg