from utils.mistral_service import analyze_fusion_metrics


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

    return analyze_fusion_metrics(metrics)