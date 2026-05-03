def normalize_metrics(metrics):

    return {
        "method": metrics.get("method", "unknown"),
        "psnr": round(float(metrics.get("psnr", 0)), 4),
        "ssim": round(float(metrics.get("ssim", 0)), 4),
        "mse": round(float(metrics.get("mse", 0)), 6),
        "mi": round(float(metrics.get("mi", 0)), 4),
        "entropy": round(float(metrics.get("entropy", 0)), 4),
        "spatial_frequency": round(float(metrics.get("spatial_frequency", 0)), 4),
    }