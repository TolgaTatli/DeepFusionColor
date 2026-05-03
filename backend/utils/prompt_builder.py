def build_fusion_prompt(metrics: dict):

    return f"""
You are a senior computer vision and image fusion expert.

Analyze the following image fusion metrics.

Fusion Method:
{metrics["method"]}

Metrics:
- PSNR: {metrics["psnr"]}
- SSIM: {metrics["ssim"]}
- MSE: {metrics["mse"]}
- Mutual Information: {metrics["mi"]}
- Entropy: {metrics["entropy"]}
- Spatial Frequency: {metrics["spatial_frequency"]}

Evaluation Requirements:

1. Evaluate reconstruction quality.
2. Evaluate structural preservation.
3. Evaluate image sharpness.
4. Evaluate information richness.
5. Evaluate possible noise/artifacts.
6. Evaluate fusion stability.
7. Explain metric tradeoffs.
8. Give overall fusion success score.

Important Metric Knowledge:
- Higher PSNR is better.
- Higher SSIM is better.
- Lower MSE is better.
- Higher MI is better.
- Higher Entropy means richer information.
- Higher Spatial Frequency means sharper image.

Return STRICT JSON.

JSON format:

{{
    "fusion_quality": "",
    "structural_preservation": "",
    "sharpness": "",
    "information_richness": "",
    "noise_assessment": "",
    "metric_tradeoffs": "",
    "overall_score": 0,
    "technical_summary": "",
    "recommendation": ""
}}

Do not return markdown.
Do not explain outside JSON.
"""