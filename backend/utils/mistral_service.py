import os
import json
import requests

from dotenv import load_dotenv

from utils.prompt_builder import build_fusion_prompt
from utils.metric_interpreter import normalize_metrics

load_dotenv()

API_KEY = os.getenv("MISTRAL_API_KEY")

URL = "https://api.mistral.ai/v1/chat/completions"


def analyze_fusion_metrics(metrics: dict):

    try:

        normalized_metrics = normalize_metrics(metrics)

        prompt = build_fusion_prompt(normalized_metrics)

        response = requests.post(
            URL,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "mistral-small-latest",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a professional computer vision "
                            "and image fusion analysis system."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.2,
                "max_tokens": 700
            },
            timeout=40
        )

        data = response.json()

        content = data["choices"][0]["message"]["content"]

        parsed = json.loads(content)

        return {
            "success": True,
            "analysis": parsed
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }