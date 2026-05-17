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
        print(f"[MISTRAL] API Response status: {response.status_code}")
        print(f"[MISTRAL] Response data: {data}")

        content = data["choices"][0]["message"]["content"]
        print(f"[MISTRAL] Content extracted: {content}")
        
        # Markdown code block markers'ı kaldır
        content = content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        parsed = json.loads(content)
        print(f"[MISTRAL] JSON parsed: {parsed}")

        return {
            "success": True,
            "analysis": parsed
        }

    except Exception as e:
        print(f"[MISTRAL ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }