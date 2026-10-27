import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def ask_teacher(prediction, confidence, low_confidence, anomaly_flag):
    api_key = os.getenv("NEBIUS_API_KEY")

    if not api_key:
        return "Teacher unavailable: NEBIUS_API_KEY is missing."

    client = OpenAI(
        base_url="https://api.tokenfactory.nebius.com/v1/",
        api_key=api_key,
        timeout=30.0,
        max_retries=0
    )

    prompt = f"""
You are a diagnostic assistant for an industrial bearing
monitoring prototype.

The edge classifier produced:
- Predicted condition: {prediction}
- Confidence: {confidence:.2%}
- Low-confidence flag: {low_confidence}
- Distribution-anomaly flag: {anomaly_flag}

The prediction has been escalated for review.

Explain briefly:
1. Why the prediction should not be trusted automatically.
2. What an industrial maintenance operator should inspect.
3. What additional evidence would help confirm the bearing condition.

Do not claim to know the true bearing condition.
Do not recommend automatic machine shutdown or repair.
Keep the response concise.
"""

    try:
        response = client.chat.completions.create(
            model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=500
        )

        content = response.choices[0].message.content

        return content or "Teacher returned no explanation."

    except Exception as error:
        return f"Teacher unavailable: {type(error).__name__}: {error}"