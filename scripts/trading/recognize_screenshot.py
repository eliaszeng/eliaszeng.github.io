"""
命令行工具：接收交易截图路径，调用 Vision API 识别，
输出结构化 JSON 到 stdout。

用法:
    python scripts/trading/recognize_screenshot.py <image_path> [--output <json_file>]

环境变量（任选一组）:
    方式一: VISION_API_KEY + VISION_BASE_URL + VISION_MODEL
    方式二: ANTHROPIC_AUTH_TOKEN + ANTHROPIC_BASE_URL + ANTHROPIC_MODEL
"""

import argparse
import base64
import json
import os
import sys
from pathlib import Path

from openai import OpenAI


_PROMPT = """分析这张交易截图，提取以下信息并严格以JSON格式返回（不要包含其他文字）：
{
  "stockCode": "6位股票代码",
  "stockName": "股票名称",
  "direction": "buy或sell",
  "price": 交易价格(数字),
  "quantity": 交易数量(数字),
  "amount": 交易金额(数字),
  "tradeTime": "交易时间 ISO格式 如 2026-05-12T10:30:00",
  "reason": "交易理由/分析",
  "tags": ["标签1", "标签2"]
}
无法确定的字段返回null。如果图片不是交易相关内容，返回 {"error": "not_trade_related"}。"""


def encode_image(image_path: Path) -> str:
    data = image_path.read_bytes()
    suffix = image_path.suffix.lower()
    media_type = {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".gif": "image/gif", ".webp": "image/webp",
    }.get(suffix, "image/png")
    return f"data:{media_type};base64,{base64.b64encode(data).decode()}"


def recognize(image_path: str) -> dict:
    """调用 Vision API 识别交易截图。"""
    path = Path(image_path)
    if not path.exists():
        print(f"错误: 文件不存在 — {image_path}", file=sys.stderr)
        sys.exit(1)

    api_key = os.environ.get("VISION_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
    base_url = os.environ.get("VISION_BASE_URL") or os.environ.get("ANTHROPIC_BASE_URL", "")
    model = os.environ.get("VISION_MODEL") or os.environ.get("ANTHROPIC_MODEL", "mimo-v2.5-pro")

    if not api_key or not base_url:
        print("错误: 请设置 VISION_API_KEY + VISION_BASE_URL 环境变量", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url=base_url.rstrip("/") + "/v1")
    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": _PROMPT},
                {"type": "image_url", "image_url": {"url": encode_image(path)}},
            ],
        }],
        max_tokens=1024,
    )

    text = (response.choices[0].message.content or "").strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        print(f"AI 返回内容无法解析为 JSON:\n{text[:500]}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="交易截图 AI 识别")
    parser.add_argument("image", help="截图文件路径")
    parser.add_argument("--output", "-o", help="输出 JSON 文件路径（默认 stdout）")
    args = parser.parse_args()

    result = recognize(args.image)
    output = json.dumps(result, ensure_ascii=False, indent=2)

    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"已写入 {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
