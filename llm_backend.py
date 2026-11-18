"""
LLM backend abstraction for pdfproc.
Supports: ollama (local), anthropic (Claude API), openai (GPT-4o API).

Backend is selected by the MODEL string prefix:
  anthropic:claude-haiku-4-5  → use anthropic package
  openai:gpt-4o-mini          → use openai package
  qwen2.5vl:7b (or any other) → use ollama package
"""

import io
import base64

MODEL_REGISTRY = {
    "qwen2.5vl:7b":   {"provider": "ollama", "vision": True,  "vram_gb": 8,  "quality": 2},
    "qwen3-vl:8b":    {"provider": "ollama", "vision": True,  "vram_gb": 8,  "quality": 3},
    "qwen3.5:9b":     {"provider": "ollama", "vision": True,  "vram_gb": 8,  "quality": 3},
    "gemma3:12b":     {"provider": "ollama", "vision": True,  "vram_gb": 10, "quality": 3},
    "anthropic:claude-haiku-4-5":  {"provider": "anthropic", "vision": True,  "vram_gb": 0, "quality": 4},
    "anthropic:claude-sonnet-4-5": {"provider": "anthropic", "vision": True,  "vram_gb": 0, "quality": 5},
    "openai:gpt-4o-mini":          {"provider": "openai",    "vision": True,  "vram_gb": 0, "quality": 4},
    "openai:gpt-4o":               {"provider": "openai",    "vision": True,  "vram_gb": 0, "quality": 5},
}


def _parse_model(model: str):
    """Return (provider, model_id) from a model string."""
    if model.startswith("anthropic:"):
        return "anthropic", model[len("anthropic:"):]
    elif model.startswith("openai:"):
        return "openai", model[len("openai:"):]
    else:
        return "ollama", model


def _image_to_b64(image) -> str:
    """Encode a PIL Image to base64 PNG string."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def vision_query(image, prompt: str, model: str, system: str = "") -> str:
    """Send image + prompt to the configured backend. Returns text."""
    provider, model_id = _parse_model(model)

    if provider == "anthropic":
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package not installed. Run: pip install anthropic"
            )
        b64 = _image_to_b64(image)
        client = anthropic.Anthropic()
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        kwargs = {"model": model_id, "max_tokens": 4096, "messages": messages}
        if system:
            kwargs["system"] = system
        response = client.messages.create(**kwargs)
        return response.content[0].text

    elif provider == "openai":
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai package not installed. Run: pip install openai"
            )
        b64 = _image_to_b64(image)
        client = openai.OpenAI()
        content = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            },
            {"type": "text", "text": prompt},
        ]
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})
        response = client.chat.completions.create(
            model=model_id,
            max_tokens=4096,
            messages=messages,
        )
        return response.choices[0].message.content

    else:  # ollama
        try:
            import ollama
        except ImportError:
            raise ImportError(
                "ollama package not installed. Run: pip install ollama"
            )
        b64 = _image_to_b64(image)
        response = ollama.chat(
            model=model_id,
            messages=[{"role": "user", "content": prompt, "images": [b64]}],
            options={"num_ctx": 8192},
        )
        return response["message"]["content"]


def text_query(prompt: str, model: str, system: str = "") -> str:
    """Send text-only prompt to the configured backend. Returns text."""
    provider, model_id = _parse_model(model)

    if provider == "anthropic":
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package not installed. Run: pip install anthropic"
            )
        client = anthropic.Anthropic()
        messages = [{"role": "user", "content": prompt}]
        kwargs = {"model": model_id, "max_tokens": 4096, "messages": messages}
        if system:
            kwargs["system"] = system
        response = client.messages.create(**kwargs)
        return response.content[0].text

    elif provider == "openai":
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai package not installed. Run: pip install openai"
            )
        client = openai.OpenAI()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(
            model=model_id,
            max_tokens=4096,
            messages=messages,
        )
        return response.choices[0].message.content

    else:  # ollama
        try:
            import ollama
        except ImportError:
            raise ImportError(
                "ollama package not installed. Run: pip install ollama"
            )
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = ollama.chat(model=model_id, messages=messages)
        return response["message"]["content"]
