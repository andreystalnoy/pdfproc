"""
model_detect.py — автоопределение лучшей доступной vision-модели.

Шаги:
1. Определить VRAM GPU через nvidia-smi
2. Получить список установленных моделей Ollama
3. Выбрать лучшую модель по качеству (из MODEL_REGISTRY)
4. Если локальных нет — проверить облачные API-ключи
5. Если ничего нет — вывести инструкции
"""

import os
import subprocess
from llm_backend import MODEL_REGISTRY


def _detect_vram_gb() -> int:
    """Определить VRAM GPU через nvidia-smi. Возвращает 0 если GPU не найден."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            first_line = result.stdout.strip().splitlines()[0].strip()
            mb = int(first_line)
            return mb // 1024
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, IndexError):
        pass
    return 0


def _get_ollama_models() -> list:
    """Получить список установленных моделей Ollama. Возвращает [] при ошибке."""
    try:
        import ollama
        client = ollama.Client()
        models_response = client.list()
        # Поддержка разных версий API ollama
        if hasattr(models_response, "models"):
            models_list = models_response.models
        elif isinstance(models_response, dict):
            models_list = models_response.get("models", [])
        else:
            models_list = list(models_response)
        names = []
        for m in models_list:
            if hasattr(m, "model"):
                names.append(m.model)
            elif hasattr(m, "name"):
                names.append(m.name)
            elif isinstance(m, dict):
                names.append(m.get("model") or m.get("name", ""))
        return [n for n in names if n]
    except Exception:
        return []


def auto_select_model(require_vision: bool = True, verbose: bool = True) -> str:
    """
    Detect GPU VRAM, check installed Ollama models, pick the best one.
    If nothing suitable found locally, print suggestions and return None.
    """
    vram_gb = _detect_vram_gb()
    ollama_models = _get_ollama_models()

    if verbose:
        gpu_str = f"{vram_gb} GB" if vram_gb > 0 else "не найден"
        print(f"[auto] GPU VRAM: {gpu_str}")
        if ollama_models:
            print(f"[auto] Ollama модели: {', '.join(ollama_models)}")
        else:
            print("[auto] Ollama: нет установленных моделей / не запущен")

    # Нормализовать имена ollama-моделей (убрать :latest суффиксы)
    ollama_set = set()
    for name in ollama_models:
        ollama_set.add(name)
        # Также добавим без тэга если есть тэг
        if ":" in name:
            base = name.split(":")[0]
            ollama_set.add(base)

    # Собрать кандидатов из реестра
    best_model = None
    best_quality = -1

    for model_id, meta in MODEL_REGISTRY.items():
        if require_vision and not meta.get("vision"):
            continue
        provider = meta["provider"]
        vram_needed = meta.get("vram_gb", 0)
        quality = meta.get("quality", 0)

        if provider == "ollama":
            # Нужно чтобы модель была установлена и влезала в VRAM
            if vram_gb < vram_needed:
                continue
            # Проверяем что модель есть локально
            model_base = model_id.split(":")[0]
            installed = model_id in ollama_set or model_base in ollama_set
            if not installed:
                continue
        # Для облачных провайдеров — не нужно проверять VRAM, они доступны если есть ключ
        # (Ключи проверим ниже в fallback, здесь пропускаем облако)
        elif provider in ("anthropic", "openai"):
            continue

        if quality > best_quality:
            best_quality = quality
            best_model = model_id

    if best_model:
        if verbose:
            print(f"[auto] Выбрана модель: {best_model}")
        return best_model

    # Локальных нет — проверить облачные ключи
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    openai_key = os.environ.get("OPENAI_API_KEY", "")

    if anthropic_key:
        chosen = "anthropic:claude-haiku-4-5"
        if verbose:
            print(f"[auto] ANTHROPIC_API_KEY найден → {chosen}")
        return chosen

    if openai_key:
        chosen = "openai:gpt-4o-mini"
        if verbose:
            print(f"[auto] OPENAI_API_KEY найден → {chosen}")
        return chosen

    # Ничего нет — вывести инструкции
    gpu_str = f"{vram_gb}GB" if vram_gb > 0 else "GPU не найден"
    ollama_str = ", ".join(ollama_models) if ollama_models else "нет"

    print()
    print("╔══ pdfproc: нет подходящей vision-модели ══╗")
    print()
    print(f"  GPU VRAM обнаружено: {gpu_str}")
    print(f"  Установленные модели Ollama: {ollama_str}")
    print()
    print("  Варианты:")
    print("    [локально]  ollama pull qwen2.5vl:7b     — нужно ~8 GB VRAM")
    print("    [локально]  ollama pull qwen3-vl:8b      — нужно ~8 GB VRAM, лучше качество")
    print("    [облако]    set ANTHROPIC_API_KEY=...    — pip install anthropic")
    print("                --model anthropic:claude-haiku-4-5")
    print("    [облако]    set OPENAI_API_KEY=...       — pip install openai")
    print("                --model openai:gpt-4o-mini")
    print()
    print("╚══════════════════════════════════════════╝")
    print()

    return None
