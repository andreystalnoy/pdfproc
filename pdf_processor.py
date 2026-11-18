"""
pdf_processor.py — дешёвый pipeline для больших PDF-документов

Стратегия (по возрастанию стоимости):
  1. pdfplumber  → нативный текст (бесплатно, мгновенно)
  2. pytesseract → OCR для сканов (CPU, бесплатно)
  3. ollama      → локальный LLM для структуризации (бесплатно, оффлайн)

Задачи:
  manifest       — классифицировать все страницы (тип + язык + резюме) → manifest.json
  vision_extract — извлечь данные через vision LLM → файл или папка с секциями
  summarize      — краткое резюме через LLM
  qa             — ответить на вопрос по документу
  info           — быстрая информация о PDF

Типы секций:
  title           — титульный лист
  contract_header — реквизиты договора / заказчика
  pricing         — ценовая ведомость, стоимость
  tech_spec       — техническая спецификация, характеристики
  requirements    — требования к поставщику / товару
  terms           — условия контракта, сроки
  appendix        — приложение
  other           — остальное

Примеры:
  python pdf_processor.py doc.pdf --task manifest --model qwen2.5vl:7b
  python pdf_processor.py doc.pdf --task vision_extract --manifest manifest.json --lang ru --skip-types tech_spec --format md --split-by-section --out result/
  python pdf_processor.py doc.pdf --task vision_extract --pages 1-10 --format md
"""

import argparse
import json
import sys
from pathlib import Path

from llm_backend import vision_query, text_query, MODEL_REGISTRY  # noqa: F401
import model_detect

try:
    import pypdfium2 as pdfium
except ImportError:
    print("ERROR: pypdfium2 not installed. Run: pip install pypdfium2", file=sys.stderr)
    sys.exit(1)


# ── утилиты ────────────────────────────────────────────────────────────────

def parse_page_range(spec: str, total: int) -> list[int]:
    """'1-5,8,10-12' → [0,1,2,3,4,7,9,10,11] (0-based)"""
    pages = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            pages.extend(range(int(a) - 1, min(int(b), total)))
        else:
            pages.append(int(part) - 1)
    return sorted(set(pages))


def slugify(text: str) -> str:
    """Простой slugify для имён файлов."""
    import re
    text = text.lower().replace(" ", "_")
    return re.sub(r"[^\w]", "", text)


# ── извлечение текста ──────────────────────────────────────────────────────

def extract_text_pdfplumber(pdf_path: str, page_indices: list[int]) -> dict[int, str]:
    import pdfplumber
    results = {}
    with pdfplumber.open(pdf_path) as pdf:
        for i in page_indices:
            if i >= len(pdf.pages):
                continue
            results[i] = pdf.pages[i].extract_text() or ""
    return results


def extract_tables_pdfplumber(pdf_path: str, page_indices: list[int]) -> dict[int, list]:
    import pdfplumber
    results = {}
    with pdfplumber.open(pdf_path) as pdf:
        for i in page_indices:
            if i >= len(pdf.pages):
                continue
            tables = pdf.pages[i].extract_tables()
            if tables:
                results[i] = tables
    return results


def render_page(pdf_path: str, page_index: int, dpi: int = 150) -> "object":
    """Render one PDF page to PIL Image. page_index is 0-based."""
    doc = pdfium.PdfDocument(pdf_path)
    page = doc[page_index]
    scale = dpi / 72
    bitmap = page.render(scale=scale, rotation=0)
    pil_image = bitmap.to_pil()
    doc.close()
    return pil_image


def ocr_pages(pdf_path: str, page_indices: list[int], dpi: int = 200) -> dict[int, str]:
    try:
        import pytesseract
    except ImportError:
        print("Установи: pip install pytesseract")
        sys.exit(1)
    results = {}
    for page_i in page_indices:
        img = render_page(pdf_path, page_i, dpi=dpi)
        text = pytesseract.image_to_string(img, lang="rus+eng")
        results[page_i] = text
        print(f"  OCR страница {page_i + 1}: {len(text)} симв.")
    return results


def is_garbage_text(text: str) -> bool:
    if len(text) < 20:
        return True
    garbage_chars = sum(1 for c in text if c in "{}|\\[]<>_^~`")
    if garbage_chars / len(text) > 0.03:
        return True
    words = text.split()
    if words and sum(len(w) for w in words) / len(words) > 12:
        return True
    return False


def get_page_texts(pdf_path: str, page_indices: list[int], force_ocr: bool = False) -> dict[int, str]:
    if force_ocr:
        return ocr_pages(pdf_path, page_indices)
    texts = extract_text_pdfplumber(pdf_path, page_indices)
    need_ocr = [i for i in page_indices if is_garbage_text(texts.get(i, ""))]
    if need_ocr:
        print(f"Мусорный/пустой текст: {len(need_ocr)} стр. → OCR...")
        texts.update(ocr_pages(pdf_path, need_ocr))
    return texts


# ── vision / ollama ────────────────────────────────────────────────────────

def pages_to_images(pdf_path: str, page_indices: list[int], dpi: int = 200) -> dict[int, object]:
    result = {}
    for page_i in page_indices:
        result[page_i] = render_page(pdf_path, page_i, dpi=dpi)
    return result


def ollama_vision(image, prompt: str, model: str) -> str:
    """Delegate to llm_backend.vision_query (supports ollama, anthropic, openai)."""
    return vision_query(image, prompt, model=model)


import base64  # нужен выше


def ollama_query(prompt: str, model: str, system: str = "") -> str:
    """Delegate to llm_backend.text_query (supports ollama, anthropic, openai)."""
    return text_query(prompt, model=model, system=system)


def chunk_text(text: str, max_chars: int = 3000) -> list[str]:
    lines = text.splitlines()
    chunks, current, size = [], [], 0
    for line in lines:
        if size + len(line) > max_chars and current:
            chunks.append("\n".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


def parse_json_from_response(text: str):
    """Вытащить JSON из ответа модели — поддерживает ```json блоки и голый JSON."""
    import re
    # Сначала пробуем ```json ... ``` блок
    code_block = re.search(r"```(?:json)?\s*([\[{].*?)\s*```", text, re.DOTALL)
    if code_block:
        try:
            return json.loads(code_block.group(1))
        except json.JSONDecodeError:
            pass
    # Затем ищем голый JSON
    for start_char, end_char in [("[", "]"), ("{", "}")]:
        s = text.find(start_char)
        e = text.rfind(end_char) + 1
        if s >= 0 and e > s:
            try:
                return json.loads(text[s:e])
            except json.JSONDecodeError:
                pass
    return None


def sanitize_rows(rows: list[dict], max_cell: int = 300) -> list[dict]:
    """Обрезать чрезмерно длинные значения ячеек (зацикливание внутри токена)."""
    result = []
    for row in rows:
        result.append({
            k: (str(v)[:max_cell] + "…") if isinstance(v, str) and len(str(v)) > max_cell else v
            for k, v in row.items()
        })
    return result


def is_hallucination_loop(rows: list) -> bool:
    """Определить зацикливание или пустой шаблон."""
    if not rows or not isinstance(rows[0], dict):
        return False
    # Пустой шаблон: ключ == значение во всех ячейках первой строки
    if all(str(k) == str(v) for k, v in rows[0].items()):
        return True
    if len(rows) < 5:
        return False
    from collections import Counter
    serialized = [json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows]
    counter = Counter(serialized)
    # Идентичные строки: 70%+ совпадают с первой
    if counter[serialized[0]] / len(rows) > 0.7:
        return True
    # Чередующийся паттерн: мало уникальных строк (<=5) покрывают >85% всех строк
    top_coverage = sum(v for _, v in counter.most_common(5))
    if top_coverage / len(rows) > 0.85 and len(counter) <= 5:
        return True
    return False


def is_text_loop(text: str) -> bool:
    """Определить зацикливание в текстовом ответе (повторяющиеся строки)."""
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if len(lines) < 10:
        return False
    from collections import Counter
    counter = Counter(lines)
    top5 = sum(v for _, v in counter.most_common(5))
    # Если топ-5 строк покрывают >70% всего текста — зацикливание
    return top5 / len(lines) > 0.70


# ── задача: manifest ───────────────────────────────────────────────────────

MANIFEST_PROMPT = """Это страница документа.

Определи:
1. type — тип страницы, ОДНО из: title, contract_header, pricing, tech_spec, requirements, terms, appendix, other
2. lang — язык текста: "ru" (русский), "kz" (казахский), "both" (оба языка), "other" (иной)
3. summary — одно предложение: что именно на этой странице

Ответь ТОЛЬКО JSON-объектом, без пояснений:
{"type": "...", "lang": "...", "summary": "..."}"""


def task_manifest(pdf_path: str, page_indices: list[int], out_path: str, model: str, force: bool = False):
    """
    Быстрый проход: классифицировать каждую страницу (тип + язык + резюме).
    Результат — manifest.json, который используется в vision_extract для фильтрации.
    """
    out = out_path or str(Path(pdf_path).with_suffix(".manifest.json"))
    cache_dir = Path(out).parent / ".pdfproc_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_cache = cache_dir / "manifest.json"

    # Загрузить из кэша если есть и не форс
    if not force and manifest_cache.exists():
        cached = json.loads(manifest_cache.read_text(encoding="utf-8"))
        if cached.get("model") == model:
            manifest = cached["manifest"]
            out_path = Path(out)
            if out_path.is_dir():
                out_path = out_path / "manifest.json"
            out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[cache] Манифест загружен из кэша ({len(manifest)} стр.) → {out_path}")
            # Статистика
            from collections import Counter
            types = Counter(e["type"] for e in manifest)
            langs = Counter(e["lang"] for e in manifest)
            print("\nТипы страниц:")
            for t, n in types.most_common():
                print(f"  {t}: {n} стр.")
            print("Языки:")
            for l, n in langs.most_common():
                print(f"  {l}: {n} стр.")
            return

    manifest = []

    for page_i in sorted(page_indices):
        img = render_page(pdf_path, page_i, dpi=150)
        print(f"  Классификация стр.{page_i + 1}...", end=" ", flush=True)
        try:
            result = ollama_vision(img, MANIFEST_PROMPT, model=model)
            data = parse_json_from_response(result)
            if isinstance(data, dict):
                entry = {
                    "page": page_i + 1,
                    "type": data.get("type", "other"),
                    "lang": data.get("lang", "both"),
                    "summary": data.get("summary", ""),
                }
            else:
                entry = {"page": page_i + 1, "type": "other", "lang": "both", "summary": result.strip()[:120]}
            manifest.append(entry)
            print(f"{entry['type']} [{entry['lang']}] — {entry['summary'][:60]}")
        except Exception as e:
            manifest.append({"page": page_i + 1, "type": "other", "lang": "both", "summary": f"ошибка: {e}"})
            print(f"ошибка: {e}")
        # Пишем кэш после каждой страницы — чтобы не потерять при сбое
        manifest_cache.write_text(json.dumps({"model": model, "manifest": manifest}, ensure_ascii=False, indent=2), encoding="utf-8")

    out_path = Path(out)
    if out_path.is_dir():
        out_path = out_path / "manifest.json"
    out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nМанифест ({len(manifest)} стр.) → {out_path}")

    # Краткая статистика
    from collections import Counter
    types = Counter(e["type"] for e in manifest)
    langs = Counter(e["lang"] for e in manifest)
    print("\nТипы страниц:")
    for t, n in types.most_common():
        print(f"  {t}: {n} стр.")
    print("Языки:")
    for l, n in langs.most_common():
        print(f"  {l}: {n} стр.")


# ── задача: vision_extract ─────────────────────────────────────────────────

# qwen2.5vl:7b нестабильно генерирует JSON из любых страниц, включая ценовые.
# Все страницы обрабатываются в текстовом режиме для надёжности.
# Для pricing используется специальный промпт с перечислением лотов.
TABLE_TYPES: set = set()
TEXT_TYPES = {"terms", "requirements", "contract_header", "appendix", "title", "pricing", "other"}

EXTRACT_TABLE_PROMPT_RU = """Это страница документа с таблицей данных.
Извлеки строки таблицы как JSON-массив объектов. Каждый объект — одна строка.
Ключи объекта — названия колонок, значения — содержимое ячеек.
Верни ТОЛЬКО JSON-массив, без пояснений. Текст на РУССКОМ."""

EXTRACT_TEXT_PROMPT_RU = """Это страница документа на русском языке.
Перепиши ВЕСЬ текст страницы полностью в Markdown, ничего не пропуская:
- ## для заголовков разделов
- Нумерованные и маркированные списки для перечислений
- **жирный** для ключевых терминов, сумм, дат, сроков
- Таблица Markdown если на странице есть таблица с данными
Выводи текст напрямую — БЕЗ обёртки в ```markdown``` блоки. Только РУССКИЙ текст, без JSON."""

EXTRACT_TABLE_PROMPT_KZ = """Бұл кесте деректері бар құжат беті.
Кесте жолдарын JSON массиві ретінде шығарып ал. Әр объект — бір жол.
ТЕК JSON массивін қайтар, түсіндірмесіз. Мәтін ҚАЗАҚ тілінде."""

EXTRACT_TEXT_PROMPT_KZ = """Бұл құжаттың беті.
Беттің БАРЛЫҚ мәтінін Markdown форматында толық жаз, ештеңені өткізіп жіберме:
- ## тақырыптар үшін
- Тізімдер мен **маңызды** терминдерді, сомаларды, мерзімдерді ерекшеле
- Кесте болса — Markdown кестесі
Мәтінді тікелей шығар — ```markdown``` блоктарға ОРАМА. Тек қазақ мәтіні, JSON жоқ."""

EXTRACT_PRICING_PROMPT_RU = """Это страница с ценовой таблицей / прайс-листом.
Перепиши ПОЛНОСТЬЮ все строки как таблицу Markdown, ни одну строку не пропускай:

| № | Наименование | Ед.изм. | Кол-во | Цена/Сумма |
|---|--------------|---------|--------|-----------|

Если колонки называются иначе — адаптируй заголовки таблицы под фактическое содержимое.
Выводи таблицу напрямую, без ```markdown``` обёртки, без JSON."""

EXTRACT_PRICING_PROMPT_KZ = """Бұл бағалық кесте / прайс-лист беті.
БАРЛЫҚ жолдарды Markdown кестесі ретінде толық жаз, ешқандай жолды өткізіп жіберме:

| № | Атауы | Өлш.бірл. | Саны | Баға/Сома |
|---|-------|-----------|------|-----------|

Кестені тікелей шығар, ```markdown``` блоксыз, JSON жоқ."""


def rows_to_md(rows: list[dict], page_i: int) -> list[str]:
    """Список строк-dict → Markdown-таблица."""
    lines = [f"\n## Страница {page_i + 1}\n"]
    if not rows:
        return lines
    # Если одна строка с полем "text" — просто текст (без ```markdown``` обёрток)
    if len(rows) == 1 and "text" in rows[0] and len(rows[0]) == 1:
        import re
        text = rows[0]["text"]
        text = re.sub(r'^```markdown\s*\n', '', text, flags=re.MULTILINE)
        text = re.sub(r'\n```\s*\n', '\n\n', text)
        text = re.sub(r'\n```\s*$', '', text)
        lines.append(text.strip())
        return lines
    keys = list(rows[0].keys())
    lines.append("| " + " | ".join(keys) + " |")
    lines.append("|" + "|".join("---" for _ in keys) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(k, "")).replace("|", "\\|") for k in keys) + " |")
    return lines


def task_vision_extract(
    pdf_path: str,
    page_indices: list[int],
    out_path: str | None,
    model: str,
    out_format: str = "md",
    manifest_path: str | None = None,
    lang_filter: str | None = None,
    skip_types: list[str] | None = None,
    only_types: list[str] | None = None,
    split_by_section: bool = False,
    force: bool = False,
):
    """
    Vision-пайплайн с фильтрацией по манифесту.

    Фильтры (применяются если передан --manifest):
      --lang ru/kz      — оставить только страницы на этом языке (или "both")
      --skip-types      — пропустить эти типы (напр. tech_spec)
      --only-types      — взять только эти типы

    Вывод:
      --format md/xlsx
      --split-by-section → папка, каждая секция = отдельный файл
    """
    # Загрузить манифест если есть
    manifest_map: dict[int, dict] = {}
    if manifest_path and Path(manifest_path).exists():
        raw = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        manifest_map = {entry["page"]: entry for entry in raw}
        print(f"Манифест загружен: {len(manifest_map)} стр.")

    # Применить фильтры
    filtered = []
    skipped = []
    for page_i in page_indices:
        page_num = page_i + 1
        entry = manifest_map.get(page_num)

        if entry:
            ptype = entry.get("type", "other")
            plang = entry.get("lang", "both")

            if only_types and ptype not in only_types:
                skipped.append((page_num, f"тип {ptype} не в only-types"))
                continue
            if skip_types and ptype in skip_types:
                skipped.append((page_num, f"тип {ptype} пропущен"))
                continue
            if lang_filter:
                if plang != lang_filter and plang != "both":
                    skipped.append((page_num, f"язык {plang} ≠ {lang_filter}"))
                    continue

        filtered.append(page_i)

    if skipped:
        print(f"Пропущено {len(skipped)} стр.: " + ", ".join(f"{p}({r})" for p, r in skipped[:5]) +
              ("..." if len(skipped) > 5 else ""))
    print(f"Обрабатываем: {len(filtered)} стр.")

    if not filtered:
        print("Нечего извлекать после фильтрации.")
        return

    # Настроить кэш
    cache_base = Path(out_path) if out_path and (split_by_section or Path(out_path).suffix == "") else Path(pdf_path).parent
    if out_path:
        cache_base = Path(out_path) if split_by_section else Path(out_path).parent
    cache_dir = cache_base / ".pdfproc_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Собрать данные с метаданными секции
    pages_data: list[dict] = []  # {page, type, rows}

    for page_i in filtered:
        page_num = page_i + 1
        cache_file = cache_dir / f"{page_num:04d}.json"

        # Проверить кэш
        if not force and cache_file.exists():
            try:
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                if cached.get("model") == model:
                    entry = manifest_map.get(page_num, {})
                    ptype = entry.get("type", "other")
                    print(f"  стр.{page_num} [cache]")
                    pages_data.append({"page": page_num, "type": cached.get("type", ptype), "rows": cached.get("rows", [])})
                    continue
            except Exception:
                pass  # Повреждённый кэш — перезаписать

        img = render_page(pdf_path, page_i, dpi=200)
        entry = manifest_map.get(page_num, {})
        ptype = entry.get("type", "other")

        # Контекст из manifest для текущей и предыдущей страницы
        context_parts = []
        prev_entry = manifest_map.get(page_num - 1)
        if prev_entry and prev_entry.get("summary"):
            context_parts.append(f"Предыдущая стр. ({prev_entry['type']}): {prev_entry['summary']}")
        if entry.get("summary"):
            context_parts.append(f"Эта стр. ({ptype}): {entry['summary']}")
        context_prefix = ("Контекст: " + " | ".join(context_parts) + "\n\n") if context_parts else ""

        # Выбираем промпт по типу страницы и языку
        # Если фильтр не задан — берём язык из manifest; "both" → ru
        page_lang = entry.get("lang", "ru")
        use_lang = lang_filter or ("ru" if page_lang in ("ru", "both") else "kz")
        if ptype in TABLE_TYPES:
            base_prompt = EXTRACT_TABLE_PROMPT_RU if use_lang == "ru" else EXTRACT_TABLE_PROMPT_KZ
            mode = "table"
        elif ptype == "pricing":
            base_prompt = EXTRACT_PRICING_PROMPT_RU if use_lang == "ru" else EXTRACT_PRICING_PROMPT_KZ
            mode = "text"
        else:
            base_prompt = EXTRACT_TEXT_PROMPT_RU if use_lang == "ru" else EXTRACT_TEXT_PROMPT_KZ
            mode = "text"
        prompt = context_prefix + base_prompt

        print(f"  стр.{page_num} [{ptype}/{mode}]...", end=" ", flush=True)

        import time
        t0 = time.time()
        try:
            result = ollama_vision(img, prompt, model=model)

            if mode == "table":
                parsed = parse_json_from_response(result)
                if isinstance(parsed, list):
                    rows = [r for r in parsed if isinstance(r, dict)]
                elif isinstance(parsed, dict):
                    rows = [parsed]
                else:
                    rows = [{"text": result.strip()}]

                # Сначала проверяем loop на оригинальных строках (до sanitize!)
                if is_hallucination_loop(rows):
                    print(f"⚠ зацикливание → retry текст...", end=" ", flush=True)
                    text_prompt = EXTRACT_TEXT_PROMPT_RU if use_lang == "ru" else EXTRACT_TEXT_PROMPT_KZ
                    result2 = ollama_vision(img, text_prompt, model=model)
                    rows = [{"text": result2.strip()}]
                    print(f"{len(result2.split())} слов")
                else:
                    rows = sanitize_rows(rows)  # обрезаем длинные значения только если нет loop
                    print(f"{len(rows)} строк")
            else:
                # Текстовый режим — проверяем зацикливание строк
                text_result = result.strip()
                if is_text_loop(text_result):
                    print(f"⚠ текст-зацикливание → retry...", end=" ", flush=True)
                    retry_prompt = EXTRACT_TEXT_PROMPT_RU if use_lang == "ru" else EXTRACT_TEXT_PROMPT_KZ
                    text_result = ollama_vision(img, retry_prompt, model=model).strip()
                    if is_text_loop(text_result):
                        text_result = f"[Страница {page_num}: не удалось извлечь — зацикливание модели]"
                        print(f"⚠ повторное зацикливание, пропущено")
                    else:
                        print(f"{len(text_result.split())} слов (retry)")
                else:
                    print(f"{len(text_result.split())} слов")
                rows = [{"text": text_result}]

            elapsed = round(time.time() - t0, 1)
            # Записать кэш страницы
            cache_entry = {"page": page_num, "type": ptype, "lang": entry.get("lang", "both"), "model": model, "rows": rows, "elapsed": elapsed}
            cache_file.write_text(json.dumps(cache_entry, ensure_ascii=False, indent=2), encoding="utf-8")
            pages_data.append({"page": page_num, "type": ptype, "rows": rows})
        except Exception as e:
            print(f"ошибка: {e}")
            pages_data.append({"page": page_num, "type": ptype, "rows": [{"error": str(e)}]})

    if not pages_data:
        print("Данных не извлечено.")
        return

    # Вывод
    if split_by_section:
        _write_split(pages_data, out_path, pdf_path, out_format, manifest_map)
    else:
        _write_single(pages_data, out_path, pdf_path, out_format)


def _write_readme(sections: list[dict], out_dir: Path, pdf_path: str, manifest_map: dict):
    """Генерировать README.md — индекс секций для навигации агентов."""
    lines = []
    lines.append(f"# {Path(pdf_path).stem}\n")
    lines.append(f"_Документ обработан автоматически. Страниц в выборке: "
                 f"{sum(len(s['pages']) for s in sections)}_\n")

    lines.append("\n## Содержание\n")
    lines.append("| Файл | Секция | Страницы | Описание |")
    lines.append("|------|--------|----------|----------|")
    for idx, section in enumerate(sections, 1):
        ptype = section["type"]
        page_nums = [p["page"] for p in section["pages"]]
        fname = f"{idx:02d}_{ptype}.md"
        pages_str = f"{page_nums[0]}–{page_nums[-1]}" if len(page_nums) > 1 else str(page_nums[0])
        summaries = [manifest_map.get(p, {}).get("summary", "") for p in page_nums]
        desc = " ".join(s for s in summaries if s)[:120]
        lines.append(f"| [{fname}]({fname}) | {ptype} | {pages_str} | {desc} |")

    lines.append("\n## Краткое содержание\n")
    for idx, section in enumerate(sections, 1):
        ptype = section["type"]
        page_nums = [p["page"] for p in section["pages"]]
        fname = f"{idx:02d}_{ptype}.md"
        lines.append(f"### [{ptype.replace('_', ' ').title()}]({fname})\n")
        for p in page_nums:
            summary = manifest_map.get(p, {}).get("summary", "")
            if summary:
                lines.append(f"- **Стр. {p}**: {summary}")
        lines.append("")

    readme = out_dir / "README.md"
    readme.write_text("\n".join(lines), encoding="utf-8")
    print(f"  → README.md")


def _write_split(pages_data: list[dict], out_dir: str | None, pdf_path: str, out_format: str,
                 manifest_map: dict | None = None):
    """Записать каждую секцию в отдельный файл внутри папки."""
    base = Path(out_dir) if out_dir else Path(pdf_path).with_suffix("")
    base.mkdir(parents=True, exist_ok=True)

    # Группировать последовательные страницы одного типа в секции
    sections = []
    for page_data in pages_data:
        ptype = page_data["type"]
        if sections and sections[-1]["type"] == ptype:
            sections[-1]["pages"].append(page_data)
        else:
            sections.append({"type": ptype, "pages": [page_data]})

    written = []
    for idx, section in enumerate(sections, 1):
        ptype = section["type"]
        page_nums = [p["page"] for p in section["pages"]]
        fname = f"{idx:02d}_{ptype}.{out_format}"
        fpath = base / fname

        if out_format == "md":
            lines = [f"# {ptype.replace('_', ' ').title()}\n",
                     f"_Страницы: {page_nums[0]}–{page_nums[-1] if len(page_nums) > 1 else page_nums[0]}_\n"]
            for pd_ in section["pages"]:
                lines.extend(rows_to_md(pd_["rows"], pd_["page"] - 1))
            fpath.write_text("\n".join(lines), encoding="utf-8")
        else:
            import pandas as pd
            all_rows = []
            for pd_ in section["pages"]:
                for row in pd_["rows"]:
                    row["__page"] = pd_["page"]
                    all_rows.append(row)
            pd.DataFrame(all_rows).to_excel(str(fpath), index=False)

        written.append(fname)
        print(f"  → {fname}  ({len(page_nums)} стр.)")

    if manifest_map:
        _write_readme(sections, base, pdf_path, manifest_map)

    print(f"\nПапка: {base}")
    print(f"Файлов: {len(written) + (1 if manifest_map else 0)}")


def _write_single(pages_data: list[dict], out_path: str | None, pdf_path: str, out_format: str):
    """Записать всё в один файл."""
    if out_format == "md":
        out = out_path or str(Path(pdf_path).with_suffix(".md"))
        lines = []
        for pd_ in pages_data:
            lines.extend(rows_to_md(pd_["rows"], pd_["page"] - 1))
        Path(out).write_text("\n".join(lines), encoding="utf-8")
    else:
        import pandas as pd
        all_rows = []
        for pd_ in pages_data:
            for row in pd_["rows"]:
                row["__page"] = pd_["page"]
                row["__type"] = pd_["type"]
                all_rows.append(row)
        out = out_path or str(Path(pdf_path).with_suffix(".xlsx"))
        pd.DataFrame(all_rows).to_excel(out, index=False)

    total_rows = sum(len(p["rows"]) for p in pages_data)
    print(f"\nСохранено {total_rows} строк → {out}")


# ── задача: summarize ──────────────────────────────────────────────────────

def task_summarize(pdf_path: str, page_indices: list[int], model: str):
    texts = get_page_texts(pdf_path, page_indices)
    full_text = "\n\n".join(f"[Стр.{i+1}]\n{t}" for i, t in sorted(texts.items()) if t.strip())
    if not full_text.strip():
        print("Текст не извлечён.")
        return
    summaries = []
    for chunk in chunk_text(full_text, max_chars=4000):
        summaries.append(ollama_query(
            f"Кратко изложи суть этого фрагмента документа на русском:\n\n{chunk}",
            model=model,
            system="Отвечай кратко, по-русски.",
        ))
    if len(summaries) > 1:
        result = ollama_query(f"Объедини эти части в единое краткое резюме:\n\n" + "\n---\n".join(summaries), model=model)
    else:
        result = summaries[0]
    print("\n" + "=" * 60)
    print(result)


# ── задача: qa ─────────────────────────────────────────────────────────────

def task_qa(pdf_path: str, page_indices: list[int], question: str, model: str):
    texts = get_page_texts(pdf_path, page_indices)
    context = "\n\n".join(f"[Стр.{i+1}]\n{t}" for i, t in sorted(texts.items()) if t.strip())
    if not context.strip():
        print("Текст не извлечён.")
        return
    answers = []
    for i, chunk in enumerate(chunk_text(context, max_chars=3500)):
        result = ollama_query(
            f"Контекст:\n{chunk}\n\nВопрос: {question}\n\nЕсли ответ есть — дай его. Иначе: 'нет данных'.",
            model=model, system="Отвечай только по контексту. По-русски.",
        )
        if "нет данных" not in result.lower():
            answers.append(result)
    if answers:
        final = ollama_query(f"Вопрос: {question}\n\n" + "\n---\n".join(answers) + "\n\nФинальный ответ:", model=model) if len(answers) > 1 else answers[0]
        print("\n" + "=" * 60)
        print(final)
    else:
        print("Ответ не найден.")


# ── задача: info ───────────────────────────────────────────────────────────

def task_info(pdf_path: str):
    import pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        print(f"Страниц: {total}")
        sample = list(range(min(5, total)))
        texts = {i: (pdf.pages[i].extract_text() or "") for i in sample}
        has_text = sum(1 for t in texts.values() if len(t) > 50)
        print(f"Нативный текст: {'есть' if has_text else 'нет (скан)'} (проверено {len(sample)} стр.)")
        tables_count = sum(len(pdf.pages[i].extract_tables()) for i in sample)
        print(f"Таблиц на первых {len(sample)} стр.: {tables_count}")


# ── точка входа ────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Дешёвый PDF-процессор с манифестом и фильтрами")
    p.add_argument("pdf", help="Путь к PDF")
    p.add_argument("--task", choices=["info", "manifest", "vision_extract", "summarize", "qa"],
                   default="info")
    p.add_argument("--pages", default=None, help="Страницы: '1-10' или '1,3,5-8'")
    p.add_argument("--model", default="auto", help="Модель: 'auto' (автовыбор), ollama (qwen2.5vl:7b), облако (anthropic:claude-haiku-4-5, openai:gpt-4o-mini)")

    # Фильтры (работают с --manifest)
    p.add_argument("--manifest", default=None, help="Путь к manifest.json для фильтрации")
    p.add_argument("--lang", choices=["ru", "kz"], default=None,
                   help="Оставить только страницы на этом языке")
    p.add_argument("--skip-types", default=None,
                   help="Пропустить типы через запятую: tech_spec,appendix")
    p.add_argument("--only-types", default=None,
                   help="Взять только эти типы: pricing,contract_header")

    # Вывод
    p.add_argument("--out", default=None, help="Выходной файл или папка")
    p.add_argument("--format", choices=["md", "xlsx"], default="md",
                   help="Формат вывода (default: md)")
    p.add_argument("--split-by-section", action="store_true",
                   help="Создать папку, каждая секция = отдельный файл")

    # Прочее
    p.add_argument("--question", "-q", default="", help="Вопрос для задачи qa")
    p.add_argument("--ocr", action="store_true", help="Принудительный OCR")
    p.add_argument("--force", action="store_true", help="Игнорировать кэш и пересчитать всё заново")
    args = p.parse_args()

    # Автовыбор модели
    if args.model == "auto":
        selected = model_detect.auto_select_model(require_vision=True, verbose=True)
        if selected is None:
            sys.exit(1)
        args.model = selected

    import pdfplumber
    with pdfplumber.open(args.pdf) as pdf:
        total_pages = len(pdf.pages)

    if args.pages:
        page_indices = parse_page_range(args.pages, total_pages)
    else:
        page_indices = list(range(total_pages))

    print(f"PDF: {args.pdf} ({total_pages} стр.)")
    print(f"Задача: {args.task} | Модель: {args.model} | Стр.: {len(page_indices)}")
    print("-" * 60)

    skip_types = [s.strip() for s in args.skip_types.split(",")] if args.skip_types else None
    only_types = [s.strip() for s in args.only_types.split(",")] if args.only_types else None

    if args.task == "info":
        task_info(args.pdf)
    elif args.task == "manifest":
        task_manifest(args.pdf, page_indices, args.out, args.model, force=args.force)
    elif args.task == "vision_extract":
        task_vision_extract(
            args.pdf, page_indices, args.out, args.model,
            out_format=args.format,
            manifest_path=args.manifest,
            lang_filter=args.lang,
            skip_types=skip_types,
            only_types=only_types,
            split_by_section=args.split_by_section,
            force=args.force,
        )
    elif args.task == "summarize":
        task_summarize(args.pdf, page_indices, args.model)
    elif args.task == "qa":
        if not args.question:
            print("Укажи вопрос: --question '...'")
            sys.exit(1)
        task_qa(args.pdf, page_indices, args.question, args.model)


if __name__ == "__main__":
    main()
