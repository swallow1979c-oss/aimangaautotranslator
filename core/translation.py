import re
import json
from typing import List, Dict, Any, Optional
from utils.logging import log_message
import os
import time
from core.models import TranslationConfig
from utils.endpoints import (
    call_gemini_endpoint,
    call_openai_endpoint,
    call_anthropic_endpoint,
    call_openrouter_endpoint,
    call_openai_compatible_endpoint,
)
# Regex to find numbered lines in LLM responses 503 
TRANSLATION_PATTERN = re.compile(
    r'^\s*(\d+)\s*:\s*"?\s*(.*?)\s*"?\s*(?=\s*\n\s*\d+\s*:|\s*$)', re.MULTILINE | re.DOTALL
)

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        # убираем первую строку ```... и конечную ```
        lines = s.splitlines()
        # убираем первую и последнюю строку, если там ```
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return s

def _parse_json_id_map(response_text: Optional[str], expected_ids, provider: str, debug: bool = False) -> Optional[dict]:
    """
    Ожидаем JSON-массив объектов вида {"id": "<id>", "text": "<...>"}.
    Возвращаем dict id->text. Если что-то не так — None.
    """
    if not response_text:
        return None
    try:
        cleaned = _strip_code_fences(response_text)
        data = json.loads(cleaned)
        if isinstance(data, dict) and "items" in data:
            data = data["items"]
        if not isinstance(data, list):
            return None
        out = {}
        for item in data:
            if not isinstance(item, dict):
                continue
            bid = item.get("id")
            txt = item.get("text", "")
            if isinstance(bid, str):
                out[bid] = txt if isinstance(txt, str) else ""
        # фильтруем только ожидаемые id
        return {bid: out.get(bid, "") for bid in expected_ids}
    except Exception as e:
        log_message(f"JSON parse failed: {e}", verbose=debug)
        return None

# Helper functions for sorting keys
def _sort_key_ltr(d):
    return (d["grid_pos"][0], d["center_y"], d["center_x"])


def _sort_key_rtl(d):
    return (d["grid_pos"][0], d["center_y"], -d["center_x"])


def sort_bubbles_by_reading_order(
    detections,
    image_height,
    image_width,
    reading_direction="ltr",
    grid_rows=12,
    grid_cols=12,
):
    """
    Сортировка баблов в порядке чтения.
    LTR -> слева направо, сверху вниз.
    RTL -> справа налево, сверху вниз (как в японской манге).
    """

    def _sort_key_ltr(d):
        # сверху вниз по рядам, внутри ряда — слева направо
        return (d["grid_pos"][0], float(d["center_x"]), float(d["center_y"]))

    def _sort_key_rtl(d):
        # сверху вниз по рядам, внутри ряда — справа налево
        return (d["grid_pos"][0], -float(d["center_x"]), float(d["center_y"]))

    sorted_detections = []
    for detection in detections:
        # гарантируем, что bbox — числа
        x1, y1, x2, y2 = map(float, detection["bbox"])
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        row_idx = int((center_y / image_height) * grid_rows)
        col_idx = int((center_x / image_width) * grid_cols)

        detection["grid_pos"] = (row_idx, col_idx)
        detection["center_x"] = center_x
        detection["center_y"] = center_y

        sorted_detections.append(detection)

    # выбор способа сортировки
    if reading_direction.lower() == "rtl":
        sort_key = _sort_key_rtl
    else:
        sort_key = _sort_key_ltr

    return sorted(sorted_detections, key=sort_key)



MAX_RETRIES = 5
RETRY_DELAY = 10

# --- Менеджер Gemini ключей ---
class GeminiKeyManager:
    def __init__(self, keys: List[str]):
        self.keys = keys
        self.exhausted = set()  # номера ключей, которые нельзя использовать
        self.current_index = 0

    def get_next_key(self):
        total_keys = len(self.keys)
        for _ in range(total_keys):
            idx = self.current_index
            key = self.keys[idx]
            self.current_index = (self.current_index + 1) % total_keys
            if idx not in self.exhausted:
                return idx, key
        return None, None  # все ключи исчерпаны

    def mark_exhausted(self, idx: int):
        self.exhausted.add(idx)


# Создаём один глобальный менеджер ключей для всей сессии/батча
gemini_key_manager = None  # будет инициализирован перед первым вызовом
current_key_index = None   # активный ключ


# --- Основная функция вызова LLM с retry ---
def _call_llm_with_retry(
    config: TranslationConfig, parts: List[Dict[str, Any]], prompt_text: str, debug: bool = False
) -> Optional[str]:
    global gemini_key_manager, current_key_index

    provider = config.provider
    model_name = config.model_name
    temperature = config.temperature
    top_p = config.top_p
    top_k = config.top_k

    api_parts = parts + [{"text": prompt_text}]

    if provider == "Gemini":
        if not config.gemini_api_keys:
            raise ValueError("Gemini API key(s) missing in TranslationConfig.")

        # Инициализация глобального менеджера ключей при первом вызове
        if gemini_key_manager is None:
            gemini_key_manager = GeminiKeyManager(config.gemini_api_keys)

        is_gemini_25_series = model_name.startswith("gemini-2.5")
        max_output_tokens = 10240 if is_gemini_25_series else 2048
        generation_config = {
            "temperature": temperature,
            "topP": top_p,
            "topK": top_k,
            "maxOutputTokens": max_output_tokens,
        }

        if "gemini-2.5-flash" in model_name and config.enable_thinking:
            log_message(f"Including thoughts for {model_name}", verbose=debug)
        elif "gemini-2.5-flash" in model_name:
            generation_config["thinkingConfig"] = {"thinkingBudget": 0}
            log_message(f"Not including thoughts for {model_name}", verbose=debug)

        last_error = None

        last_error = None

        while True:
            # если нет активного ключа → взять новый
            if current_key_index is None:
                idx, api_key = gemini_key_manager.get_next_key()
                if api_key is None:
                    log_message("❌ Все Gemini ключи исчерпаны/запрещены. Либо вводить новые, либо ждать.", always_print=True)
                    os._exit(1)
                current_key_index = idx
                log_message(f"Using Gemini key {idx + 1}/{len(config.gemini_api_keys)}", verbose=True)
            else:
                idx = current_key_index
                api_key = config.gemini_api_keys[idx]

            # пробуем вызвать модель с этим ключом
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    return call_gemini_endpoint(
                        api_key=api_key,
                        model_name=model_name,
                        parts=api_parts,
                        generation_config=generation_config,
                        debug=debug,
                    )
                except Exception as e:
                    msg = str(e).lower()
                    last_error = e
                    if "503" in msg or "unavailable" in msg or "overloaded" in msg:
                        log_message(
                            f"Gemini key {idx + 1} got 503/unavailable (attempt {attempt}/{MAX_RETRIES}). Retrying in {RETRY_DELAY}s...",
                            verbose=True,
                        )
                        time.sleep(RETRY_DELAY)
                        continue
                    elif "500" in msg or "internal" in msg:
                        log_message(
                            f"Gemini key {idx + 1} got 500/internal error (attempt {attempt}/{MAX_RETRIES}). Retrying in {RETRY_DELAY}s...",
                            verbose=True,
                        )
                        time.sleep(RETRY_DELAY)
                        continue
                    elif "quota" in msg or "403" in msg:
                        log_message(
                            f"Gemini key {idx + 1} exhausted/forbidden. Switching to next key...",
                            verbose=True,
                        )
                        gemini_key_manager.mark_exhausted(idx)
                        current_key_index = None  # сбросить текущий ключ → взять новый
                        break  # выйти из retry цикла → переключиться на новый ключ
                    else:
                        raise

            else:
                # все 5 попыток закончились
                if "500" in str(last_error).lower() or "internal" in str(last_error).lower():
                    log_message(
                        f"Gemini key {idx + 1} failed with 500/internal after {MAX_RETRIES} attempts. Aborting...",
                        verbose=True,
                    )
                    raise last_error
                else:
                    # для 503 → помечаем ключ исчерпанным и пробуем следующий
                    log_message(
                        f"Gemini key {idx + 1} failed {MAX_RETRIES}/{MAX_RETRIES} attempts. Marking as exhausted...",
                        verbose=True,
                    )
                    gemini_key_manager.mark_exhausted(idx)
                    current_key_index = None
                    continue

    elif provider == "OpenAI":
        api_key = config.openai_api_key
        if not api_key:
            raise ValueError("OpenAI API key is missing.")
        lm = model_name.lower()
        is_gpt5 = lm.startswith("gpt-5")
        is_reasoning_capable = is_gpt5 or lm.startswith("o1") or lm.startswith("o3") or lm.startswith("o4-mini")
        max_output_tokens = 10240 if is_reasoning_capable else 2048
        generation_config = {"temperature": temperature, "top_p": top_p, "max_output_tokens": max_output_tokens}
        if config.reasoning_effort:
            generation_config["reasoning_effort"] = config.reasoning_effort
        return call_openai_endpoint(api_key=api_key, model_name=model_name, parts=api_parts, generation_config=generation_config, debug=debug)

    elif provider == "Anthropic":
        api_key = config.anthropic_api_key
        if not api_key:
            raise ValueError("Anthropic API key is missing.")
        clamped_temp = min(temperature, 1.0)
        lm = model_name.lower()
        reasoning_models = ["claude-opus-4-1", "claude-opus-4", "claude-sonnet-4", "claude-3-7-sonnet"]
        is_reasoning_model = any(lm.startswith(p) for p in reasoning_models)
        max_tokens = 10240 if is_reasoning_model else 2048
        generation_config = {
            "temperature": clamped_temp,
            "top_p": top_p,
            "top_k": top_k,
            "max_tokens": max_tokens,
            "anthropic_thinking": bool(config.enable_thinking and is_reasoning_model),
        }
        return call_anthropic_endpoint(api_key=api_key, model_name=model_name, parts=api_parts, generation_config=generation_config, debug=debug)

    elif provider == "OpenRouter":
        api_key = config.openrouter_api_key
        if not api_key:
            raise ValueError("OpenRouter API key is missing.")
        return call_openrouter_endpoint(api_key=api_key, model_name=model_name, parts=api_parts, generation_config={"temperature": temperature, "top_p": top_p, "top_k": top_k}, debug=debug)

    elif provider == "OpenAI-Compatible":
        base_url = config.openai_compatible_url
        api_key = config.openai_compatible_api_key
        if not base_url:
            raise ValueError("OpenAI-Compatible URL is missing.")
        generation_config = {"temperature": temperature, "top_p": top_p, "top_k": top_k, "max_tokens": 2048}
        return call_openai_compatible_endpoint(base_url=base_url, api_key=api_key, model_name=model_name, parts=api_parts, generation_config=generation_config, debug=debug)

    else:
        raise ValueError(f"Unknown translation provider: {provider}")

def _parse_llm_response(
    response_text: Optional[str], expected_count: int, provider: str, debug: bool = False
) -> Optional[List[str]]:
    """Internal helper to parse numbered list responses from LLM."""
    if response_text is None:
        log_message(f"Parsing response: Received None from {provider} API call.", verbose=debug)
        return None
    elif response_text == "":
        log_message(
            f"Parsing response: Received empty string from {provider} API,"
            f" likely no text detected or processing failed.",
            verbose=debug,
        )
        return [f"[{provider}: No text/empty response]" for _ in range(expected_count)]

    try:
        log_message(f"Parsing response: Raw text received from {provider}:\n---\n{response_text}\n---", verbose=debug)

        matches = TRANSLATION_PATTERN.findall(response_text)
        log_message(
            f"Parsing response: Regex matches found: {len(matches)} vs expected {expected_count}", verbose=debug
        )

        # Fallback for non-numbered lists
        if len(matches) < expected_count:
            log_message(
                f"Parsing response warning: Regex found fewer items ({len(matches)}) than expected ({expected_count}). "
                f"Attempting newline split fallback.",
                verbose=debug,
                always_print=True,
            )
            lines = [line.strip() for line in response_text.split("\n") if line.strip()]
            lines = [re.sub(r'^\d+\s*[:\-\)]\s*', '', line) for line in lines]
            # Remove potential leading/trailing markdown code blocks
            if lines and lines[0].startswith("```"):
                lines.pop(0)
            if lines and lines[-1].endswith("```"):
                lines.pop(-1)

            # Basic check: does the number of lines match expected bubbles?
            if len(lines) == expected_count:
                log_message(
                    f"Parsing response: Fallback successful. Using {len(lines)} lines as translations.",
                    verbose=debug,
                    always_print=True,
                )
                return lines
            else:
                log_message(
                    f"Parsing response warning: Fallback failed. Newline count ({len(lines)}) "
                    f"still doesn't match expected count ({expected_count}). "
                    f"Proceeding with regex matches found ({len(matches)}).",
                    verbose=debug,
                    always_print=True,
                )

        # Original logic if regex worked or fallback failed
        parsed_dict = {}
        for num_str, text in matches:
            try:
                num = int(num_str)
                if 1 <= num <= expected_count:
                    parsed_dict[num] = text.strip()
                else:
                    log_message(
                        f"Parsing response warning: "
                        f"Parsed number {num} is out of expected range (1-{expected_count}).",
                        verbose=debug,
                    )
            except ValueError:
                log_message(
                    f"Parsing response warning: Could not parse number '{num_str}' in response line.", verbose=debug
                )

        final_list = []
        for i in range(1, expected_count + 1):
            final_list.append(parsed_dict.get(i, f"[{provider}: No text for bubble {i}]"))

        if len(final_list) != expected_count:
            log_message(
                f"Parsing response warning: Expected {expected_count} items, but constructed {len(final_list)}."
                f" Check raw response and parsing logic.",
                verbose=debug,
            )

        return final_list

    except Exception as e:
        log_message(f"Error parsing successful {provider} API response content: {str(e)}", always_print=True)
        # Treat parsing error as a failure for all bubbles
        return [f"[{provider}: Error parsing response]" for _ in range(expected_count)]

def call_translation_api_batch(
    config: TranslationConfig,
    images_b64: List[str],
    full_image_b64: str,
    debug: bool = False,
    bubble_ids: Optional[List[str]] = None,
):
    """
    Если передан bubble_ids: возвращает dict id->text (идемпотентное сопоставление).
    Иначе оставляет старое поведение и возвращает List[str] по порядку.
    """
    provider = config.provider
    input_language = config.input_language
    output_language = config.output_language
    reading_direction = config.reading_direction
    translation_mode = config.translation_mode
    num_bubbles = len(images_b64)
    reading_order_desc = (
        "right-to-left, top-to-bottom" if reading_direction == "rtl" else "left-to-right, top-to-bottom"
    )

    # --- Подготовка parts ---
    # Если есть bubble_ids: между маркерами и картинками держим строгую структуру:
    # [FULL_PAGE], then for each: [TEXT "BUBBLE_ID:<id>"], [IMAGE bubble]
    if bubble_ids:
        if len(bubble_ids) != num_bubbles:
            raise ValueError("bubble_ids length must match images_b64 length")

        parts_with_ids = [{"inline_data": {"mime_type": "image/jpeg", "data": full_image_b64}}]
        for bid, img_b64 in zip(bubble_ids, images_b64):
            parts_with_ids.append({"text": f"[BUBBLE_ID:{bid}]"})
            parts_with_ids.append({"inline_data": {"mime_type": "image/jpeg", "data": img_b64}})

    else:
        # старое поведение: один полный кадр + все кропы без id
        parts_with_ids = [{"inline_data": {"mime_type": "image/jpeg", "data": full_image_b64}}]
        for img_b64 in images_b64:
            parts_with_ids.append({"inline_data": {"mime_type": "image/jpeg", "data": img_b64}})

    try:
        if translation_mode == "two-step":
            # ---------- STEP 1: OCR ----------
            if bubble_ids:
                ocr_prompt = f"""You will receive one full manga/comic page image for context, then for each bubble:
a text marker "[BUBBLE_ID:<id>]" immediately followed by the bubble image.
For EACH bubble, extract ONLY the original {input_language} text.
Return a STRICT JSON array. Each item: {{"id":"<id>","text":"<extracted>"}}.
Return ONLY JSON, no explanations, no markdown."""
                ocr_response_text = _call_llm_with_retry(config, parts_with_ids, ocr_prompt, debug)
                ocr_map = _parse_json_id_map(ocr_response_text, bubble_ids, provider + "-OCR", debug)
                if ocr_map is None:
                    log_message("OCR JSON parse failed. Falling back to placeholders.", verbose=debug, always_print=True)
                    return {bid: f"[{provider}: OCR call failed/blocked]" for bid in bubble_ids}
                # ---------- STEP 2: TRANSLATE ----------
                # Формируем компактный блок для перевода (только текст), но сохраняем ID
                items_block = [{"id": bid, "text": ocr_map.get(bid, "")} for bid in bubble_ids]
                items_json = json.dumps(items_block, ensure_ascii=False)

                translation_prompt = f"""Analyze the provided full page context (already given above).
Translate the following extracted bubble texts from {input_language} to {output_language}.
DO NOT translate Japanese names with suffix "-san" into Mr./Ms. 
Choose wording that matches tone and context. Style rules:
- Italic: *text* for thoughts/flashbacks/distant sounds/devices.
- Bold: **text** for SFX, shouting, timestamps.
- Bold Italic: ***text*** for loud SFX/dialogue in flashbacks/devices.
Return a STRICT JSON array mirroring the input, each item: {{"id":"<id>","text":"<translation or [OCR FAILED]>"}}.
Return ONLY JSON.

INPUT:
{items_json}"""

                # Для шага перевода достаточно дать только full image (контекст) + текстовый prompt
                translation_parts = [{"inline_data": {"mime_type": "image/jpeg", "data": full_image_b64}}]
                translation_response_text = _call_llm_with_retry(config, translation_parts, translation_prompt, debug)
                tr_map = _parse_json_id_map(translation_response_text, bubble_ids, provider + "-Translate", debug)
                if tr_map is None:
                    log_message("Translate JSON parse failed. Returning OCR map as placeholders.", verbose=debug, always_print=True)
                    # если не разобрали перевод, вернём OCR или пусто по ID
                    return {bid: ocr_map.get(bid, "") for bid in bubble_ids}
                return tr_map

            else:
                # старое двухшаговое поведение БЕЗ id → вернём список
                # --- Step 1: OCR (как было) ---
                ocr_prompt = f"""Analyze the {num_bubbles} individual speech bubble images extracted from a manga/comic page in reading order ({reading_order_desc}).
For each individual speech bubble image, only extract the original {input_language} text.
Provide your response in this exact format, with each extraction on a new line:
1: [Extracted text for bubble 1]
...
{num_bubbles}: [Extracted text for bubble {num_bubbles}]
Do not include translations, explanations, or any other text in your response."""
                # parts только для кропов
                ocr_parts = []
                for img_b64 in images_b64:
                    ocr_parts.append({"inline_data": {"mime_type": "image/jpeg", "data": img_b64}})
                ocr_response_text = _call_llm_with_retry(config, ocr_parts, ocr_prompt, debug)
                extracted_texts = _parse_llm_response(ocr_response_text, num_bubbles, provider + "-OCR", debug)
                if extracted_texts is None:
                    return [f"[{provider}: OCR call failed/blocked]"] * num_bubbles

                # --- Step 2: Translate (как было) ---
                formatted = []
                ocr_failed_indices = set()
                for i, t in enumerate(extracted_texts):
                    if f"[{provider}-OCR:" in t:
                        formatted.append(f"{i + 1}: [OCR FAILED]")
                        ocr_failed_indices.add(i)
                    else:
                        formatted.append(f"{i + 1}: {t}")
                extracted_text_block = "\n".join(formatted)

                translation_prompt = f"""Analyze the full manga/comic page image provided for context ({reading_order_desc} reading direction).
Then, translate the following extracted speech bubble texts (numbered 1 to {num_bubbles}) from {input_language} to {output_language}.
DO NOT translate Japanese names with suffix "-san" into Mr./Ms. 
Apply styling markers as needed (*italic*, **bold**, ***bold italic***).
Respond exactly as:
1: [...]
2: [...]
...
{num_bubbles}: [...]
For any entry "[OCR FAILED]" return exactly "[OCR FAILED]"."""
                translation_parts = [{"inline_data": {"mime_type": "image/jpeg", "data": full_image_b64}}]
                translation_response_text = _call_llm_with_retry(config, translation_parts + [{"text": translation_prompt}], "", debug)
                final_translations = _parse_llm_response(translation_response_text, num_bubbles, provider + "-Translate", debug)
                if final_translations is None:
                    final_translations = [f"[{provider}: Translation call failed/blocked]"] * num_bubbles
                return final_translations

        elif translation_mode == "one-step":
            if bubble_ids:
                one_step_prompt = f"""You will receive: one full page image for context, then for each bubble a text marker "[BUBBLE_ID:<id>]" immediately followed by that bubble's image.
For EACH bubble:
1) Extract the {input_language} text and translate it to {output_language} with natural tone matching the context.
2) Apply styling if needed:
   - *italic* for thoughts/flashbacks/distant sounds/devices
   - **bold** for SFX/shouting/timestamps
   - ***bold italic*** for loud SFX/dialogue in flashbacks/devices
Return a STRICT JSON array. Each item: {{"id":"<id>","text":"<translation>"}}.
Return ONLY JSON. No explanations, no markdown fences.
Reading direction: {reading_order_desc}."""
                response_text = _call_llm_with_retry(config, parts_with_ids, one_step_prompt, debug)
                id_map = _parse_json_id_map(response_text, bubble_ids, provider, debug)
                if id_map is None:
                    log_message("One-step JSON parse failed. Falling back to numbered parsing (order-based).", verbose=debug, always_print=True)
                    # как fallback: старый парсер + склейка по порядку
                    translations = _parse_llm_response(response_text, num_bubbles, provider, debug) or []
                    # упакуем в dict по текущему порядку IDs
                    return {bid: (translations[i] if i < len(translations) else "") for i, bid in enumerate(bubble_ids)}
                return id_map
            else:
                # Старое поведение (без id) → список
                one_step_prompt = f"""Analyze the full page image, then the {num_bubbles} bubble images in reading order ({reading_order_desc}).
For each bubble: extract {input_language} and translate to {output_language}, applying *italic*, **bold**, ***bold italic*** if needed.
Respond exactly as:
1: [...]
2: [...]
...
{num_bubbles}: [...]
No extra text."""
                response_text = _call_llm_with_retry(config, parts_with_ids, one_step_prompt, debug)
                translations = _parse_llm_response(response_text, num_bubbles, provider, debug)
                if translations is None:
                    translations = [f"[{provider}: API call failed/blocked]"] * num_bubbles
                return translations
        else:
            raise ValueError(f"Unknown translation_mode specified in config: {translation_mode}")
    except (ValueError, RuntimeError) as e:
        log_message(f"Translation failed: {e}", always_print=True)
        if bubble_ids:
            return {bid: f"[Translation Error: {e}]" for bid in bubble_ids}
        return [f"[Translation Error: {e}]"] * num_bubbles


