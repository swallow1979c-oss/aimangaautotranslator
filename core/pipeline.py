import argparse
import base64
import os
import time
import re
from pathlib import Path
from typing import Optional, Union, Callable, Dict, Any
import sys
import cv2
from PIL import Image
import torch

from core.models import (
    DetectionConfig,
    CleaningConfig,
    TranslationConfig,
    RenderingConfig,
    OutputConfig,
    MangaTranslatorConfig,
)
from .detection import detect_speech_bubbles
from .cleaning import clean_speech_bubbles
from .image_utils import pil_to_cv2, cv2_to_pil, save_image_with_compression
from .translation import call_translation_api_batch, sort_bubbles_by_reading_order
from utils.logging import log_message
from .rendering import render_text_skia

def translate_and_render(
    image_path: Union[str, Path], config: MangaTranslatorConfig, output_path: Optional[Union[str, Path]] = None
):
    """
    Main function to translate manga speech bubbles and render translations using a config object.
    Retries the whole page if Gemini-OCR returns 'No text/empty response'.
    """
    MAX_PAGE_RETRIES = 5  # сколько раз пробовать заново страницу

    for attempt in range(MAX_PAGE_RETRIES):
        start_time = time.time()
        image_path = Path(image_path)
        verbose = config.verbose
        device = config.device

        log_message(f"Using device: {device}", verbose=verbose)

        try:
            pil_original = Image.open(image_path)
        except FileNotFoundError:
            log_message(f"Error: Input image not found at {image_path}", always_print=True)
            raise
        except Exception as e:
            log_message(f"Error opening image {image_path}: {e}", always_print=True)
            raise

        original_mode = pil_original.mode
        log_message(f"Original image mode: {original_mode}", verbose=verbose)

        desired_format = config.output.output_format
        output_ext_for_mode = Path(output_path).suffix.lower() if output_path else image_path.suffix.lower()

        if desired_format == "jpeg" or (desired_format == "auto" and output_ext_for_mode in [".jpg", ".jpeg"]):
            target_mode = "RGB"
        else:
            target_mode = "RGBA"
        log_message(
            f"Target image mode determined as: {target_mode} "
            f"(based on format: {desired_format}, output_ext: {output_ext_for_mode})",
            verbose=verbose,
        )

        pil_image_processed = pil_original
        if pil_image_processed.mode != target_mode:
            if target_mode == "RGB":
                if (
                    pil_image_processed.mode == "RGBA"
                    or pil_image_processed.mode == "LA"
                    or (pil_image_processed.mode == "P" and "transparency" in pil_image_processed.info)
                ):
                    log_message(f"Converting {pil_image_processed.mode} to RGB (flattening transparency)", verbose=verbose)
                    background = Image.new("RGB", pil_image_processed.size, (255, 255, 255))
                    try:
                        mask = None
                        if pil_image_processed.mode == "RGBA":
                            mask = pil_image_processed.split()[3]
                        elif pil_image_processed.mode == "LA":
                            mask = pil_image_processed.split()[1]
                        elif pil_image_processed.mode == "P" and "transparency" in pil_image_processed.info:
                            temp_rgba = pil_image_processed.convert("RGBA")
                            mask = temp_rgba.split()[3]

                        if mask:
                            background.paste(pil_image_processed, mask=mask)
                            pil_image_processed = background
                        else:
                            pil_image_processed = pil_image_processed.convert("RGB")
                    except Exception as paste_err:
                        log_message(
                            f"Warning: Error during RGBA/LA/P -> RGB conversion paste: {paste_err}. "
                            f"Trying alpha_composite.",
                            verbose=verbose,
                        )
                        try:
                            background_comp = Image.new("RGB", pil_image_processed.size, (255, 255, 255))
                            img_rgba_for_composite = (
                                pil_image_processed
                                if pil_image_processed.mode == "RGBA"
                                else pil_image_processed.convert("RGBA")
                            )
                            pil_image_processed = Image.alpha_composite(
                                background_comp.convert("RGBA"), img_rgba_for_composite
                            ).convert("RGB")
                            log_message("Successfully used alpha_composite for RGB conversion.", verbose=verbose)
                        except Exception as composite_err:
                            log_message(
                                f"Warning: alpha_composite also failed: {composite_err}. Using simple convert.",
                                verbose=verbose,
                            )
                            pil_image_processed = pil_image_processed.convert("RGB")
                else:
                    log_message(f"Converting {pil_image_processed.mode} to RGB", verbose=verbose)
                    pil_image_processed = pil_image_processed.convert("RGB")
            elif target_mode == "RGBA":
                log_message(f"Converting {pil_image_processed.mode} to RGBA", verbose=verbose)
                pil_image_processed = pil_image_processed.convert("RGBA")

        original_cv_image = pil_to_cv2(pil_image_processed)

        # --- Encode Full Image for Context ---
        full_image_b64 = None
        try:
            is_success, buffer = cv2.imencode(".jpg", original_cv_image)
            if not is_success:
                raise ValueError("Full image encoding to JPEG failed")
            full_image_b64 = base64.b64encode(buffer).decode("utf-8")
            log_message("Encoded full image for translation context.", verbose=verbose)
        except Exception as e:
            log_message(f"Error encoding full image to base64: {e}. Translation might lack context.", always_print=True)

        # --- Detection ---
        log_message("Detecting speech bubbles...", verbose=verbose)
        try:
            bubble_data = detect_speech_bubbles(
                image_path, config.detector_model_path, config.detection.confidence, verbose=verbose, device=device
            )
        except Exception as e:
            log_message(f"Error during bubble detection: {e}. Proceeding without bubbles.", always_print=True)
            bubble_data = []

        final_image_to_save = pil_image_processed

        if not bubble_data:
            log_message("No speech bubbles detected or detection failed.", always_print=True)
        else:
            log_message(f"Detected {len(bubble_data)} potential bubbles.", verbose=verbose)

            # --- Cleaning ---
            try:
                cleaned_image_cv, processed_bubbles_info = clean_speech_bubbles(
                    image_path=image_path,
                    cleaner_model_path=config.cleaner_model_path,
                    confidence=config.detection.confidence,
                    pre_computed_detections=bubble_data,  # bbox от детектора
                    device=device,
                    dilation_kernel_size=config.cleaning.dilation_kernel_size,
                    dilation_iterations=config.cleaning.dilation_iterations,
                    use_otsu_threshold=config.cleaning.use_otsu_threshold,
                    min_contour_area=config.cleaning.min_contour_area,
                    closing_kernel_size=config.cleaning.closing_kernel_size,
                    closing_iterations=config.cleaning.closing_iterations,
                    closing_kernel_shape=config.cleaning.closing_kernel_shape,
                    constraint_erosion_kernel_size=config.cleaning.constraint_erosion_kernel_size,
                    constraint_erosion_iterations=config.cleaning.constraint_erosion_iterations,
                    verbose=verbose,
                )
            except Exception as e:
                log_message(
                    f"Error during bubble cleaning: {e}. Proceeding with uncleaned image.",
                    always_print=True
                )
                cleaned_image_cv = original_cv_image.copy()
                processed_bubbles_info = []

            pil_cleaned_image = cv2_to_pil(cleaned_image_cv)
            if pil_cleaned_image.mode != target_mode:
                pil_cleaned_image = pil_cleaned_image.convert(target_mode)
            final_image_to_save = pil_cleaned_image

            if not config.cleaning_only:
                # --- Prepare bubble crops + stable IDs ---
                for bubble in bubble_data:
                    x1, y1, x2, y2 = bubble["bbox"]

                    # Стабильный ID для бабла (на уровне страницы достаточно bbox-коорд)
                    bubble["id"] = f"{x1}_{y1}_{x2}_{y2}"

                    bubble_image_cv = original_cv_image[y1:y2, x1:x2].copy()
                    try:
                        is_success, buffer = cv2.imencode(".jpg", bubble_image_cv)
                        if not is_success:
                            raise ValueError("cv2.imencode failed")
                        bubble["image_b64"] = base64.b64encode(buffer).decode("utf-8")
                    except Exception as e:
                        bubble["image_b64"] = None
                        log_message(f"Error encoding bubble: {e}", always_print=True)

                valid_bubble_data = [b for b in bubble_data if b.get("image_b64")]
                if valid_bubble_data and full_image_b64:
                    H, W = original_cv_image.shape[:2]
                    sorted_bubble_data = sort_bubbles_by_reading_order(
                        valid_bubble_data,
                        H,
                        W,
                        config.translation.reading_direction,
                    )

                    # 2) Готовим батч и список ID в том же порядке
                    bubble_images_b64 = [b["image_b64"] for b in sorted_bubble_data]
                    bubble_ids = [b["id"] for b in sorted_bubble_data]

                    # 3) Вызываем перевод с ID-маркировкой (вернёт dict id->text)
                    try:
                        translations_by_id = call_translation_api_batch(
                            config=config.translation,
                            images_b64=bubble_images_b64,
                            full_image_b64=full_image_b64,
                            debug=verbose,
                            bubble_ids=bubble_ids,  # <<< ВАЖНО: новый аргумент
                        )
                    except RuntimeError as e:
                        if "Все Gemini ключи исчерпаны" in str(e):
                            log_message("🚨 Все ключи Gemini исчерпаны. Прекращаем выполнение скрипта.", always_print=True)
                            raise RuntimeError("Все Gemini ключи исчерпаны/запрещены. Скрипт остановлен.")
                        else:
                            raise
                    except Exception as e:
                        log_message(f"Error during API translation: {e}", always_print=True)
                        translations_by_id = {bid: "[Translation Error]" for bid in bubble_ids}

                    # 4) Проверяем на «пустые/ошибочные» ответы
                    error_strings = [
                        "[Gemini-OCR: No text/empty response]",
                        "[Gemini-Translate: No text/empty response]",
                        "[OCR FAILED]",
                        "[Translation Error]",
                        "[Gemini: OCR call failed/blocked]"
                    ]
                    values = list(translations_by_id.values()) if isinstance(translations_by_id, dict) else (translations_by_id or [])
                    if any(isinstance(v, str) and any(v.strip().startswith(es) for es in error_strings) for v in values):
                        if attempt < MAX_PAGE_RETRIES - 1:
                            log_message(
                                f"[Retry {attempt+1}] Gemini empty/failed response detected on {image_path}, retrying...",
                                always_print=True,
                            )
                            time.sleep(1)
                            continue
                        else:
                            log_message(
                                f"[Retry {attempt+1}] Max retries reached, proceeding with available translations...",
                                always_print=True,
                            )


                    # 5) Маппим переводы КАЖДОМУ баблу ПО ID (даже если перевод пустой)
                    for b in sorted_bubble_data:
                        t = translations_by_id.get(b["id"], "")
                        b["translation"] = t if isinstance(t, str) else ""

                    # 6) Рендерим только заполненные
                    for bubble in sorted_bubble_data:
                        text = bubble.get("translation", "")
                        if not text or text.startswith("[Translation Error]"):
                            continue

                        bbox = bubble["bbox"]

                        rendered_image, success = render_text_skia(
                            pil_image=pil_cleaned_image,
                            text=text,
                            bbox=bbox,
                            lir_bbox=None,
                            cleaned_mask=None,
                            bubble_color_bgr=(255,255,255),
                            font_dir=config.rendering.font_dir,
                            min_font_size=config.rendering.min_font_size,
                            max_font_size=config.rendering.max_font_size,
                            line_spacing_mult=config.rendering.line_spacing,
                            use_subpixel_rendering=config.rendering.use_subpixel_rendering,
                            font_hinting=config.rendering.font_hinting,
                            use_ligatures=config.rendering.use_ligatures,
                            verbose=verbose,
                        )
                        if success:
                            pil_cleaned_image = rendered_image
                            final_image_to_save = pil_cleaned_image

        # --- Save Output ---
        if output_path:
            if final_image_to_save.mode != target_mode:
                final_image_to_save = final_image_to_save.convert(target_mode)
            save_image_with_compression(
                final_image_to_save,
                output_path,
                jpeg_quality=config.output.jpeg_quality,
                png_compression=config.output.png_compression,
                verbose=verbose,
            )

        end_time = time.time()
        log_message(f"Processing finished in {end_time - start_time:.2f} seconds.", always_print=True)
        return final_image_to_save

    # если все попытки неудачны
    log_message(f"Page {image_path} failed after {MAX_PAGE_RETRIES} retries.", always_print=True)
    return pil_original

def natural_sort_key(path: Path):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', path.name)]

def batch_translate_images(
    input_dir: Union[str, Path],
    config: MangaTranslatorConfig,
    output_dir: Optional[Union[str, Path]] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Dict[str, Any]:
    """
    Process all images in a directory (recursively) using a configuration object.
    Files in each folder are processed fully before moving to the next folder.
    Skips already processed files inside output folder.
    """
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        log_message(f"Input path '{input_dir}' is not a directory", always_print=True)
        return {"success_count": 0, "error_count": 0, "errors": {}}

    if output_dir:
        output_dir = Path(output_dir)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path("./output") / timestamp

    os.makedirs(output_dir, exist_ok=True)

    image_extensions = [".jpg", ".jpeg", ".png", ".webp"]
    results = {"success_count": 0, "error_count": 0, "errors": {}}
    total_images = 0

    # Сначала считаем общее количество изображений (без output)
    def count_images(folder: Path) -> int:
        if folder.resolve() == output_dir.resolve():
            return 0
        count = sum(1 for f in folder.iterdir() if f.is_file() and f.suffix.lower() in image_extensions)
        for sub in folder.iterdir():
            if sub.is_dir() and sub.resolve() != output_dir.resolve():
                count += count_images(sub)
        return count

    total_images = count_images(input_dir)
    if total_images == 0:
        log_message(f"No image files found in '{input_dir}'", always_print=True)
        return results

    start_batch_time = time.time()
    log_message(f"Starting batch processing of {total_images} images...", always_print=True)
    if progress_callback:
        progress_callback(0.0, f"Starting batch processing of {total_images} images...")

    processed_count = 0

    # Основная рекурсивная функция обработки
    def process_folder(folder: Path):
        nonlocal processed_count
        if folder.resolve() == output_dir.resolve():
            return

        # Сначала файлы текущей папки
        files = sorted(
            [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in image_extensions],
            key=natural_sort_key
        )
        for img_path in files:
            try:
                if progress_callback:
                    progress_callback(processed_count / total_images,
                                      f"Processing image {processed_count + 1}/{total_images}: {img_path.name}")

                # Определяем расширение для сохранения
                original_ext = img_path.suffix.lower()
                desired_format = config.output.output_format
                if desired_format == "jpeg":
                    output_ext = ".jpg"
                elif desired_format == "png":
                    output_ext = ".png"
                elif desired_format == "auto":
                    output_ext = original_ext
                else:
                    output_ext = original_ext

                # Сохраняем структуру папок
                rel_path = img_path.relative_to(input_dir).parent
                target_dir = output_dir / rel_path
                target_dir.mkdir(parents=True, exist_ok=True)

                output_path = target_dir / f"{img_path.stem}{output_ext}"
                log_message(f"Processing {img_path}", always_print=True)

                translate_and_render(img_path, config, output_path)
                results["success_count"] += 1

            except Exception as e:
                log_message(f"Error processing {img_path}: {str(e)}", always_print=True)
                results["error_count"] += 1
                results["errors"][str(img_path)] = str(e)

            processed_count += 1
            if progress_callback:
                progress_callback(processed_count / total_images,
                                  f"Completed {processed_count}/{total_images} images")

        # Потом рекурсивно подпапки
        subdirs = sorted([d for d in folder.iterdir() if d.is_dir()], key=natural_sort_key)
        for subdir in subdirs:
            if subdir.resolve() != output_dir.resolve():
                process_folder(subdir)

    process_folder(input_dir)

    if progress_callback:
        progress_callback(1.0, "Processing complete")

    end_batch_time = time.time()
    total_batch_time = end_batch_time - start_batch_time
    seconds_per_image = total_batch_time / total_images if total_images > 0 else 0

    log_message(
        f"\nBatch processing complete: {results['success_count']} of {total_images} images processed "
        f"in {total_batch_time:.2f} seconds ({seconds_per_image:.2f} seconds/image).\n",
        always_print=True,
    )
    if results["error_count"] > 0:
        log_message(f"Failed to process {results['error_count']} images.", always_print=True)
        for filename, error_msg in results["errors"].items():
            log_message(f"  - {filename}: {error_msg}", always_print=True)

    return results

def _load_gemini_keys_from_env() -> list[str]:
    raw = os.environ.get("GEMINI_API_KEYS", "")
    return [k.strip() for k in re.split(r"[,\s]+", raw) if k.strip()]

def main():
    parser = argparse.ArgumentParser(description="Translate manga/comic speech bubbles using a configuration approach")
    parser.add_argument("--input", type=str, required=True, help="Path to input image or directory (if using --batch)")
    parser.add_argument(
        "--output", type=str, required=False, help="Path to save the translated image or directory (if using --batch)"
    )
    parser.add_argument("--batch", action="store_true", help="Process all images in the input directory")
    parser.add_argument("--detector-model", type=str, default="/content/aimangaautotranslator/models/comic-speech-bubble-detector.pt", help="YOLO model for detecting speech bubbles")
    parser.add_argument("--cleaner-model", type=str, default="/content/aimangaautotranslator/models/yolov8m_seg-speech-bubble.pt", help="YOLO model for cleaning bubbles (removing text)")
    # --- Provider and API Key Arguments ---
    parser.add_argument(
        "--provider",
        type=str,
        default="Gemini",
        choices=["Gemini", "OpenAI", "Anthropic", "OpenRouter", "OpenAI-Compatible"],
        help="LLM provider to use for translation",
    )
    parser.add_argument(
        "--gemini-api-key",
        type=str,
        default=None,
        help="Gemini API key (overrides GOOGLE_API_KEY env var if --provider is Gemini)",
    )
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default=None,
        help="OpenAI API key (overrides OPENAI_API_KEY env var if --provider is OpenAI)",
    )
    parser.add_argument(
        "--anthropic-api-key",
        type=str,
        default=None,
        help="Anthropic API key (overrides ANTHROPIC_API_KEY env var if --provider is Anthropic)",
    )
    parser.add_argument(
        "--openrouter-api-key",
        type=str,
        default=None,
        help="OpenRouter API key (overrides OPENROUTER_API_KEY env var if --provider is OpenRouter)",
    )
    parser.add_argument(
        "--openai-compatible-url",
        type=str,
        default="http://localhost:11434/v1",
        help="Base URL for the OpenAI-Compatible endpoint (e.g., http://localhost:11434/v1)",
    )
    parser.add_argument(
        "--openai-compatible-api-key",
        type=str,
        default=None,
        help="Optional API key for the OpenAI-Compatible endpoint (overrides OPENAI_COMPATIBLE_API_KEY env var)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model name for the selected provider (e.g., 'gemini-2.0-flash'). "
        "If not provided, a default will be attempted based on the provider.",
    )
    parser.add_argument("--font-dir", type=str, default="./fonts", help="Directory containing font files")
    parser.add_argument("--input-language", type=str, default="Japanese", help="Source language")
    parser.add_argument("--output-language", type=str, default="English", help="Target language")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold for detection")
    parser.add_argument(
        "--reading-direction",
        type=str,
        default="rtl",
        help="Reading direction for sorting bubbles (rtl or ltr)"
    )
    # Cleaning args
    parser.add_argument("--dilation-kernel-size", type=int, default=7, help="ROI Dilation Kernel Size")
    parser.add_argument("--dilation-iterations", type=int, default=1, help="ROI Dilation Iterations")
    parser.add_argument(
        "--use-otsu-threshold",
        action="store_true",
        help="Use Otsu's method for thresholding instead of the fixed value (210)",
    )
    parser.add_argument("--min-contour-area", type=int, default=50, help="Min Bubble Contour Area")
    parser.add_argument("--closing-kernel-size", type=int, default=7, help="Mask Closing Kernel Size")
    parser.add_argument("--closing-iterations", type=int, default=1, help="Mask Closing Iterations")
    parser.add_argument(
        "--constraint-erosion-kernel-size", type=int, default=9, help="Edge Constraint Erosion Kernel Size"
    )
    parser.add_argument(
        "--constraint-erosion-iterations", type=int, default=1, help="Edge Constraint Erosion Iterations"
    )
    # Translation args
    parser.add_argument("--temperature", type=float, default=0.1, help="Controls randomness in output (0.0-2.0)")
    parser.add_argument("--special-instructions", type=str, default=None, help="Add special directions to prompt")
    parser.add_argument("--top-p", type=float, default=0.95, help="Nucleus sampling parameter (0.0-1.0)")
    parser.add_argument("--top-k", type=int, default=1, help="Limits to top k tokens")
    parser.add_argument(
        "--translation-mode",
        type=str,
        default="two-step",
        help="Translation process mode (one-step or two-step)"
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default="medium",
        choices=["minimal", "low", "medium", "high"],
        help=(
            "Internal reasoning effort for OpenAI reasoning models (o1/o3/o4-mini/gpt-5*). "
            "Note: 'minimal' is only supported by gpt-5 series."
        ),
    )
    # Rendering args
    parser.add_argument("--max-font-size", type=int, default=14, help="Max font size for rendering text.")
    parser.add_argument("--min-font-size", type=int, default=8, help="Min font size for rendering text.")
    parser.add_argument("--line-spacing", type=float, default=1.0, help="Line spacing multiplier for rendered text.")
    # Output args 
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG compression quality (1-100)")
    parser.add_argument("--png-compression", type=int, default=6, help="PNG compression level (0-9)")
    parser.add_argument(
        "--image-mode", type=str, default="RGB", choices=["RGB", "RGBA"], help="Processing image mode (RGB or RGBA)"
    )
    # General args
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
    parser.add_argument(
        "--cleaning-only",
        action="store_true",
        help="Only perform detection and cleaning, skip translation and rendering.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable 'thinking' capabilities for Gemini 2.5 Flash models.",
    )
    parser.set_defaults(verbose=False, cpu=False, cleaning_only=False, enable_thinking=False)

    args = parser.parse_args()

    # --- Create Config Object ---
    provider = args.provider
    api_key = None
    api_key_arg_name = ""
    api_key_env_var = ""
    compatible_url = None

    if provider == "Gemini":
        api_key = args.gemini_api_key or os.environ.get("GOOGLE_API_KEY")
        api_key_arg_name = "--gemini-api-key"
        api_key_env_var = "GOOGLE_API_KEY"
        default_model = "gemini-2.0-flash"
    elif provider == "OpenAI":
        api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
        api_key_arg_name = "--openai-api-key"
        api_key_env_var = "OPENAI_API_KEY"
        default_model = "gpt-4o"
    elif provider == "Anthropic":
        api_key = args.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        api_key_arg_name = "--anthropic-api-key"
        api_key_env_var = "ANTHROPIC_API_KEY"
        default_model = "claude-3.7-sonnet-latest"
    elif provider == "OpenRouter":
        api_key = args.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
        api_key_arg_name = "--openrouter-api-key"
        api_key_env_var = "OPENROUTER_API_KEY"
        default_model = "openrouter/auto"
    elif provider == "OpenAI-Compatible":
        compatible_url = args.openai_compatible_url
        api_key = args.openai_compatible_api_key or os.environ.get("OPENAI_COMPATIBLE_API_KEY")
        api_key_arg_name = "--openai-compatible-api-key"
        api_key_env_var = "OPENAI_COMPATIBLE_API_KEY"
        default_model = "default"

    if provider != "OpenAI-Compatible" and not api_key:
        print(
            f"Warning: {provider} API key not provided via {api_key_arg_name} or {api_key_env_var} "
            f"environment variable. Translation will likely fail."
        )

    model_name = args.model_name or default_model
    if not args.model_name:
        print(f"Using default model for {provider}: {model_name}")

    target_device = torch.device("cpu") if args.cpu or not torch.cuda.is_available() else torch.device("cuda")
    print(f"Using {'CPU' if target_device.type == 'cpu' else 'CUDA'} device.")

    use_otsu_config_val = args.use_otsu_threshold

    config = MangaTranslatorConfig(
        detector_model_path=args.detector_model,
        cleaner_model_path=args.cleaner_model,
        verbose=args.verbose,
        device=target_device,
        cleaning_only=args.cleaning_only,
        detection=DetectionConfig(confidence=args.conf),
        cleaning=CleaningConfig(
            dilation_kernel_size=args.dilation_kernel_size,
            dilation_iterations=args.dilation_iterations,
            use_otsu_threshold=use_otsu_config_val,
            min_contour_area=args.min_contour_area,
            closing_kernel_size=args.closing_kernel_size,
            closing_iterations=args.closing_iterations,
            constraint_erosion_kernel_size=args.constraint_erosion_kernel_size,
            constraint_erosion_iterations=args.constraint_erosion_iterations,
        ),
        translation=TranslationConfig(
            provider=provider,
            gemini_api_keys=_load_gemini_keys_from_env() if provider == "Gemini" else None,
            openai_api_key=api_key if provider == "OpenAI" else os.environ.get("OPENAI_API_KEY", ""),
            anthropic_api_key=api_key if provider == "Anthropic" else os.environ.get("ANTHROPIC_API_KEY", ""),
            openrouter_api_key=api_key if provider == "OpenRouter" else os.environ.get("OPENROUTER_API_KEY", ""),
            openai_compatible_url=compatible_url,
            openai_compatible_api_key=(
                api_key if provider == "OpenAI-Compatible" else os.environ.get("OPENAI_COMPATIBLE_API_KEY", "")
            ),
            model_name=model_name,
            temperature=args.temperature,
            special_instructions=args.special_instructions,
            top_p=args.top_p,
            top_k=args.top_k,
            input_language=args.input_language,
            output_language=args.output_language,
            reading_direction=args.reading_direction,
            translation_mode=args.translation_mode,
            enable_thinking=args.enable_thinking,
            reasoning_effort=args.reasoning_effort,
        ),
        rendering=RenderingConfig(
            font_dir=args.font_dir,
            max_font_size=args.max_font_size,
            min_font_size=args.min_font_size,
            line_spacing=args.line_spacing,
        ),
        output=OutputConfig(
            jpeg_quality=args.jpeg_quality, png_compression=args.png_compression, image_mode=args.image_mode
        ),
    )

    # --- Execute --- 
    if args.batch:
        input_path = Path(args.input)
        if not input_path.is_dir():
            print(f"Error: --batch requires --input '{args.input}' to be a directory.")
            exit(1)

        output_dir = Path(args.output) if args.output else None

        if args.output:
            output_dir = Path(args.output)
            if not output_dir.exists():
                print(f"Creating output directory: {output_dir}")
                output_dir.mkdir(parents=True, exist_ok=True)
            elif not output_dir.is_dir():
                print(f"Error: Specified --output '{output_dir}' is not a directory.")
                exit(1)

        batch_translate_images(input_path, config, output_dir)
    else:
        input_path = Path(args.input)
        if not input_path.is_file():
            print(f"Error: Input '{args.input}' is not a valid file.")
            exit(1)

        output_path_arg = args.output
        if not output_path_arg:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            original_ext = input_path.suffix.lower()
            output_ext = original_ext
            output_dir = Path("./output")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"MangaTranslator_{timestamp}{output_ext}"
            print(f"--output not specified, using default: {output_path}")
        else:
            output_path = Path(output_path_arg)
            if not output_path.parent.exists():
                print(f"Creating directory for output file: {output_path.parent}")
                output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            log_message(f"Processing {input_path}...", always_print=True)
            translate_and_render(input_path, config, output_path)
            log_message(f"Translation complete. Result saved to {output_path}", always_print=True)
        except Exception as e:
            log_message(f"Error processing {input_path}: {e}", always_print=True)


if __name__ == "__main__":
    main()
