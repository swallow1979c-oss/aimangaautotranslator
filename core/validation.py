from pathlib import Path
from typing import List, Tuple, Optional

from core.models import TranslationConfig, RenderingConfig
from utils.exceptions import ValidationError


def get_available_yolo_models(models_directory: Path) -> List[str]:
    """Helper to get list of available YOLO models."""
    if not models_directory.is_dir():
        return []
    model_files = list(models_directory.glob("*.pt"))
    models = [p.name for p in model_files]
    models.sort()
    return models


def validate_core_inputs(
    translation_cfg: TranslationConfig,
    rendering_cfg: RenderingConfig,
    selected_detector_model_name: Optional[str],
    selected_cleaner_model_name: Optional[str],
    models_dir: Path,
    fonts_base_dir: Path,
) -> Tuple[Path, Path, Path]:
    """
    Validates core inputs required for translation, raising standard exceptions.

    Returns:
        (detector_model_path, cleaner_model_path, font_dir_path)
    """

    # --- YOLO Models Validation ---
    if not models_dir.is_dir():
        raise FileNotFoundError(f"YOLO models directory not found: {models_dir}")

    available_models = get_available_yolo_models(models_dir)
    if not available_models:
        raise ValidationError(f"No YOLO models (*.pt) found in {models_dir}")

    # Detector (bbox)
    if not selected_detector_model_name:
        raise ValidationError("No detector YOLO model selected.")
    if selected_detector_model_name not in available_models:
        raise ValidationError(
            f"Detector model '{selected_detector_model_name}' is not available. "
            f"Available: {', '.join(available_models)}"
        )
    detector_model_path = models_dir / selected_detector_model_name

    # Cleaner (segmentation)
    if not selected_cleaner_model_name:
        raise ValidationError("No cleaner YOLO model selected.")
    if selected_cleaner_model_name not in available_models:
        raise ValidationError(
            f"Cleaner model '{selected_cleaner_model_name}' is not available. "
            f"Available: {', '.join(available_models)}"
        )
    cleaner_model_path = models_dir / selected_cleaner_model_name

    # --- Font Validation ---
    if not fonts_base_dir.is_dir():
        raise FileNotFoundError(f"Fonts base directory not found: {fonts_base_dir}")

    if not rendering_cfg.font_dir:
        raise ValidationError("Font pack (font_dir in rendering config) not specified.")

    font_dir_path = fonts_base_dir / rendering_cfg.font_dir
    if not font_dir_path.is_dir():
        raise FileNotFoundError(
            f"Specified font pack directory '{rendering_cfg.font_dir}' not found within {fonts_base_dir}"
        )

    font_files = list(font_dir_path.glob("*.ttf")) + list(font_dir_path.glob("*.otf"))
    if not font_files:
        raise ValidationError(f"No font files (.ttf or .otf) found in the font pack directory: '{font_dir_path}'")

    # --- Rendering Config Validation ---
    if not (isinstance(rendering_cfg.max_font_size, int) and rendering_cfg.max_font_size > 0):
        raise ValueError("Max Font Size must be a positive integer.")
    if not (isinstance(rendering_cfg.min_font_size, int) and rendering_cfg.min_font_size > 0):
        raise ValueError("Min Font Size must be a positive integer.")
    if not (isinstance(rendering_cfg.line_spacing, (int, float)) and float(rendering_cfg.line_spacing) > 0):
        raise ValueError("Line Spacing must be a positive number.")
    if rendering_cfg.min_font_size > rendering_cfg.max_font_size:
        raise ValueError("Min Font Size cannot be larger than Max Font Size.")
    if rendering_cfg.font_hinting not in ["none", "slight", "normal", "full"]:
        raise ValueError("Invalid Font Hinting value. Must be one of: none, slight, normal, full.")

    # --- Translation Config Validation ---
    if not translation_cfg.provider:
        raise ValueError("Translation provider cannot be empty.")
    if not translation_cfg.model_name:
        raise ValueError("Translation model name cannot be empty.")
    if not translation_cfg.input_language:
        raise ValueError("Input language cannot be empty.")
    if not translation_cfg.output_language:
        raise ValueError("Output language cannot be empty.")
    if translation_cfg.reading_direction not in ["rtl", "ltr"]:
        raise ValueError("Reading direction must be 'rtl' or 'ltr'.")

    return detector_model_path.resolve(), cleaner_model_path.resolve(), font_dir_path.resolve()
