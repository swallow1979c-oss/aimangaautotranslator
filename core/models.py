from dataclasses import dataclass, field
from typing import Optional
import torch
import cv2


@dataclass
class DetectionConfig:
    """Configuration for speech bubble detection."""
    confidence: float = 0.35


@dataclass
class CleaningConfig:
    """Configuration for speech bubble cleaning."""
    dilation_kernel_size: int = 7
    dilation_iterations: int = 1
    use_otsu_threshold: bool = False
    min_contour_area: int = 50
    closing_kernel_size: int = 7
    closing_iterations: int = 1
    closing_kernel_shape: int = cv2.MORPH_ELLIPSE
    constraint_erosion_kernel_size: int = 9
    constraint_erosion_iterations: int = 1


@dataclass
class TranslationConfig:
    """Configuration for text translation with Gemini API key rotation."""

    provider: str = "Gemini"
    gemini_api_keys: list[str] = field(default_factory=list)
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    openrouter_api_key: str = ""
    openai_compatible_url: str = "http://localhost:11434/v1"
    openai_compatible_api_key: Optional[str] = ""
    model_name: str = "gemini-2.5-flash"
    provider_models: dict[str, Optional[str]] = field(default_factory=dict)
    temperature: float = 0.1
    special_instructions: Optional[str] = None
    top_p: float = 0.95
    top_k: int = 64
    input_language: str = "Japanese"
    output_language: str = "English"
    reading_direction: str = "rtl"
    translation_mode: str = "one-step"
    enable_thinking: bool = True
    reasoning_effort: Optional[str] = "medium"

    # --- Новое поле для отслеживания текущего ключа Gemini ---
    current_key_index: int = 0

    def get_next_gemini_key(self) -> str:
        """Возвращает текущий Gemini API ключ и сдвигает индекс на следующий."""
        if not self.gemini_api_keys:
            raise ValueError("Gemini API key(s) missing in TranslationConfig.")
        
        key = self.gemini_api_keys[self.current_key_index]
        self.current_key_index = (self.current_key_index + 1) % len(self.gemini_api_keys)
        return key


@dataclass
class RenderingConfig:
    """Configuration for rendering translated text."""
    font_dir: str = "./fonts"
    max_font_size: int = 14
    min_font_size: int = 8
    line_spacing: float = 1.0
    use_subpixel_rendering: bool = False
    font_hinting: str = "none"
    use_ligatures: bool = False


@dataclass
class OutputConfig:
    """Configuration for saving output images."""
    jpeg_quality: int = 80
    png_compression: int = 6
    image_mode: str = "RGB"
    output_format: str = "jpeg"


@dataclass
class MangaTranslatorConfig:
    """Main configuration for the MangaTranslator pipeline."""
    detector_model_path: str
    cleaner_model_path: str
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    rendering: RenderingConfig = field(default_factory=RenderingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    verbose: bool = False
    device: Optional[torch.device] = None
    cleaning_only: bool = False

    def __post_init__(self):
        # Load API keys from environment variables if not already set
        import os

        # --- Исправлено для Gemini ---
        if self.translation.provider == "Gemini" and not self.translation.gemini_api_keys:
            env_key = os.environ.get("GOOGLE_API_KEY")
            if env_key:
                self.translation.gemini_api_keys = [env_key]

        # OpenAI
        if not self.translation.openai_api_key:
            self.translation.openai_api_key = os.environ.get("OPENAI_API_KEY", "")

        # Anthropic
        if not self.translation.anthropic_api_key:
            self.translation.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", "")

        # OpenRouter
        if not self.translation.openrouter_api_key:
            self.translation.openrouter_api_key = os.environ.get("OPENROUTER_API_KEY", "")

        # OpenAI-Compatible
        if not self.translation.openai_compatible_api_key:
            self.translation.openai_compatible_api_key = os.environ.get("OPENAI_COMPATIBLE_API_KEY", "")

        # Autodetect device if not specified
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
