import re
import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import skia
import uharfbuzz as hb
from PIL import Image

from utils.logging import log_message

# Cache loaded font data
_font_data_cache = {}
_typeface_cache = {}
_hb_face_cache = {}

# Cache font features
_font_features_cache = {}

# Cache font variants
_font_variants_cache: Dict[str, Dict[str, Optional[Path]]] = {}
FONT_KEYWORDS = {
    "bold": {"bold", "heavy", "black"},
    "italic": {"italic", "oblique", "slanted", "inclined"},
    "regular": {"regular", "normal", "roman", "medium"},
}

# --- Style Parsing Helper ---
STYLE_PATTERN = re.compile(r"(\*{1,3})(.*?)(\1)")  # Matches *text*, **text**, ***text***


def get_font_features(font_path: str) -> Dict[str, List[str]]:
    """
    Uses fontTools to list GSUB and GPOS features in a font file. Caches results.

    Args:
        font_path (str): Path to the font file.

    Returns:
        Dict[str, List[str]]: Dictionary with 'GSUB' and 'GPOS' keys,
                              each containing a list of feature tags.
    """
    if font_path in _font_features_cache:
        return _font_features_cache[font_path]

    features = {"GSUB": [], "GPOS": []}
    try:
        from fontTools.ttLib import TTFont

        font = TTFont(font_path, fontNumber=0)

        if "GSUB" in font and hasattr(font["GSUB"].table, "FeatureList") and font["GSUB"].table.FeatureList:
            features["GSUB"] = sorted([fr.FeatureTag for fr in font["GSUB"].table.FeatureList.FeatureRecord])

        if "GPOS" in font and hasattr(font["GPOS"].table, "FeatureList") and font["GPOS"].table.FeatureList:
            features["GPOS"] = sorted([fr.FeatureTag for fr in font["GPOS"].table.FeatureList.FeatureRecord])

    except ImportError:
        log_message("fontTools not installed, cannot inspect font features.", always_print=True)
    except Exception as e:
        log_message(f"Could not inspect font features for {os.path.basename(font_path)}: {e}", always_print=True)

    _font_features_cache[font_path] = features
    return features


def _find_font_variants(font_dir: str, verbose: bool = False) -> Dict[str, Optional[Path]]:
    """
    Finds regular, italic, bold, and bold-italic font variants (.ttf, .otf)
    in a directory based on filename keywords. Caches results per directory.

    Args:
        font_dir (str): Directory containing font files.
        verbose (bool): Whether to print detailed logs.

    Returns:
        Dict[str, Optional[Path]]: Dictionary mapping style names
                                   ("regular", "italic", "bold", "bold_italic")
                                   to their respective Path objects, or None if not found.
    """
    resolved_dir = str(Path(font_dir).resolve())
    if resolved_dir in _font_variants_cache:
        return _font_variants_cache[resolved_dir]

    log_message(f"Scanning font directory: {resolved_dir}", verbose=verbose)
    font_files: List[Path] = []
    font_variants: Dict[str, Optional[Path]] = {"regular": None, "italic": None, "bold": None, "bold_italic": None}
    identified_files: set[Path] = set()

    try:
        font_dir_path = Path(resolved_dir)
        if font_dir_path.exists() and font_dir_path.is_dir():
            font_files = list(font_dir_path.glob("*.ttf")) + list(font_dir_path.glob("*.otf"))
        else:
            log_message(f"Font directory '{font_dir_path}' does not exist or is not a directory.", always_print=True)
            _font_variants_cache[resolved_dir] = font_variants
            return font_variants
    except Exception as e:
        log_message(f"Error accessing font directory '{font_dir}': {e}", always_print=True)
        _font_variants_cache[resolved_dir] = font_variants
        return font_variants

    if not font_files:
        log_message(f"No font files (.ttf, .otf) found in '{resolved_dir}'", always_print=True)
        _font_variants_cache[resolved_dir] = font_variants
        return font_variants

    # Sort by name length (desc) to potentially prioritize more specific names like "BoldItalic" over "Bold"
    font_files.sort(key=lambda x: len(x.name), reverse=True)

    # --- Font File Detection (Multi-pass) ---
    # Pass 1: Exact matches for combined styles first
    for font_file in font_files:
        if font_file in identified_files:
            continue
        stem_lower = font_file.stem.lower()
        is_bold = any(kw in stem_lower for kw in FONT_KEYWORDS["bold"])
        is_italic = any(kw in stem_lower for kw in FONT_KEYWORDS["italic"])
        assigned = False
        if is_bold and is_italic:
            if not font_variants["bold_italic"]:
                font_variants["bold_italic"] = font_file
                assigned = True
                log_message(f"Found Bold Italic: {font_file.name}", verbose=verbose)
        if assigned:
            identified_files.add(font_file)

    # Pass 2: Exact matches for single styles
    for font_file in font_files:
        if font_file in identified_files:
            continue
        stem_lower = font_file.stem.lower()
        is_bold = any(kw in stem_lower for kw in FONT_KEYWORDS["bold"])
        is_italic = any(kw in stem_lower for kw in FONT_KEYWORDS["italic"])
        assigned = False
        if is_bold and not is_italic:  # Only bold
            if not font_variants["bold"]:
                font_variants["bold"] = font_file
                assigned = True
                log_message(f"Found Bold: {font_file.name}", verbose=verbose)
        elif is_italic and not is_bold:  # Only italic
            if not font_variants["italic"]:
                font_variants["italic"] = font_file
                assigned = True
                log_message(f"Found Italic: {font_file.name}", verbose=verbose)
        if assigned:
            identified_files.add(font_file)

    # Pass 3: Explicit regular matches
    for font_file in font_files:
        if font_file in identified_files:
            continue
        stem_lower = font_file.stem.lower()
        is_regular = any(kw in stem_lower for kw in FONT_KEYWORDS["regular"])
        is_bold = any(kw in stem_lower for kw in FONT_KEYWORDS["bold"])
        is_italic = any(kw in stem_lower for kw in FONT_KEYWORDS["italic"])
        assigned = False
        if is_regular and not is_bold and not is_italic:
            if not font_variants["regular"]:
                font_variants["regular"] = font_file
                assigned = True
                log_message(f"Found Regular (explicit): {font_file.name}", verbose=verbose)
        if assigned:
            identified_files.add(font_file)

    # Pass 4: Infer regular (no style keywords found)
    if not font_variants["regular"]:
        for font_file in font_files:
            if font_file in identified_files:
                continue
            stem_lower = font_file.stem.lower()
            is_bold = any(kw in stem_lower for kw in FONT_KEYWORDS["bold"])
            is_italic = any(kw in stem_lower for kw in FONT_KEYWORDS["italic"])
            if not is_bold and not is_italic and not any(kw in stem_lower for kw in FONT_KEYWORDS["regular"]):
                font_name_lower = font_file.name.lower()
                is_likely_specific = any(
                    spec in font_name_lower
                    for spec in [
                        "light",
                        "thin",
                        "condensed",
                        "expanded",
                        "semi",
                        "demi",
                        "extra",
                        "ultra",
                        "book",
                        "medium",
                        "black",
                        "heavy",
                    ]
                )
                if not is_likely_specific:
                    font_variants["regular"] = font_file
                    identified_files.add(font_file)
                    log_message(f"Inferred Regular (no keywords): {font_file.name}", verbose=verbose)
                    break

    # Pass 5: Fallback regular (use first available unidentified font)
    if not font_variants["regular"]:
        first_available = next((f for f in font_files if f not in identified_files), None)
        if first_available:
            font_variants["regular"] = first_available
            if first_available not in identified_files:
                identified_files.add(first_available)
            log_message(f"Fallback Regular (first unidentified): {first_available.name}", verbose=verbose)

    # Pass 6: Final fallback (use *any* identified font if regular is still missing)
    if not font_variants["regular"]:
        backup_regular = next(
            (
                f
                for f in [font_variants.get("bold"), font_variants.get("italic"), font_variants.get("bold_italic")]
                if f
            ),
            None,
        )
        if backup_regular:
            font_variants["regular"] = backup_regular
            log_message(f"Fallback Regular (using existing variant): {backup_regular.name}", verbose=verbose)
        elif font_files:  # Absolute last resort: just grab the first font file found
            font_variants["regular"] = font_files[0]
            log_message(f"Fallback Regular (absolute first file): {font_files[0].name}", verbose=verbose)

    if not font_variants["regular"]:
        log_message(
            (
                f"CRITICAL: Could not identify or fallback to any regular font file in '{resolved_dir}'. "
                "Rendering will likely fail."
            ),
            always_print=True,
        )
        # Still cache the (lack of) results
    else:
        log_message(f"Final Font Variants Found in {resolved_dir}:", verbose=verbose)
        for style, path in font_variants.items():
            log_message(f"  - {style}: {path.name if path else 'None'}", verbose=verbose)

    _font_variants_cache[resolved_dir] = font_variants
    return font_variants


def _parse_styled_segments(text: str) -> List[Tuple[str, str]]:
    """
    Parses text with markdown-like style markers into segments.

    Args:
        text (str): Input text potentially containing ***bold italic***, **bold**, *italic*.

    Returns:
        List[Tuple[str, str]]: List of (segment_text, style_name) tuples.
                               style_name is one of "regular", "italic", "bold", "bold_italic".
    """
    segments = []
    last_end = 0
    for match in STYLE_PATTERN.finditer(text):
        start, end = match.span()
        marker = match.group(1)
        content = match.group(2)

        # Add preceding regular text
        if start > last_end:
            segments.append((text[last_end:start], "regular"))

        style = "regular"
        if len(marker) == 3:
            style = "bold_italic"
        elif len(marker) == 2:
            style = "bold"
        elif len(marker) == 1:
            style = "italic"

        segments.append((content, style))
        last_end = end

    if last_end < len(text):
        segments.append((text[last_end:], "regular"))

    return [(txt, style) for txt, style in segments if txt]


# --- Skia/HarfBuzz Rendering ---
def _load_font_resources(font_path: str) -> Tuple[Optional[bytes], Optional[skia.Typeface], Optional[hb.Face]]:
    """Loads font data, Skia Typeface, and HarfBuzz Face, using caching."""
    if font_path not in _font_data_cache:
        try:
            with open(font_path, "rb") as f:
                _font_data_cache[font_path] = f.read()
        except Exception as e:
            log_message(f"ERROR: Failed to read font file {font_path}: {e}", always_print=True)
            return None, None, None
    font_data = _font_data_cache[font_path]

    if font_path not in _typeface_cache:
        skia_data = skia.Data.MakeWithoutCopy(font_data)
        _typeface_cache[font_path] = skia.Typeface.MakeFromData(skia_data)
        if _typeface_cache[font_path] is None:
            log_message(f"ERROR: Skia could not load typeface from {font_path}", always_print=True)
            del _typeface_cache[font_path]
            del _font_data_cache[font_path]
            return None, None, None
    typeface = _typeface_cache[font_path]

    if font_path not in _hb_face_cache:
        try:
            _hb_face_cache[font_path] = hb.Face(font_data)
        except Exception as e:
            log_message(f"ERROR: HarfBuzz could not load face from {font_path}: {e}", always_print=True)
            if font_path in _typeface_cache:
                del _typeface_cache[font_path]
            if font_path in _font_data_cache:
                del _font_data_cache[font_path]
            return None, None, None
    hb_face = _hb_face_cache[font_path]

    return font_data, typeface, hb_face


def _shape_line(
    text_line: str, hb_font: hb.Font, features: Dict[str, bool]
) -> Tuple[List[hb.GlyphInfo], List[hb.GlyphPosition]]:
    """Shapes a line of text with HarfBuzz."""
    hb_buffer = hb.Buffer()
    hb_buffer.add_str(text_line)
    hb_buffer.guess_segment_properties()
    try:
        hb.shape(hb_font, hb_buffer, features)
        return hb_buffer.glyph_infos, hb_buffer.glyph_positions
    except Exception as e:
        log_message(f"ERROR: HarfBuzz shaping failed for line '{text_line[:30]}...': {e}", always_print=True)
        return [], []


def _calculate_line_width(positions: List[hb.GlyphPosition], scale_factor: float) -> float:
    """
    Calculates the width of a shaped line from HarfBuzz positions.
    Assumes positions are in 26.6 fixed point if hb_font.ptem was set before shaping.
    """
    if not positions:
        return 0.0
    # Sum advances (which are in 26.6 fixed point)
    total_advance_fixed = sum(pos.x_advance for pos in positions)
    # Convert from 26.6 fixed point to pixels by dividing by 64.0
    HB_26_6_SCALE_FACTOR = 64.0
    return float(total_advance_fixed / HB_26_6_SCALE_FACTOR)


def _pil_to_skia_surface(pil_image: Image.Image) -> Optional[skia.Surface]:
    """Converts a PIL image to a Skia Surface."""
    try:
        if pil_image.mode != "RGBA":
            pil_image = pil_image.convert("RGBA")
        skia_image = skia.Image.frombytes(pil_image.tobytes(), pil_image.size, skia.kRGBA_8888_ColorType)
        if skia_image is None:
            log_message("Failed to create Skia image from PIL bytes", always_print=True)
            return None
        surface = skia.Surface(pil_image.width, pil_image.height)
        with surface as canvas:
            canvas.drawImage(skia_image, 0, 0)
        return surface
    except Exception as e:
        log_message(f"Error converting PIL to Skia Surface: {e}", always_print=True)
        return None


def _skia_surface_to_pil(surface: skia.Surface) -> Optional[Image.Image]:
    """Converts a Skia Surface back to a PIL image."""
    try:
        skia_image: Optional[skia.Image] = surface.makeImageSnapshot()
        if skia_image is None:
            log_message("Failed to create Skia snapshot from surface", always_print=True)
            return None

        skia_image = skia_image.convert(alphaType=skia.kUnpremul_AlphaType, colorType=skia.kRGBA_8888_ColorType)
        pil_image = Image.fromarray(skia_image)
        return pil_image
    except Exception as e:
        log_message(f"Error converting Skia Surface to PIL: {e}", always_print=True)
        return None

def _split_token_to_fit(token: str, hb_font: hb.Font, features: Dict[str, bool], max_w: float) -> List[str]:
    """
    Делит один "кирпич" на части, чтобы каждая часть влезала в max_w при текущем шрифте.
    Сначала пытаемся делить по естественным разделителям, затем посимвольно.
    Возвращает список кусочков (порядок сохранён).
    """
    # 1) если и так влезает — возвращаем как есть
    _, p = _shape_line(token, hb_font, features)
    if _calculate_line_width(p, 1.0) <= max_w:
        return [token]

    # 2) пробуем мягкие разбиения по разделителям
    for sep in ["-", "–", "—", "/", "\\", "·", "•", ":", ".", "|", "_"]:
        if sep in token:
            parts = []
            for chunk in token.split(sep):
                if not chunk:
                    continue
                # рекурсивно проверим каждый чанк (вдруг всё ещё длинный)
                sub = _split_token_to_fit(chunk, hb_font, features, max_w)
                parts.extend(sub + [sep])  # возвращаем разделитель как отдельный токен
            if parts and parts[-1] in ["-", "–", "—", "/", "\\", "·", "•", ":", ".", "|", "_"]:
                parts.pop()  # убрать хвостовой разделитель
            if parts:
                return parts

    # 3) посимвольное разбиение 
    pieces, cur = [], ""
    for ch in token:
        test = cur + ch
        _, p = _shape_line(test, hb_font, features)
        if _calculate_line_width(p, 1.0) <= max_w or not cur:
            cur = test
        else:
            pieces.append(cur)
            cur = ch
    if cur:
        pieces.append(cur)
    return pieces

VOWELS = set("aeiouаеёиоуыэюяAEIOUАЕЁИОУЫЭЮЯ")  # латиница + кириллица

def _apply_hyphenation_rules(text: str, min_len: int = 12) -> str:
    """
    Обрабатывает блок текста по правилам:
    - если слово >= min_len символов → перенос;
    - если есть дефис и слово длинное → перенос после дефиса;
    - иначе перенос после ближайшей к середине гласной.
    """

    def split_word(word: str) -> str:
        # короткие не трогаем
        if len(word) < min_len:
            return word

        # если есть дефис — переносим после него
        if "-" in word:
            parts = word.split("-", 1)
            return parts[0] + "-\n" + parts[1]

        # иначе ищем гласную, ближайшую к середине
        mid = len(word) // 2
        closest_idx = None
        min_dist = len(word)
        for i, ch in enumerate(word):
            if ch in VOWELS:
                dist = abs(i - mid)
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i
        if closest_idx is not None:
            return word[: closest_idx + 1] + "-\n" + word[closest_idx + 1 :]
        else:
            # fallback: режем просто по середине
            return word[:mid] + "-\n" + word[mid:]

    # разбиваем по пробельным разделителям, сохраняя их
    tokens = re.split(r"(\s+)", text)
    processed = [split_word(tok) if tok.strip() else tok for tok in tokens]
    return "".join(processed)

# --- Helper for LIR Padding --- final_font_size
def _calculate_lir_padding_for_flat_edges(
    cleaned_mask: np.ndarray, lir_bbox: List[int], padding_percentage: float = 0.05, flatness_threshold: float = 0.80
) -> Tuple[float, float, float, float]:
    """
    Calculates padding ratios for LIR edges based on how "flat" they are against the mask background.

    Args:
        cleaned_mask: The binary (0/255) mask of the cleaned bubble.
        lir_bbox: The LIR coordinates [x, y, w, h].
        padding_percentage: The amount of padding to apply (as a ratio of LIR dimension) if an edge is flat.
        flatness_threshold: The ratio of edge points adjacent to background required to consider an edge flat.

    Returns:
        Tuple[float, float, float, float]: Padding ratios for (left, right, top, bottom).
    """
    pad_left, pad_right, pad_top, pad_bottom = 0.0, 0.0, 0.0, 0.0
    if cleaned_mask is None or lir_bbox is None or len(lir_bbox) != 4:
        return pad_left, pad_right, pad_top, pad_bottom

    lir_x, lir_y, lir_w, lir_h = map(int, lir_bbox)
    mask_h, mask_w = cleaned_mask.shape

    if lir_w <= 0 or lir_h <= 0:
        return pad_left, pad_right, pad_top, pad_bottom

    # Distance to check overlap (pixels)
    check_offset = 2

    # --- Check Left Edge ---
    x = lir_x
    if x >= check_offset:
        total_edge_points = 0
        flat_edge_points = 0
        nx = x - check_offset
        for y in range(max(0, lir_y), min(mask_h, lir_y + lir_h)):
            if cleaned_mask[y, x] == 255:
                total_edge_points += 1
                if cleaned_mask[y, nx] == 0:
                    flat_edge_points += 1
        if total_edge_points > 0:
            ratio = flat_edge_points / total_edge_points
            if ratio >= flatness_threshold:
                pad_left = padding_percentage

    # --- Check Right Edge ---
    x = lir_x + lir_w - 1
    if x < mask_w - check_offset:
        total_edge_points = 0
        flat_edge_points = 0
        nx = x + check_offset
        for y in range(max(0, lir_y), min(mask_h, lir_y + lir_h)):
            if cleaned_mask[y, x] == 255:
                total_edge_points += 1
                if cleaned_mask[y, nx] == 0:
                    flat_edge_points += 1
        if total_edge_points > 0:
            ratio = flat_edge_points / total_edge_points
            if ratio >= flatness_threshold:
                pad_right = padding_percentage

    return pad_left, pad_right, pad_top, pad_bottom


def render_text_skia(
    pil_image: Image.Image,
    text: str,
    bbox: Tuple[int, int, int, int],
    font_dir: str,
    lir_bbox: Optional[List[int]] = None,
    cleaned_mask: Optional[np.ndarray] = None,
    bubble_color_bgr: Optional[Tuple[int, int, int]] = (255, 255, 255),
    min_font_size: int = 8,
    max_font_size: int = 14,
    line_spacing_mult: float = 1.0,
    scale_factor_env: float = float(os.environ.get("FONT_SCALE", "1.01")),
    use_subpixel_rendering: bool = False,
    font_hinting: str = "none",
    use_ligatures: bool = False,
    verbose: bool = False,
) -> Tuple[Optional[Image.Image], bool]:
    """
    Fits and renders text within a bounding box using Skia and HarfBuzz.

    Args:
        pil_image: PIL Image object to draw onto.
        text: Text to render.
        bbox: Bounding box coordinates (x1, y1, x2, y2).
        font_path: Path to the font file.
        bubble_color_bgr: Background color of the bubble (BGR tuple). Used to determine text color. Defaults to white.
        min_font_size: Minimum font size to try.
        max_font_size: Starting font size to try.
        line_spacing_mult: Multiplier for line height (based on font metrics).
        use_subpixel_rendering: Whether to use subpixel anti-aliasing.
        font_hinting: Font hinting level ("none", "slight", "normal", "full").
        use_ligatures: Whether to enable standard ligatures ('liga').
        verbose: Whether to print detailed logs.

    Returns:
        Tuple containing:
        - Modified PIL Image object (or original if failed).
        - Boolean indicating success.
    """
    # --- Use original bbox for wrapping constraints --- 
    x1, y1, x2, y2 = bbox
    bubble_width = x2 - x1
    bubble_height = y2 - y1

    if bubble_width <= 0 or bubble_height <= 0:
        log_message(f"Invalid original bbox dimensions: {bbox}", always_print=True)
        return pil_image, False

    clean_text = " ".join(text.split())
    if not clean_text:
        return pil_image, True

    clean_text = _apply_hyphenation_rules(clean_text, min_len=12)

    # --- Determine Rendering Boundaries and Target Center ---
    use_lir = False
    max_render_width = 0.0
    max_render_height = 0.0
    target_center_x = 0.0
    target_center_y = 0.0

    if lir_bbox is not None and len(lir_bbox) == 4 and lir_bbox[2] > 0 and lir_bbox[3] > 0:
        orig_lir_x, orig_lir_y, orig_lir_w, orig_lir_h = lir_bbox

        # --- Calculate Padding for Flat Edges ---
        pad_left, pad_right, pad_top, pad_bottom = 0.0, 0.0, 0.0, 0.0
        if cleaned_mask is not None and orig_lir_w > 0 and orig_lir_h > 0:
            pad_left, pad_right, pad_top, pad_bottom = _calculate_lir_padding_for_flat_edges(cleaned_mask, lir_bbox)
            if any(p > 0 for p in [pad_left, pad_right, pad_top, pad_bottom]):
                log_message(
                    (f"  Applying LIR padding for flat edges: L={pad_left:.1%}, R={pad_right:.1%}, "
                     f"T={pad_top:.1%}, B={pad_bottom:.1%}"),
                    verbose=verbose,
                )

        # --- Adjust LIR based on padding ---
        lir_x = orig_lir_x + orig_lir_w * pad_left
        lir_y = orig_lir_y + orig_lir_h * pad_top
        lir_w = orig_lir_w * (1.0 - pad_left - pad_right)
        lir_h = orig_lir_h * (1.0 - pad_top - pad_bottom)

        if lir_w <= 0 or lir_h <= 0:
            log_message(
                (f"Warning: LIR padding resulted in non-positive dimensions ({lir_w}x{lir_h}). "
                 f"Reverting to original LIR."),
                verbose=verbose,
                always_print=True,
            )
            lir_x, lir_y, lir_w, lir_h = orig_lir_x, orig_lir_y, orig_lir_w, orig_lir_h

        log_message(
            (f"Using LIR (potentially padded): x={lir_x:.1f}, y={lir_y:.1f}, w={lir_w:.1f}, h={lir_h:.1f}. "
             f"Calculating expanded boundaries."),
            verbose=verbose,
        )

        # --- LIR Expansion Parameters (Tunable) ---
        base_expand_factor = 0.15  # Base expansion factor (applied to width/height)
        tall_threshold = 1.1  # H/W ratio above which we add extra width expansion
        wide_threshold = 1.1  # W/H ratio above which we add extra height expansion
        ratio_scaling_factor = 0.10  # How much extra expansion per unit of ratio above threshold

        # --- Calculate Aspect Ratios ---
        height_width_ratio = lir_h / lir_w if lir_w > 0 else float("inf")
        width_height_ratio = lir_w / lir_h if lir_h > 0 else float("inf")

        # --- Calculate Base Expansion (Applied Always) ---
        base_expand_x = (base_expand_factor * lir_w) / 2.0
        base_expand_y = (base_expand_factor * lir_h) / 2.0

        # --- Calculate Additional Expansion (Based on Ratio) ---
        additional_expand_x = 0.0
        additional_expand_y = 0.0
        if height_width_ratio > tall_threshold:
            additional_expand_x = (ratio_scaling_factor * (height_width_ratio - tall_threshold) * lir_w) / 2.0
            log_message(
                (
                    f"  LIR is tall (H/W={height_width_ratio:.2f} > {tall_threshold}). "
                    f"Adding extra width expansion: {additional_expand_x * 2:.1f}px"
                ),
                verbose=verbose,
            )
        elif width_height_ratio > wide_threshold:
            additional_expand_y = (ratio_scaling_factor * (width_height_ratio - wide_threshold) * lir_h) / 2.0
            log_message(
                (
                    f"  LIR is wide (W/H={width_height_ratio:.2f} > {wide_threshold}). "
                    f"Adding extra height expansion: {additional_expand_y * 2:.1f}px"
                ),
                verbose=verbose,
            )
        else:
            log_message("  LIR aspect ratio within thresholds. Using base expansion only.", verbose=verbose)

        # --- Calculate Total Expansion ---
        total_expand_x = base_expand_x + additional_expand_x
        total_expand_y = base_expand_y + additional_expand_y
        log_message(f"  Total expansion: X={total_expand_x * 2:.1f}px, Y={total_expand_y * 2:.1f}px", verbose=verbose)

        # --- Calculate Expanded Boundaries ---
        expanded_x1 = lir_x - total_expand_x
        expanded_y1 = lir_y - total_expand_y
        expanded_x2 = lir_x + lir_w + total_expand_x
        expanded_y2 = lir_y + lir_h + total_expand_y

        # --- Constrain Expansion by Original bbox ---
        final_render_x1 = max(expanded_x1, float(x1))
        final_render_y1 = max(expanded_y1, float(y1))
        final_render_x2 = min(expanded_x2, float(x2))
        final_render_y2 = min(expanded_y2, float(y2))
        log_message(
            f"  Expanded LIR: ({expanded_x1:.1f}, {expanded_y1:.1f}, {expanded_x2:.1f}, {expanded_y2:.1f})",
            verbose=verbose,
        )
        log_message(f"  Capped by bbox: ({x1}, {y1}, {x2}, {y2})", verbose=verbose)
        log_message(
            (
                f"  Final boundaries: ({final_render_x1:.1f}, {final_render_y1:.1f}, "
                f"{final_render_x2:.1f}, {final_render_y2:.1f})"
            ),
            verbose=verbose,
        )

        # --- Set Final Rendering Constraints ---
        max_render_width = final_render_x2 - final_render_x1
        max_render_height = final_render_y2 - final_render_y1

        # --- Guard: если LIR слишком мелкий по сравнению с исходным bbox, откатываемся к padded bbox ---
        min_dim_ratio = 0.35  # 35% от размеров исходного bbox
        if (max_render_width < bubble_width * min_dim_ratio) or (max_render_height < bubble_height * min_dim_ratio):
            padding_ratio = 0.08
            max_render_width = bubble_width * (1 - 2 * padding_ratio)
            max_render_height = bubble_height * (1 - 2 * padding_ratio)
            max_render_width = max(1.0, float(max_render_width))
            max_render_height = max(1.0, float(max_render_height))
            target_center_x = x1 + bubble_width / 2.0
            target_center_y = y1 + bubble_height / 2.0
            use_lir = False
        else:
            # как и было
            target_center_x = final_render_x1 + max_render_width / 2.0
            target_center_y = final_render_y1 + max_render_height / 2.0
            use_lir = True

        if max_render_width <= 0 or max_render_height <= 0:
            log_message(
                (
                    f"Warning: Expanded LIR resulted in non-positive dimensions after capping "
                    f"({max_render_width:.1f}x{max_render_height:.1f}). Falling back to bbox."
                ),
                verbose=verbose,
                always_print=True,
            )
            # Fallback logic (same as the 'else' block below)
            padding_ratio = 0.05  # Default padding
            max_render_width = bubble_width * (1 - 2 * padding_ratio)
            max_render_height = bubble_height * (1 - 2 * padding_ratio)
            if max_render_width <= 0 or max_render_height <= 0:
                max_render_width = max(1.0, float(bubble_width))
                max_render_height = max(1.0, float(bubble_height))
            target_center_x = x1 + bubble_width / 2.0
            target_center_y = y1 + bubble_height / 2.0
            use_lir = False
        else:
            # --- Set Target Center Point (Center of the final rendering area) ---
            target_center_x = final_render_x1 + max_render_width / 2.0
            target_center_y = final_render_y1 + max_render_height / 2.0
            use_lir = True
            log_message(
                (
                    f"Using Expanded LIR: Max W={max_render_width:.1f}, Max H={max_render_height:.1f}, "
                    f"Center=({target_center_x:.1f}, {target_center_y:.1f})"
                ),
                verbose=verbose,
            )

    else:
        # --- Fallback to Original Bbox with Padding ---
        if lir_bbox:
            log_message(
                f"Invalid LIR received: {lir_bbox}. Falling back to original bbox with padding.", verbose=verbose
            )
        else:
            log_message("LIR not provided. Falling back to original bbox with padding.", verbose=verbose)

        padding_ratio = 0.08  # Adjust padding as needed
        max_render_width = bubble_width * (1 - 2 * padding_ratio)
        max_render_height = bubble_height * (1 - 2 * padding_ratio)

        if max_render_width <= 0 or max_render_height <= 0:
            log_message(f"Original bbox too small for padding: w={bubble_width}, h={bubble_height}", verbose=verbose)
            max_render_width = max(1.0, float(bubble_width))  # Use base width/height if padding makes it too small
            max_render_height = max(1.0, float(bubble_height))

        target_center_x = x1 + bubble_width / 2.0
        target_center_y = y1 + bubble_height / 2.0
        use_lir = False
        log_message(
            (
                f"Using Fallback (Padded Bbox): Max W={max_render_width:.1f}, Max H={max_render_height:.1f}, "
                f"Center=({target_center_x:.1f}, {target_center_y:.1f})"
            ),
            verbose=verbose,
        )

    # --- Find and Load Font Variants ---
    font_variants = _find_font_variants(font_dir, verbose=verbose)
    regular_font_path = font_variants.get("regular")

    if not regular_font_path:
        log_message(f"CRITICAL: Regular font variant not found in '{font_dir}'. Cannot render text.", always_print=True)
        return pil_image, False

    _, regular_typeface, regular_hb_face = _load_font_resources(str(regular_font_path))

    if not regular_typeface or not regular_hb_face:
        log_message(f"CRITICAL: Failed to load regular font resources for {regular_font_path.name}.", always_print=True)
        return pil_image, False

    # --- Determine HarfBuzz Features ---
    available_features = get_font_features(str(regular_font_path))
    features_to_enable = {
        # Enable kerning if available (GPOS 'kern' or legacy 'kern' table)
        "kern": "kern" in available_features["GPOS"],
        # Enable ligatures based on config and availability
        "liga": use_ligatures and "liga" in available_features["GSUB"],
        # Add other features as needed, e.g., "dlig", "calt"
        "calt": "calt" in available_features["GSUB"],
    }
    log_message(f"HarfBuzz features to enable: {features_to_enable}", verbose=verbose)

    # --- Font Size Iteration --- 
    best_fit_size = -1
    best_fit_lines_data = None
    best_fit_metrics = None
    best_fit_max_line_width = float("inf")

    current_size = max_font_size
    while current_size >= min_font_size:
        log_message(f"Trying font size: {current_size}", verbose=verbose)
        # Use REGULAR font face for size fitting calculations
        hb_font = hb.Font(regular_hb_face)
        hb_font.ptem = float(current_size)

        # Scale factor to convert font units to pixels
        scale_factor = 1.0
        if regular_hb_face.upem > 0:
            scale_factor = current_size / regular_hb_face.upem
        else:
            log_message(
                f"Warning: Regular font {regular_font_path.name} has upem=0. Using scale factor 1.0.", verbose=verbose
            )

        # Convert scale factor to 16.16 fixed-point integer for HarfBuzz
        hb_scale = int(scale_factor * (2**16))
        hb_font.scale = (hb_scale, hb_scale)

        skia_font_test = skia.Font(regular_typeface, current_size)
        try:
            metrics = skia_font_test.getMetrics()
            single_line_height = (-metrics.fAscent + metrics.fDescent + metrics.fLeading) * line_spacing_mult
            if single_line_height <= 0:
                single_line_height = current_size * 1.2 * line_spacing_mult
        except Exception as e:
            log_message(f"Could not get font metrics at size {current_size}: {e}", verbose=verbose)
            single_line_height = current_size * 1.2 * line_spacing_mult

        # --- Text Wrapping at current_size ---
        text_for_measurement = STYLE_PATTERN.sub(r"\2", clean_text)
        words = text_for_measurement.split()
        original_words = clean_text.split()
        word_map = {STYLE_PATTERN.sub(r"\2", w): w for w in original_words}

        wrapped_lines_text = []
        current_line_stripped_words = []
        longest_word_width_at_size = 0

        for word in words:
            _, w_positions = _shape_line(word, hb_font, features_to_enable)
            word_width = _calculate_line_width(w_positions, 1.0)
            longest_word_width_at_size = max(longest_word_width_at_size, word_width)

            if word_width > max_render_width:
                if current_size > min_font_size:
                    original_word = word_map.get(word, word)
                    log_message(
                        (f"Size {current_size}: Word '{original_word}' ({word_width:.1f}px) wider than max width "
                         f"({max_render_width:.1f}px). Trying smaller size."),
                        verbose=verbose,
                    )
                    current_line_stripped_words = None
                    break

                # emergency splitting into pieces
                pieces = _split_token_to_fit(word, hb_font, features_to_enable, max_render_width)
                for piece in pieces:
                    test_line = (current_line_stripped_words or []) + [piece]
                    _, test_positions = _shape_line(" ".join(test_line), hb_font, features_to_enable)
                    tentative_width = _calculate_line_width(test_positions, 1.0)
                    if tentative_width <= max_render_width:
                        current_line_stripped_words = test_line
                        # Если кусок кончается дефисом — это означает, что мы хотим
                        # обязательный перенос после него: завершаем текущую строку.
                        if piece.endswith("-"):
                            original_line_words = [word_map.get(w, w) for w in current_line_stripped_words]
                            wrapped_lines_text.append(" ".join(original_line_words))
                            current_line_stripped_words = []
                    else:
                        if current_line_stripped_words:
                            original_line_words = [word_map.get(w, w) for w in current_line_stripped_words]
                            wrapped_lines_text.append(" ".join(original_line_words))
                        current_line_stripped_words = [piece]
                        # если этот new piece сам оканчивается дефисом — сразу переносим
                        if piece.endswith("-"):
                            original_line_words = [word_map.get(w, w) for w in current_line_stripped_words]
                            wrapped_lines_text.append(" ".join(original_line_words))
                            current_line_stripped_words = []
                continue

            # обычный путь — слово влезает целиком
            test_line_stripped_words = (current_line_stripped_words or []) + [word]
            test_line_stripped_text = " ".join(test_line_stripped_words)
            _, test_positions = _shape_line(test_line_stripped_text, hb_font, features_to_enable)
            tentative_width = _calculate_line_width(test_positions, 1.0)

            if tentative_width <= max_render_width:
                current_line_stripped_words = test_line_stripped_words
                # если слово оканчивается дефисом (наш вставленный перенос) —
                # обязательно завершаем строку здесь
                if word.endswith("-"):
                    original_line_words = [word_map.get(w, w) for w in current_line_stripped_words]
                    wrapped_lines_text.append(" ".join(original_line_words))
                    current_line_stripped_words = []
            else:
                if current_line_stripped_words:
                    original_line_words = [word_map.get(w, w) for w in current_line_stripped_words]
                    wrapped_lines_text.append(" ".join(original_line_words))
                current_line_stripped_words = [word]
                # и если он-закончился дефисом — переносим сразу
                if word.endswith("-"):
                    original_line_words = [word_map.get(w, w) for w in current_line_stripped_words]
                    wrapped_lines_text.append(" ".join(original_line_words))
                    current_line_stripped_words = []

        if current_line_stripped_words is None:
            current_size -= 1
            continue

        if current_line_stripped_words:
            original_line_words = [word_map.get(w, w) for w in current_line_stripped_words]
            wrapped_lines_text.append(" ".join(original_line_words))

        if not wrapped_lines_text:
            log_message(f"Size {current_size}: Wrapping failed, no lines generated.", verbose=verbose)
            current_size -= 1
            continue

        # --- Measure Wrapped Block at current_size ---
        current_max_line_width = 0
        lines_data_at_size = []
        for line_text_with_markers in wrapped_lines_text:
            line_text_stripped = STYLE_PATTERN.sub(r"\2", line_text_with_markers)
            infos, positions = _shape_line(line_text_stripped, hb_font, features_to_enable)
            width = _calculate_line_width(positions, 1.0)
            lines_data_at_size.append({"text_with_markers": line_text_with_markers, "width": width})
            current_max_line_width = max(current_max_line_width, width)

        total_block_height = (-metrics.fAscent + metrics.fDescent) + (len(wrapped_lines_text) - 1) * single_line_height

        log_message(
            (
                f"Size {current_size}: Block W={current_max_line_width:.1f} (Max W={max_render_width:.1f}), "
                f"H={total_block_height:.1f} (Max H={max_render_height:.1f})"
            ),
            verbose=verbose,
        )

        # --- Check Fit ---
        if current_max_line_width <= max_render_width and total_block_height <= max_render_height:
            log_message(f"Size {current_size} fits!", verbose=verbose)
            best_fit_size = current_size
            best_fit_lines_data = lines_data_at_size
            best_fit_metrics = metrics
            best_fit_max_line_width = current_max_line_width
            break

        current_size -= 1

    # --- Check if any size fit ---
    if best_fit_size == -1:
        log_message(
            f"Could not fit text in bubble even at min size {min_font_size}: '{clean_text[:30]}...'", always_print=True
        )
        return pil_image, False

    # --- Prepare for Rendering ---
    final_font_size = int(best_fit_size * scale_factor_env)
    final_lines_data = best_fit_lines_data  # сами строки не меняются

    # Пересчитываем метрики под новый размер
    skia_font_fix = skia.Font(regular_typeface, final_font_size)
    final_metrics = skia_font_fix.getMetrics()

    # Высота строки с учётом line_spacing_mult
    final_line_height = (-final_metrics.fAscent + final_metrics.fDescent + final_metrics.fLeading) * line_spacing_mult
    if final_line_height <= 0:
        final_line_height = final_font_size * 1.2 * line_spacing_mult

    # Пересчёт максимальной ширины строки
    new_max_line_width = 0
    for line_data in final_lines_data:
        line_text_stripped = STYLE_PATTERN.sub(r"\2", line_data["text_with_markers"])
        infos, positions = _shape_line(line_text_stripped, hb.Font(regular_hb_face), features_to_enable)
        width = _calculate_line_width(positions, 1.0) * (final_font_size / best_fit_size)
        new_max_line_width = max(new_max_line_width, width)

    final_max_line_width = new_max_line_width

    # --- Load Needed Font Resources for Rendering ---
    # Determine which styles are actually present in the text
    required_styles = {"regular"} | {style for _, style in _parse_styled_segments(clean_text)}
    log_message(f"Required font styles: {required_styles}", verbose=verbose)

    # Initialize dictionaries to hold loaded resources, starting with regular
    loaded_typefaces: Dict[str, Optional[skia.Typeface]] = {"regular": regular_typeface}
    loaded_hb_faces: Dict[str, Optional[hb.Face]] = {"regular": regular_hb_face}

    for style in ["italic", "bold", "bold_italic"]:
        if style in required_styles:
            font_path = font_variants.get(style)
            if font_path:
                log_message(f"Loading {style} font variant: {font_path.name}", verbose=verbose)
                _, typeface, hb_face = _load_font_resources(str(font_path))
                if typeface and hb_face:
                    loaded_typefaces[style] = typeface
                    loaded_hb_faces[style] = hb_face
                else:
                    log_message(
                        f"Warning: Failed to load {style} font variant {font_path.name}. Will fallback to regular.",
                        verbose=verbose,
                    )
            else:
                log_message(
                    (
                        f"Warning: {style} style requested by text, but font variant not found in {font_dir}. "
                        "Will fallback to regular."
                    ),
                    verbose=verbose,
                )

    surface = _pil_to_skia_surface(pil_image)
    if surface is None:
        log_message("Failed to create Skia surface for rendering.", always_print=True)
        return pil_image, False

    # --- Skia Font Hinting Setup (Used per segment) ---
    hinting_map = {
        "none": skia.FontHinting.kNone,
        "slight": skia.FontHinting.kSlight,
        "normal": skia.FontHinting.kNormal,
        "full": skia.FontHinting.kFull,
    }
    skia_hinting = hinting_map.get(font_hinting.lower(), skia.FontHinting.kNone)

    # --- Determine Text Color ---
    text_color = skia.ColorBLACK

    # --- Настройка обводки (Stroke) ---
    import os
    # Коэффициент толщины обводки берём из переменной окружения STROKE_SCALE (по умолчанию 0.8)
    stroke_scale = float(os.environ.get("STROKE_SCALE", "0.8"))

    # Вычисляем итоговую толщину обводки на основе размера шрифта
    stroke_width = max(1.0, final_font_size * stroke_scale)

    stroke_paint = skia.Paint(
        AntiAlias=True,
        Color=skia.ColorWHITE,
        Style=skia.Paint.kStroke_Style,  # стиль линии
        StrokeWidth=stroke_width,        # толщина обводки
    )

    # --- Настройка заливки текста ---
    fill_paint = skia.Paint(
        AntiAlias=True,
        Color=text_color,
        Style=skia.Paint.kFill_Style  # сразу задаём стиль
    )

    # --- Calculate Starting Position (Centering the visual block around the target center) ---
    block_start_x = target_center_x - final_max_line_width / 2.0

    # Vertical centering
    num_lines = len(final_lines_data)
    if num_lines > 0:
        visual_block_height_span = (num_lines - 1) * final_line_height - final_metrics.fAscent
        first_baseline_y = target_center_y - visual_block_height_span / 2.0 - final_metrics.fAscent
    else:
        first_baseline_y = target_center_y - final_metrics.fAscent / 2.0

    log_message(
        (
            f"Rendering at size {final_font_size}. Centering target ({'LIR' if use_lir else 'BBox'}): "
            f"({target_center_x:.1f}, {target_center_y:.1f}). Block Start X: {block_start_x:.1f}. "
            f"First Baseline Y: {first_baseline_y:.1f}"
        ),
        verbose=verbose,
    )

    # --- Render whole block as a single TextBlob (stroke applied to the whole block) ---
    with surface as canvas:
        # Один общий билдер для всего блока (всех строк и сегментов)
        block_builder = skia.TextBlobBuilder()
        current_baseline_y = first_baseline_y

        # Будем сохранять данные для логирования и подстраховки (на случай ошибки сборки)
        per_segment_runs = []  # каждый элемент: (skia_font_segment, glyph_ids, skia_point_positions, segment_width, log_info)

        for i, line_data in enumerate(final_lines_data):
            line_text_with_markers = line_data["text_with_markers"]
            line_width_measured = line_data["width"]  # Width measured using regular font

            # Horizontal alignment for this specific line, centered within the block's max width
            line_start_x = block_start_x + (final_max_line_width - line_width_measured) / 2.0
            cursor_x = line_start_x

            segments = _parse_styled_segments(line_text_with_markers)
            log_message(f"Line {i} Segments: {segments}", verbose=verbose)

            for segment_text, style_name in segments:
                if not segment_text:
                    continue

                # --- Select Font Resources for Segment ---
                typeface_to_use = None
                hb_face_to_use = None
                fallback_style_used = None

                if style_name == "bold_italic":
                    typeface_to_use = loaded_typefaces.get("bold_italic")
                    hb_face_to_use = loaded_hb_faces.get("bold_italic")
                    if not typeface_to_use or not hb_face_to_use:
                        fallback_style_used = "bold"
                        typeface_to_use = loaded_typefaces.get("bold")
                        hb_face_to_use = loaded_hb_faces.get("bold")
                    if not typeface_to_use or not hb_face_to_use:
                        fallback_style_used = "italic"
                        typeface_to_use = loaded_typefaces.get("italic")
                        hb_face_to_use = loaded_hb_faces.get("italic")
                    if not typeface_to_use or not hb_face_to_use:
                        fallback_style_used = "regular"
                        typeface_to_use = regular_typeface
                        hb_face_to_use = regular_hb_face
                elif style_name == "bold":
                    typeface_to_use = loaded_typefaces.get("bold")
                    hb_face_to_use = loaded_hb_faces.get("bold")
                    if not typeface_to_use or not hb_face_to_use:
                        fallback_style_used = "regular"
                        typeface_to_use = regular_typeface
                        hb_face_to_use = regular_hb_face
                elif style_name == "italic":
                    typeface_to_use = loaded_typefaces.get("italic")
                    hb_face_to_use = loaded_hb_faces.get("italic")
                    if not typeface_to_use or not hb_face_to_use:
                        fallback_style_used = "regular"
                        typeface_to_use = regular_typeface
                        hb_face_to_use = regular_hb_face
                else:  # Regular or unknown style
                    typeface_to_use = regular_typeface
                    hb_face_to_use = regular_hb_face

                if fallback_style_used:
                    log_message(
                        f"  Style '{style_name}' not found, falling back to '{fallback_style_used}'.", verbose=verbose
                    )

                if not typeface_to_use or not hb_face_to_use:
                    log_message(
                        (
                            f"ERROR: Could not get any valid font resources (including regular) "
                            f"for style '{style_name}'. Skipping segment."
                        ),
                        always_print=True,
                    )
                    continue

                # --- Setup Skia Font for Segment ---
                skia_font_segment = skia.Font(typeface_to_use, final_font_size)
                skia_font_segment.setSubpixel(use_subpixel_rendering)
                skia_font_segment.setHinting(skia_hinting)

                # --- Setup HarfBuzz Font for Segment ---
                hb_font_segment = hb.Font(hb_face_to_use)
                hb_font_segment.ptem = float(final_font_size)
                if hb_face_to_use.upem > 0:
                    scale_factor = final_font_size / hb_face_to_use.upem
                    hb_scale = int(scale_factor * (2**16))
                    hb_font_segment.scale = (hb_scale, hb_scale)
                else:
                    hb_font_segment.scale = (int(final_font_size * (2**16)), int(final_font_size * (2**16)))

                # --- Shape Segment ---
                try:
                    infos, positions = _shape_line(segment_text, hb_font_segment, features_to_enable)
                    if not infos:
                        log_message(
                            f"Warning: HarfBuzz returned no glyphs for segment '{segment_text}'", verbose=verbose
                        )
                        continue
                except Exception as e:
                    log_message(f"ERROR: HarfBuzz shaping failed for segment '{segment_text}': {e}", always_print=True)
                    continue

                # --- Build positions for this segment (absolute coordinates) ---
                glyph_ids = [info.codepoint for info in infos]
                skia_point_positions = []
                segment_cursor_x = 0
                HB_26_6_SCALE_FACTOR = 64.0

                for _, pos in zip(infos, positions):
                    glyph_x = cursor_x + segment_cursor_x + (pos.x_offset / HB_26_6_SCALE_FACTOR)
                    glyph_y = current_baseline_y - (pos.y_offset / HB_26_6_SCALE_FACTOR)
                    skia_point_positions.append(skia.Point(glyph_x, glyph_y))
                    segment_cursor_x += pos.x_advance / HB_26_6_SCALE_FACTOR

                # --- Не рисуем сразу, а добавляем run в общий буфер или сохраняем для резервного варианта ---
                per_segment_runs.append((skia_font_segment, glyph_ids, skia_point_positions, segment_cursor_x, (segment_text, style_name)))
                cursor_x += segment_cursor_x

            # Перейти к следующей базовой линии (строке)
            current_baseline_y += final_line_height

        # --- Попробуем собрать единый TextBlob и отрисовать единожды ---
        try:
            # Добавляем в block_builder все runs
            for sk_font, glyph_ids, positions, seg_width, seg_info in per_segment_runs:
                # allocRunPos автоматически добавляет run в один буфер
                _ = block_builder.allocRunPos(sk_font, glyph_ids, positions)

            block_text_blob = block_builder.make()
            if block_text_blob:
                # Draw stroke once for весь блок, затем заливка
                canvas.drawTextBlob(block_text_blob, 0, 0, stroke_paint)
                canvas.drawTextBlob(block_text_blob, 0, 0, fill_paint)
                log_message(f"Rendered entire block as single TextBlob with {len(per_segment_runs)} runs.", verbose=verbose)
            else:
                raise RuntimeError("block_builder.make() returned None")

        except Exception as e:
            # В случае ошибки сборки TextBlob — логируем и откатываемся к по-сегментной отрисовке
            log_message(f"Warning: Failed to build/draw combined TextBlob ({e}). Falling back to per-segment rendering.", verbose=verbose, always_print=True)
            # Рисуем сегмент за сегментом (как было раньше) — stroke+fill
            for sk_font, glyph_ids, positions, seg_width, seg_info in per_segment_runs:
                try:
                    fallback_builder = skia.TextBlobBuilder()
                    _ = fallback_builder.allocRunPos(sk_font, glyph_ids, positions)
                    tb = fallback_builder.make()
                    if tb:
                        canvas.drawTextBlob(tb, 0, 0, stroke_paint)
                        canvas.drawTextBlob(tb, 0, 0, fill_paint)
                except Exception as e2:
                    log_message(f"ERROR: Fallback per-segment draw failed for segment {seg_info}: {e2}", always_print=True)

    # --- Convert back to PIL --- 
    final_pil_image = _skia_surface_to_pil(surface)
    if final_pil_image is None:
        log_message("Failed to convert final Skia surface back to PIL.", always_print=True)
        return pil_image, False

    log_message(f"Successfully rendered text at size {final_font_size}", verbose=verbose)
    return final_pil_image, True
