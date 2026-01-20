"""
AS66 downsampling utilities.

AS66 raw frames are 64×64 where each semantic tile is a uniform 4×4 block.
We reduce to 16×16 by averaging each 4×4 block (robust to small edge variations)
and rounding to the nearest integer code.

Text mode: never expose colors; only integers.
Visual mode: you may render PNGs, but do not leak numbers into text prompts.
"""

from __future__ import annotations
from typing import Iterable, List, Callable, Dict, Sequence, TYPE_CHECKING
import io
import base64
from PIL import Image, ImageDraw, ImageFont

if TYPE_CHECKING:
    from lucidgym.utils.representation import RepresentationConfig


Number = float | int


def frame_to_grid_text(frame: Sequence[Sequence[Sequence[int]]]) -> str:
    """Convert a 2D matrix (array of array of array) into ASCII text."""
    palette = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. "
    palette_len = len(palette)
    max_val = 16
    lines: list[str] = []

    for row in frame[0]:
        cells: list[str] = []
        for pixel in row:
            idx = int((pixel / max_val) * (palette_len - 1))
            cells.append(palette[idx])
        lines.append("".join(cells))
    return "\n".join(lines)


def flatten_frame(frame: Sequence[Sequence[Sequence[int]]]) -> list[int]:
    """Flatten a 2D RGB frame into a 1D list."""
    flat: list[int] = []
    for row in frame:
        for pixel in row:
            flat.extend(int(v) for v in pixel)
    return flat


def _mean(vals: Iterable[Number]) -> float:
    vs = list(vals)
    return (sum(vs) / float(len(vs))) if vs else 0.0


def downsample_blocks(
    grid: List[List[int]],
    block_h: int = 4,
    block_w: int = 4,
    reducer: Callable[[Iterable[Number]], float] = _mean,
    *,
    round_to_int: bool = True,
) -> List[List[int | float]]:
    if not grid or not grid[0]:
        return []
    H, W = len(grid), len(grid[0])
    out_h = (H + block_h - 1) // block_h
    out_w = (W + block_w - 1) // block_w
    out: List[List[int | float]] = [[0 for _ in range(out_w)] for _ in range(out_h)]
    for by in range(out_h):
        y0, y1 = by * block_h, min(H, (by + 1) * block_h)
        for bx in range(out_w):
            x0, x1 = bx * block_w, min(W, (bx + 1) * block_w)
            acc: list[Number] = []
            for y in range(y0, y1):
                acc.extend(grid[y][x0:x1])
            val = reducer(acc)
            out[by][bx] = int(round(val)) if round_to_int else val
    return out


def downsample_4x4(
    frame_3d: List[List[List[int]]] | None,
    *,
    take_last_grid: bool = True,
    round_to_int: bool = True,
) -> List[List[int]]:
    """
    Select one 2D grid from the 3D frame list, then 4×4-average → 16×16.
    """
    if not frame_3d:
        return []
    grid = frame_3d[-1] if take_last_grid else frame_3d[0]
    if not grid or not grid[0]:
        return []
    # type: ignore[return-value]
    return downsample_blocks(grid, 4, 4, _mean, round_to_int=round_to_int)


def generate_numeric_grid_image_bytes(grid: List[List[int]]) -> bytes:
    """
    Generates a PNG 'screenshot' of the 16x16 grid with numbers and headers.
    """
    cell_size = 24
    header_size = 24
    grid_size = 16
    img_size_w = (grid_size * cell_size) + header_size
    img_size_h = (grid_size * cell_size) + header_size

    bg_color = "#FFFFFF"
    line_color = "#000000"
    text_color = "#000000"
    header_bg = "#EEEEEE"

    img = Image.new("RGB", (img_size_w, img_size_h), bg_color)
    draw = ImageDraw.Draw(img)
    
    try:
        # Use a basic, widely available font. Fallback to default if needed.
        try:
            # Try a common truetype font
            font = ImageFont.truetype("Arial.ttf", 10)
        except IOError:
            # Fallback to default bitmap font
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    # Draw headers
    draw.rectangle([0, 0, img_size_w, header_size], fill=header_bg)
    draw.rectangle([0, 0, header_size, img_size_h], fill=header_bg)
    for i in range(grid_size):
        # Column headers
        x_center = header_size + (i * cell_size) + (cell_size // 2)
        y_center = header_size // 2
        draw.text((x_center, y_center), str(i), fill=text_color, font=font, anchor="mm")
        # Row headers
        x_center = header_size // 2
        y_center = header_size + (i * cell_size) + (cell_size // 2)
        draw.text((x_center, y_center), str(i), fill=text_color, font=font, anchor="mm")

    # Draw grid lines and numbers
    for y in range(grid_size):
        for x in range(grid_size):
            x0 = header_size + (x * cell_size)
            y0 = header_size + (y * cell_size)
            x1 = x0 + cell_size
            y1 = y0 + cell_size
            
            # Draw cell border
            draw.rectangle([x0, y0, x1, y1], outline=line_color)
            
            # Draw number
            if y < len(grid) and x < len(grid[y]):
                num_str = str(grid[y][x])
                draw.text((x0 + cell_size // 2, y0 + cell_size // 2), num_str, fill=text_color, font=font, anchor="mm")

    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return buf.getvalue()

KEY_COLORS = {
    0: "#FFFFFF", 1: "#CCCCCC", 2: "#999999",
    3: "#666666", 4: "#000000", 5: "#202020",
    6: "#1E93FF", 7: "#F93C31", 8: "#FF851B",
    9: "#921231", 10: "#88D8F1", 11: "#FFDC00",
    12: "#FF7BCC", 13: "#4FCC30", 14: "#2ECC71",
    15: "#7F3FBF",
}

def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.strip().lstrip("#")
    if len(h) != 6:
        return (136, 136, 136)
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))

def render_grid_to_png_bytes(grid: List[List[int]], cell: int = 22) -> bytes:
    """
    Generates a color PNG from a grid (e.g., 16x16 or 64x64).
    
    Args:
        grid: The 2D integer grid.
        cell: The pixel size (width and height) for each grid cell.
    """
    h = len(grid)
    w = len(grid[0]) if h > 0 else 0
    if h == 0 or w == 0:
        # Return a 1x1 black pixel as a fallback
        im = Image.new("RGB", (1, 1), (0, 0, 0))
        buf = io.BytesIO()
        im.save(buf, "PNG", optimize=True)
        return buf.getvalue()

    H, W = h * cell, w * cell
    im = Image.new("RGB", (W, H), (0, 0, 0))
    px = im.load()
    for y in range(h):
        row = grid[y]
        for x in range(w):
            code = row[x]
            rgb = _hex_to_rgb(KEY_COLORS.get(int(code) & 15, "#888888"))
            for dy in range(cell):
                for dx in range(cell):
                    px[x * cell + dx, y * cell + dy] = rgb
    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=True)
    return buf.getvalue()

# def downsample_grid(grid):
#     """
#     Downsamples a 64x64 grid to 16x16 by taking every 4th element 
#     from every 4th row.
#     """
#     # grid[::4] selects every 4th row (indices 0, 4, 8...)
#     # row[::4] selects every 4th item in that row
#     return [row[::4] for row in grid[::4]]


# import matplotlib.pyplot as plt
# import numpy as np

# def save_grid_visualization(grid_data, filename='grid_output.png'):
#     """
#     Converts a 2D list into a visualized image.
#     """
#     # Convert list of lists to a numpy array
#     matrix = np.array(grid_data)


def matrix16_to_lines(grid: List[List[int]]) -> str:
    """
    Convert a 16x16 (or any size) grid of integers to a text representation.

    Each row is formatted as space-separated integers with row indices.

    Args:
        grid: 2D list of integers

    Returns:
        String representation of the grid
    """
    if not grid:
        return "(empty grid)"

    lines = []
    for row_idx, row in enumerate(grid):
        row_str = " ".join(f"{int(v):2d}" for v in row)
        lines.append(f"[{row_idx:2d}] {row_str}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Configurable format functions
# ---------------------------------------------------------------------------


def format_grid(grid: List[List[int]], config: "RepresentationConfig") -> str:
    """
    Format a grid according to the representation config.

    Args:
        grid: 2D list of integers (0-15)
        config: RepresentationConfig specifying format

    Returns:
        Formatted string representation of the grid
    """
    from lucidgym.utils.representation import GridFormat

    if not grid:
        return "(empty grid)"

    match config.format:
        case GridFormat.INTEGERS_SPACED:
            return format_integers_spaced(grid, include_indices=config.include_row_indices)
        case GridFormat.INTEGERS_COMPACT:
            return format_integers_compact(grid)
        case GridFormat.INTEGERS_TUPLE:
            return format_integers_tuple(grid)
        case GridFormat.HEX:
            return format_hex(grid)
        case GridFormat.SYMBOLIC:
            return format_symbolic(grid, config.symbolic_map)
        case GridFormat.ASCII:
            return format_ascii(grid)
        case _:
            # Fallback to spaced integers
            return format_integers_spaced(grid, include_indices=True)


def format_integers_spaced(grid: List[List[int]], include_indices: bool = True) -> str:
    """
    Format grid as space-separated integers with optional row indices.

    This is the current/legacy format for backward compatibility.
    Token cost: ~745 tokens for 16x16 grid.

    Example output (with indices):
        [ 0]  1  2  3  4  5 ...
        [ 1]  1  2  3  4  5 ...

    Example output (without indices):
         1  2  3  4  5 ...
         1  2  3  4  5 ...
    """
    if not grid:
        return "(empty grid)"

    lines = []
    for row_idx, row in enumerate(grid):
        row_str = " ".join(f"{int(v):2d}" for v in row)
        if include_indices:
            lines.append(f"[{row_idx:2d}] {row_str}")
        else:
            lines.append(row_str)

    return "\n".join(lines)


def format_integers_compact(grid: List[List[int]]) -> str:
    """
    Format grid as space-separated integers without row indices.

    More token-efficient than INTEGERS_SPACED.
    Token cost: ~400 tokens for 16x16 grid.

    Example: "1 15 15 4 0 0 15 15 15 1 1 1 1 1 1 1"
    """
    if not grid:
        return "(empty grid)"

    lines = []
    for row in grid:
        row_str = " ".join(str(int(v)) for v in row)
        lines.append(row_str)

    return "\n".join(lines)


def format_integers_tuple(grid: List[List[int]]) -> str:
    """
    Format grid as tuple-like rows with parentheses and commas.

    Most verbose/native format but clear structure.
    Token cost: ~500 tokens for 16x16 grid.

    Example: "(1, 15, 15, 4, 0, 0, 15, 15, 15, 1, 1, 1, 1, 1, 1, 1)"
    """
    if not grid:
        return "(empty grid)"

    lines = []
    for row in grid:
        row_str = "(" + ", ".join(str(int(v)) for v in row) + ")"
        lines.append(row_str)

    return "\n".join(lines)


def format_hex(grid: List[List[int]]) -> str:
    """
    Format grid as hexadecimal characters (0-9, a-f).

    Most token-efficient format: ~95 tokens for 16x16 grid (87% savings).
    Each cell is exactly one character.

    Example: "0123456789abcdef"
    Where: 0-9 = 0-9, a=10, b=11, c=12, d=13, e=14, f=15
    """
    if not grid:
        return "(empty grid)"

    return "\n".join(
        "".join(format(int(v) & 15, 'x') for v in row)
        for row in grid
    )


def format_symbolic(grid: List[List[int]], symbol_map: Dict[int, str] | None = None) -> str:
    """
    Format grid using symbolic character mapping.

    Uses meaningful characters for semantic clarity:
    - W = Wall, T = Target, . = Floor, X = Enemy, | = Boundary

    Token cost: ~95 tokens for 16x16 grid (same as hex).

    Args:
        grid: 2D integer grid
        symbol_map: Optional custom mapping (int -> char)

    Example: "||||||||||||||||\n|..............|\n|...W...T......|"
    """
    if not grid:
        return "(empty grid)"

    # Default symbolic map if none provided
    if symbol_map is None:
        from lucidgym.utils.representation import DEFAULT_SYMBOLIC_MAP
        symbol_map = DEFAULT_SYMBOLIC_MAP

    return "\n".join(
        "".join(symbol_map.get(int(v) & 15, "?") for v in row)
        for row in grid
    )


# ASCII density palette for format_ascii
ASCII_PALETTE = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. "


def format_ascii(grid: List[List[int]]) -> str:
    """
    Format grid using ASCII density palette characters.

    Maps integer values (0-15) to ASCII characters based on visual density.
    Token cost: ~95 tokens for 16x16 grid.

    Example: "$@B%8&WM#*oahkbd" for values 0-15
    """
    if not grid:
        return "(empty grid)"

    palette_len = len(ASCII_PALETTE)
    max_val = 16  # Integer values range 0-15

    lines = []
    for row in grid:
        chars = []
        for v in row:
            # Map integer (0-15) to palette index, clamping to valid range
            clamped_v = max(0, min(15, int(v)))  # Clamp to 0-15
            idx = int((clamped_v / max_val) * (palette_len - 1))
            idx = min(idx, palette_len - 1)  # Safety clamp for edge cases
            chars.append(ASCII_PALETTE[idx])
        lines.append("".join(chars))

    return "\n".join(lines)


#     plt.figure(figsize=(8, 8))
#     # 'tab20' is great for distinct categories (integers)
#     # 'nearest' keeps the pixels sharp (no blurring)
#     plt.imshow(matrix, cmap='tab20', interpolation='nearest')
    
#     plt.axis('off')  # Turn off axis numbers
#     plt.savefig(filename, bbox_inches='tight', pad_inches=0)
#     plt.close()
#     print(f"Saved visualization to {filename}")