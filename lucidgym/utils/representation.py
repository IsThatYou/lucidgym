"""
Configurable grid representation framework.

Supports multiple formats for representing the game state grid:
- INTEGERS_SPACED: Current format with row indices (backward compatible)
- INTEGERS_COMPACT: No spaces between single digits
- HEX: Single hex characters (0-9, a-f) - most token efficient
- SYMBOLIC: Custom character mapping (W=wall, T=target, etc.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict


class GridFormat(Enum):
    """Grid representation formats."""

    INTEGERS_SPACED = "integers_spaced"  # "[ 0]  1  2  3..." (current default, ~745 tokens)
    INTEGERS_COMPACT = "integers_compact"  # "1 15 15 4 0..." (space-separated, ~400 tokens)
    INTEGERS_TUPLE = "integers_tuple"  # "(1, 15, 15, 4, 0...)" (tuple-like, ~500 tokens)
    HEX = "hex"  # "0123456789abcdef" (single hex chars, ~95 tokens)
    SYMBOLIC = "symbolic"  # "W.T|X..." (custom symbols, ~95 tokens)
    ASCII = "ascii"  # ASCII palette chars like "$@B%8&..." (~95 tokens)


# Default semantic meanings for AS66 game
DEFAULT_CELL_MEANINGS: Dict[int, str] = {
    0: "target (U-shaped goal region)",
    1: "boundary/border",
    4: "wall (impassable)",
    6: "move counter",
    8: "enemy/hostile",
    9: "enemy/hostile",
    14: "boundary/border",
    15: "floor/background (playable area)",
}

# Default symbolic character mapping
DEFAULT_SYMBOLIC_MAP: Dict[int, str] = {
    0: "T",  # Target
    1: "|",  # Boundary
    2: "2",
    3: "3",
    4: "W",  # Wall
    5: "5",
    6: "#",  # Counter
    7: "7",
    8: "X",  # Enemy
    9: "X",  # Enemy
    10: "A",
    11: "B",
    12: "C",
    13: "D",
    14: "|",  # Boundary
    15: ".",  # Floor
}


@dataclass
class RepresentationConfig:
    """Configuration for grid state representation.

    Attributes:
        format: The grid format to use (integers_spaced, hex, symbolic, etc.)
        downsample: Whether to downsample 64x64 to 16x16
        include_row_indices: Whether to include row indices in output
        cell_meanings: Mapping of integer codes to semantic descriptions
        symbolic_map: Mapping of integer codes to symbolic characters
    """

    format: GridFormat = GridFormat.INTEGERS_SPACED
    downsample: bool = True
    include_row_indices: bool = True  # Only used for INTEGERS_SPACED
    cell_meanings: Dict[int, str] = field(
        default_factory=lambda: DEFAULT_CELL_MEANINGS.copy()
    )
    symbolic_map: Dict[int, str] = field(
        default_factory=lambda: DEFAULT_SYMBOLIC_MAP.copy()
    )

    @classmethod
    def from_string(cls, format_str: str, **kwargs) -> "RepresentationConfig":
        """Create config from format string."""
        try:
            grid_format = GridFormat(format_str)
        except ValueError:
            raise ValueError(
                f"Unknown format: {format_str}. "
                f"Valid options: {[f.value for f in GridFormat]}"
            )
        return cls(format=grid_format, **kwargs)

    def get_format_description(self) -> str:
        """Get human-readable description of the format."""
        descriptions = {
            GridFormat.INTEGERS_SPACED: "space-separated integers with row indices",
            GridFormat.INTEGERS_COMPACT: "space-separated integers (no row indices)",
            GridFormat.INTEGERS_TUPLE: "tuple-like format with parentheses and commas",
            GridFormat.HEX: "hexadecimal characters (0-9, a-f)",
            GridFormat.SYMBOLIC: "symbolic characters (W=wall, T=target, .=floor)",
            GridFormat.ASCII: "ASCII density palette characters",
        }
        return descriptions.get(self.format, "unknown format")
