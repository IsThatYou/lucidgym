"""
Adaptive prompt system that generates format-specific legends and explanations.

This module provides prompt builders that adapt to the representation format in use,
generating appropriate legends and semantic explanations.
"""

from __future__ import annotations

from typing import Dict, Optional

from lucidgym.utils.representation import (
    GridFormat,
    RepresentationConfig,
    DEFAULT_CELL_MEANINGS,
)


def build_format_legend(config: RepresentationConfig, for_general: bool = False) -> str:
    """
    Generate a legend explaining the grid format for the system prompt.

    Args:
        config: The representation configuration
        for_general: If True, omit game-specific meanings (for general learning prompts)

    Returns:
        A string explaining how to interpret the grid format
    """
    match config.format:
        case GridFormat.INTEGERS_SPACED:
            return _build_integers_legend(config.cell_meanings, for_general)
        case GridFormat.INTEGERS_COMPACT:
            return _build_integers_legend(config.cell_meanings, for_general)
        case GridFormat.INTEGERS_TUPLE:
            return _build_tuple_legend(config.cell_meanings, for_general)
        case GridFormat.HEX:
            return _build_hex_legend(config.cell_meanings, for_general)
        case GridFormat.SYMBOLIC:
            return _build_symbolic_legend(config.symbolic_map, config.cell_meanings, for_general)
        case GridFormat.ASCII:
            return _build_ascii_legend(config.cell_meanings, for_general)
        case _:
            return _build_integers_legend(config.cell_meanings, for_general)


def _build_integers_legend(meanings: Dict[int, str], for_general: bool = False) -> str:
    """Build legend for integer-based formats.

    Args:
        meanings: Mapping of integer codes to semantic descriptions
        for_general: If True, omit game-specific meanings (for general learning prompts)
    """
    if for_general:
        return (
            "The grid shows integer codes (0-15). Each integer represents a distinct tile type.\n"
            "You must discover what each integer means through experimentation."
        )
    lines = [
        "The grid shows integer codes (0-15). Key meanings:",
    ]
    for code in sorted(meanings.keys()):
        meaning = meanings[code]
        lines.append(f"  {code:2d} = {meaning}")
    return "\n".join(lines)


def _build_tuple_legend(meanings: Dict[int, str], for_general: bool = False) -> str:
    """Build legend for tuple-style format."""
    if for_general:
        return (
            "The grid uses tuple format: each row is (val1, val2, ...). Values are 0-15.\n"
            "Each integer represents a distinct tile type. Discover meanings through experimentation."
        )
    lines = [
        "The grid uses tuple format: each row is (val1, val2, ...). Values are 0-15.",
        "Key meanings:",
    ]
    for code in sorted(meanings.keys()):
        meaning = meanings[code]
        lines.append(f"  {code:2d} = {meaning}")
    return "\n".join(lines)


def _build_ascii_legend(meanings: Dict[int, str], for_general: bool = False) -> str:
    """Build legend for ASCII density palette format."""
    base = (
        "The grid uses ASCII density characters mapping integers 0-15 to visual density.\n"
        "Dense chars like '$@B' represent low values (0-2), sparse chars like '. ' represent high values (13-15)."
    )
    if for_general:
        return base + "\nEach underlying integer represents a distinct tile type. Discover meanings through experimentation."
    lines = [base, "Key meanings by integer value:"]
    for code in sorted(meanings.keys()):
        meaning = meanings[code]
        lines.append(f"  {code:2d} = {meaning}")
    return "\n".join(lines)


def _build_hex_legend(meanings: Dict[int, str], for_general: bool = False) -> str:
    """Build legend for hexadecimal format."""
    base = "The grid uses hexadecimal: 0-9 and a-f (where a=10, b=11, c=12, d=13, e=14, f=15)."
    if for_general:
        return base + "\nEach hex character represents a distinct tile type. Discover meanings through experimentation."
    lines = [base, "Key meanings:"]
    for code in sorted(meanings.keys()):
        meaning = meanings[code]
        hex_char = format(code, 'x')
        lines.append(f"  '{hex_char}' (value {code}) = {meaning}")
    return "\n".join(lines)


def _build_symbolic_legend(
    symbol_map: Dict[int, str], meanings: Dict[int, str], for_general: bool = False
) -> str:
    """Build legend for symbolic format."""
    # Build the symbol-to-integer mapping
    symbol_mapping_lines = ["The grid uses symbolic characters. Symbol to integer mapping:"]
    for code in range(16):
        symbol = symbol_map.get(code, "?")
        symbol_mapping_lines.append(f"  '{symbol}' = integer {code}")

    if for_general:
        return "\n".join(symbol_mapping_lines) + "\nDiscover what each tile type means through experimentation."

    lines = symbol_mapping_lines + ["", "Known tile meanings:"]
    for code in sorted(meanings.keys()):
        symbol = symbol_map.get(code, "?")
        meaning = meanings[code]
        lines.append(f"  '{symbol}' (value {code}) = {meaning}")
    return "\n".join(lines)


def build_format_clarification(config: RepresentationConfig) -> str:
    """
    Build a brief clarification string for user prompts.

    This is inserted into observation/action prompts to remind the model
    about the format being used.
    """
    match config.format:
        case GridFormat.INTEGERS_SPACED:
            return ""  # Default format, no clarification needed
        case GridFormat.INTEGERS_COMPACT:
            return "\n[Note: Grid uses space-separated integers without row indices]"
        case GridFormat.INTEGERS_TUPLE:
            return "\n[Note: Grid uses tuple format - each row is (val1, val2, ...)]"
        case GridFormat.HEX:
            return "\n[Note: Grid uses hex format - 0-9 and a-f where a=10...f=15]"
        case GridFormat.SYMBOLIC:
            return "\n[Note: Grid uses symbols - W=wall, T=target, .=floor, X=enemy, |=boundary]"
        case GridFormat.ASCII:
            return "\n[Note: Grid uses ASCII density chars - dense='$@B' (low vals), sparse='. ' (high vals)]"
        case _:
            return ""


def build_adaptive_observation_system(
    config: RepresentationConfig,
    game_id: Optional[str] = None,
    use_general: bool = False,
    grid_size: str = "16x16",
) -> str:
    """
    Build an adaptive system prompt for observation.

    Args:
        config: Representation configuration
        game_id: Game identifier (e.g., "as66")
        use_general: Whether to use general learning prompts
        grid_size: Grid dimensions string ("16x16" or "64x64")

    Returns:
        System prompt adapted to the representation format
    """
    # Get the format legend
    legend = build_format_legend(config)

    # Base template
    if use_general:
        base = _GENERAL_OBS_SYSTEM_TEMPLATE.format(grid_size=grid_size)
    elif game_id and game_id.lower().startswith("as66"):
        base = _AS66_OBS_SYSTEM_ADAPTIVE
    else:
        base = _GENERAL_OBS_SYSTEM_TEMPLATE.format(grid_size=grid_size)

    # Inject the legend
    return base.replace("{format_legend}", legend)


def build_adaptive_action_system(
    config: RepresentationConfig,
    game_id: Optional[str] = None,
    use_general: bool = False,
) -> str:
    """
    Build an adaptive system prompt for action selection.

    Args:
        config: Representation configuration
        game_id: Game identifier
        use_general: Whether to use general prompts

    Returns:
        System prompt for action selection
    """
    # Action prompts typically don't need the legend since it was in observation
    if use_general or not (game_id and game_id.lower().startswith("as66")):
        return _GENERAL_ACT_SYSTEM
    return _AS66_ACT_SYSTEM


# ---------------------------------------------------------------------------
# Adaptive prompt templates
# ---------------------------------------------------------------------------

_AS66_OBS_SYSTEM_ADAPTIVE = """You are playing a game represented by a {grid_size} grid.
Your task is to observe the position and analyze potential moves.

{format_legend}

Movement model:
- There is one main movable piece. It may be a unique integer or small block.
- When you choose a direction (Up, Down, Left, Right), the piece slides until blocked.
- Sliding can wrap across board edges if unobstructed.
- If no obstacles in a direction, the piece returns to start (no movement).

Obstacles and structures:
- Walls block movement (you stop adjacent to them).
- Target region forms a U-shape (2x3 with center removed). Fill it to win.
- Background cells are the playable area.
- Boundaries delimit the play field.
- Some levels have enemies (large blocks) - collision means game over.

For observation, analyze:
1. Locate the movable piece(s) and key structures
2. For each direction, simulate where the piece would land
3. Consider enemy movement if present
4. Determine which direction best progresses toward the goal

DO NOT call an action tool here - only provide analysis.
""".replace("{grid_size}", "16x16")

_GENERAL_OBS_SYSTEM_TEMPLATE = """You are playing a puzzle game shown as a {grid_size} grid.
Learn the rules by observing state changes after actions.

{{format_legend}}

Your goals:
1. Identify what you control (movable pieces)
2. Understand how movement works (slide? step? wrap?)
3. Find the objective (targets, goals)
4. Avoid hazards

For observation:
- Locate key elements in the grid
- For each direction (Up/Down/Left/Right), predict outcomes
- Recommend the best action based on your analysis

DO NOT call an action here - only provide reasoning.
"""

_AS66_ACT_SYSTEM = """Select exactly one move by calling a single tool. Do not include prose.
Available tools:
- ACTION1 = Up
- ACTION2 = Down
- ACTION3 = Left
- ACTION4 = Right
"""

_GENERAL_ACT_SYSTEM = """Select exactly one action (function call only; no prose).
Available tools:
- RESET - Start over
- ACTION1 = Up
- ACTION2 = Down
- ACTION3 = Left
- ACTION4 = Right
- ACTION5 = Space/Enter/Confirm
- ACTION6 = Click(x,y) - Click at grid coordinates
"""


def build_observation_user_adaptive(
    grid_text: str,
    score: int,
    step: int,
    config: RepresentationConfig,
    include_matrix: bool = True,
) -> str:
    """
    Build the user message for observation with adaptive format.

    Args:
        grid_text: Pre-formatted grid string
        score: Current score
        step: Current step number
        config: Representation config
        include_matrix: Whether to include the grid

    Returns:
        User message for observation prompt
    """
    matrix_block = f"\nCurrent state:\n{grid_text}\n" if include_matrix else ""
    clarification = build_format_clarification(config)

    return f"""Score: {score}
Step: {step}
{matrix_block}
Provide your analysis:
1. Identify the movable piece(s) and key structures
2. For Up, Down, Left, Right: predict landing positions
3. Recommend the best direction and explain why
{clarification}"""


def build_action_user_adaptive(
    grid_text: str,
    last_observation: str,
    config: RepresentationConfig,
    include_matrix: bool = True,
) -> str:
    """
    Build the user message for action selection with adaptive format.

    Args:
        grid_text: Pre-formatted grid string
        last_observation: Summary from observation phase
        config: Representation config
        include_matrix: Whether to include the grid

    Returns:
        User message for action prompt
    """
    matrix_block = f"\nCurrent state:\n{grid_text}\n" if include_matrix else ""
    clarification = build_format_clarification(config)

    return f"""Choose the best single move as a function call.
{matrix_block}
Previous analysis:
{last_observation}
{clarification}"""
