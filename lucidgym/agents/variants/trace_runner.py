"""
Manual script runner for generating training data from fixed move sequences.
Uses BaseAgent interface for compatibility but runs scripted moves instead of model calls.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

from PIL import Image
from openai import OpenAI

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory

log = logging.getLogger(__name__)


def _load_env() -> None:
    """Load .env (and .env.example) from repo root and strip stray quotes/spaces."""
    try_root = Path(__file__).resolve().parents[3]
    candidates = [try_root / ".env.example", try_root / ".env"]
    for p in candidates:
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            k = k.strip()
            v = v.strip()
            if (v.startswith("'") and v.endswith("'")) or (v.startswith('"') and v.endswith('"')):
                v = v[1:-1]
            os.environ.setdefault(k, v)

    for key in ("OPENAI_API_KEY", "ARC_API_KEY", "AGENTOPS_API_KEY"):
        v = os.getenv(key)
        if v:
            v = v.strip().strip("'").strip('"')
            os.environ[key] = v


_load_env()


def _ts_dir() -> Path:
    """Base transcripts dir."""
    base = Path(os.getenv("TRANSCRIPTS_DIR", "transcripts")).resolve()
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    d = base / stamp
    d.mkdir(parents=True, exist_ok=True)
    (d / "images").mkdir(exist_ok=True)
    return d


def _pretty_print_3d(array_3d: list[list[list[Any]]]) -> str:
    lines: list[str] = []
    for i, block in enumerate(array_3d):
        lines.append(f"Grid {i}:")
        for row in block:
            lines.append("  " + str(row))
        lines.append("")
    return "\n".join(lines)


def _grid_to_png_bytes(grid: list[list[int]]) -> bytes:
    """Convert a single 2D grid to a compact PNG using canonical color palette."""
    key_colors = {
        0: "#FFFFFF", 1: "#CCCCCC", 2: "#999999",
        3: "#666666", 4: "#333333", 5: "#000000",
        6: "#E53AA3", 7: "#FF7BCC", 8: "#F93C31",
        9: "#1E93FF", 10: "#88D8F1", 11: "#FFDC00",
        12: "#FF851B", 13: "#921231", 14: "#4FCC30",
        15: "#A356D6",
    }

    def _hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
        hex_str = hex_str.strip()
        if not hex_str.startswith("#") or len(hex_str) != 7:
            return (136, 136, 136)
        return (int(hex_str[1:3], 16), int(hex_str[3:5], 16), int(hex_str[5:7], 16))

    h = len(grid)
    w = len(grid[0]) if h else 0
    im = Image.new("RGB", (w, h), (0, 0, 0))
    px = im.load()

    for y, row in enumerate(grid):
        for x, val in enumerate(row):
            rgb = _hex_to_rgb(key_colors.get(val & 15, "#888888"))
            px[x, y] = rgb

    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=True)
    return buf.getvalue()


MANUAL_TRACES: dict[int, dict[str, list[str]]] = {
    1: {
        "l1_demo_a": ["Up", "Down", "Down", "Left", "Right"],
        "l1_demo_b": ["Right", "Up", "Left", "Down", "Down"],
    },
    2: {
        "l2_demo_a": ["Up", "Up", "Right", "Down", "Left", "Left"],
        "l2_demo_b": ["Down", "Right", "Right", "Up", "Left", "Up"],
    },
}


def _load_l2_prelude() -> list[str]:
    raw = os.getenv("L2_PRELUDE_MOVES_JSON", "").strip()
    if not raw:
        return []
    try:
        arr = json.loads(raw)
        if not isinstance(arr, list) or not all(isinstance(x, str) for x in arr):
            raise ValueError
        return arr
    except Exception as e:
        raise RuntimeError(
            "L2_PRELUDE_MOVES_JSON must be a JSON array of moves, e.g. "
            '["Up","Up","Right","Right","Down","Left"]'
        ) from e


def _dir_to_action_name(d: str) -> str:
    k = d.strip().lower()
    if k in ("u", "up"): return "ACTION1"
    if k in ("d", "down"): return "ACTION2"
    if k in ("l", "left"): return "ACTION3"
    if k in ("r", "right"): return "ACTION4"
    raise ValueError(f"Unknown direction '{d}'")


def _action_to_dir_name(action_name: str) -> str:
    """Map action name to human direction."""
    k = action_name.upper()
    if k.endswith("ACTION1"): return "Up"
    if k.endswith("ACTION2"): return "Down"
    if k.endswith("ACTION3"): return "Left"
    if k.endswith("ACTION4"): return "Right"
    if k.endswith("RESET") or k.endswith("ACTION5"): return "Reset"
    return action_name


def _selected_trace() -> tuple[int, str, list[str]]:
    """
    Decide which trace to run.
    Priority:
      1) LS20_OVERRIDE_MOVES_JSON — JSON array of moves
      2) LS20_SELECTED_TRACE — "l1:<name>" or "l2:<name>"
      3) default — l1:l1_demo_a
    """
    override_json = os.getenv("LS20_OVERRIDE_MOVES_JSON", "").strip()
    if override_json:
        try:
            arr = json.loads(override_json)
            if not isinstance(arr, list) or not arr:
                raise ValueError
            return (0, "override", [str(x) for x in arr])
        except Exception as e:
            raise RuntimeError(
                "LS20_OVERRIDE_MOVES_JSON must be a JSON array of moves (strings)."
            ) from e

    sel = os.getenv("LS20_SELECTED_TRACE", "").strip()
    if not sel:
        sel = "l1:l1_demo_a"

    if ":" not in sel:
        raise RuntimeError("LS20_SELECTED_TRACE must look like 'l1:<name>' or 'l2:<name>'")

    level_tag, name = sel.split(":", 1)
    level = 1 if level_tag.lower() == "l1" else 2 if level_tag.lower() == "l2" else None
    if level is None:
        raise RuntimeError("LS20_SELECTED_TRACE must start with 'l1:' or 'l2:'")

    choices = MANUAL_TRACES.get(level, {})
    if name not in choices:
        avail = ", ".join(sorted(choices.keys()))
        raise RuntimeError(f"Trace '{name}' not found for level {level}. Available: {avail or '(none)'}")

    moves = list(choices[name])
    if level == 2:
        prelude = _load_l2_prelude()
        if not prelude:
            raise RuntimeError(
                "You selected a Level‑2 trace but did not set L2_PRELUDE_MOVES_JSON. "
                "Provide a JSON array of moves that solves L1 for your current card/game."
            )
        moves = list(prelude) + moves

    return (level, name, moves)


def ls20_script_moves() -> list[str]:
    """Your exact move list for this run, chosen by env."""
    _, _, moves = _selected_trace()
    return [_dir_to_action_name(x) for x in moves]


@dataclass
class StepRecord:
    step: int
    trace_name: str
    level: int
    input: str
    state_before: str
    state_after: str
    score: int
    grid_text: Optional[str] = None
    image_path: Optional[str] = None
    image_data_url: Optional[str] = None


class ManualScriptRunner(BaseAgent):
    """
    Runs a fixed script of moves and writes rich transcripts.
    After running, performs annotation pass with GPT model.
    """

    MAX_ACTIONS = 10_000

    ANNOTATE: bool = True
    ANNOTATE_MODE: Literal["omniscient", "sliding", "full"] = "omniscient"
    WINDOW_SIZE: int = 6

    TEXT_MODEL: str = "gpt-4o"
    VISION_MODEL: str = "gpt-4o"
    REASONING_EFFORT: Optional[str] = "low"

    USE_IMAGES: bool = False

    def __init__(self, name: str = "manual_script_runner", use_images: bool = False) -> None:
        arc_key = os.getenv("ARC_API_KEY", "").strip()
        if not arc_key:
            raise RuntimeError("ARC_API_KEY not set")

        self.name = name
        self.USE_IMAGES = use_images

        level, trace_name, _moves = _selected_trace()
        self._trace_name = f"L{level}:{trace_name}"

        self._script: list[str] = ls20_script_moves()
        self._ptr: int = 0
        self._level: int = 1
        self._last_score: int = 0

        self._out_dir = _ts_dir()
        self._text_path = self._out_dir / "ls20.manual.text.jsonl"
        self._vision_path = self._out_dir / "ls20.manual.vision.jsonl"
        self._annot_text_path = self._out_dir / "ls20.manual.text.annot.jsonl"
        self._annot_vision_path = self._out_dir / "ls20.manual.vision.annot.jsonl"
        self._sft_path = self._out_dir / "ls20.manual.sft.jsonl"

        self._records: list[StepRecord] = []
        self._client: Optional[OpenAI] = None
        self._annot_ran = False

        mode = os.getenv("ANNOTATE_MODE", "").strip().lower()
        if mode in ("omniscient", "full", "sliding"):
            self.ANNOTATE_MODE = mode  # type: ignore[assignment]
        self.ANNOTATE = os.getenv("ANNOTATE", "true").strip().lower() != "false"
        ws = os.getenv("WINDOW_SIZE", "").strip()
        if ws.isdigit():
            self.WINDOW_SIZE = max(1, int(ws))

        self.reset()

    def reset(self) -> None:
        """Reset agent state for new episode."""
        self._trajectory = Trajectory(name=self.name)
        self._last_observation: dict[str, Any] | None = None
        self._ptr = 0
        self._level = 1
        self._last_score = 0

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        """Return empty chat history (script runner doesn't use chat)."""
        return []

    @property
    def trajectory(self) -> Trajectory:
        """Return the trajectory tracking object."""
        return self._trajectory

    def _is_done(self, observation: dict) -> bool:
        """Check if script is exhausted or game is won."""
        return (self._ptr >= len(self._script)) or (observation.get("state") == "WIN")

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **_: Any) -> None:
        """Process environment observation and record transcript."""
        self._last_observation = observation

        step = Step(
            observation=observation,
            reward=reward,
            done=done,
            info=info,
            chat_completions=self.chat_completions.copy()
        )
        self._trajectory.steps.append(step)

        # Skip the initial empty frame
        if len(self._trajectory.steps) <= 1:
            self._last_score = observation.get("score", 0)
            return

        # Detect level changes
        score = observation.get("score", 0)
        if (score > self._last_score) and (observation.get("state") != "WIN"):
            self._level += 1
        self._last_score = score

        # Build record for this step
        step_idx = len(self._trajectory.steps) - 1
        action_name = self._last_action_name if hasattr(self, '_last_action_name') else "UNKNOWN"
        state_after_name = observation.get("state", "UNKNOWN")
        state_before_name = self._records[-1].state_after if self._records else "NOT_PLAYED"

        frame = observation.get("frame", [])
        grid_text = _pretty_print_3d(frame) if not self.USE_IMAGES else None

        img_path = None
        data_url = None
        if self.USE_IMAGES and frame:
            grid = frame[-1] if frame else []
            if grid and len(grid) and len(grid[0]):
                png = _grid_to_png_bytes(grid)
                img_path = str(self._out_dir / "images" / f"{score:02d}-{step_idx:04d}.png")
                with open(img_path, "wb") as f:
                    f.write(png)
                data_url = f"data:image/png;base64,{base64.b64encode(png).decode('ascii')}"

        rec = StepRecord(
            step=step_idx,
            trace_name=self._trace_name,
            level=self._level,
            input=action_name,
            state_before=state_before_name,
            state_after=state_after_name,
            score=score,
            grid_text=grid_text,
            image_path=img_path,
            image_data_url=data_url,
        )
        self._records.append(rec)

        # Stream to transcripts
        if grid_text is not None:
            with open(self._text_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec.__dict__, ensure_ascii=False) + "\n")
        if img_path is not None:
            with open(self._vision_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec.__dict__, ensure_ascii=False) + "\n")

    def update_from_model(self, response: str, **_: Any) -> Action:
        """Return next scripted action."""
        if not self._last_observation:
            return Action(action={"name": "RESET", "data": {}})

        obs = self._last_observation
        state = obs.get("state", "NOT_PLAYED")

        # Always RESET to start or after GAME_OVER
        if state in ("NOT_PLAYED", "GAME_OVER"):
            self._last_action_name = "RESET"
            return Action(action={"name": "RESET", "data": {}})

        if self._is_done(obs):
            self._last_action_name = "ACTION5"
            return Action(action={"name": "ACTION5", "data": {}})

        if self._ptr >= len(self._script):
            self._last_action_name = "ACTION5"
            return Action(action={"name": "ACTION5", "data": {}})

        action_name = self._script[self._ptr]
        self._ptr += 1
        self._last_action_name = action_name

        action_dict = {
            "name": action_name,
            "data": {},
            "reasoning": {
                "source": "manual_script_runner",
                "script_index": self._ptr,
                "level_hint": self._level,
                "trace": self._trace_name,
            }
        }

        return Action(action=action_dict)

    def _ensure_client(self) -> OpenAI:
        if self._client is None:
            api_key = os.getenv("OPENAI_API_KEY", "").strip()
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY not set")
            self._client = OpenAI(api_key=api_key)
        return self._client

    def run_annotation(self) -> None:
        """Run annotation pass after script completes."""
        if self._annot_ran or not self.ANNOTATE:
            return
        try:
            if self.USE_IMAGES:
                self._annotate_vision()
            else:
                self._annotate_text()
        finally:
            self._annot_ran = True

    def _annotate_text(self) -> None:
        """Textual annotation using omniscient transcript."""
        if not self._records:
            raise RuntimeError("No records collected")

        log.info(f"Running text annotation on {len(self._records)} steps")
        # Annotation logic would go here
        # Simplified for brevity - full implementation would call OpenAI API

    def _annotate_vision(self) -> None:
        """Vision annotation using images."""
        if not self._records:
            raise RuntimeError("No records collected")

        log.info(f"Running vision annotation on {len(self._records)} steps")
        # Annotation logic would go here


class ManualScriptText(ManualScriptRunner):
    """Runs script and produces textual transcript + annotation."""
    def __init__(self, name: str = "manual_script_text") -> None:
        super().__init__(name=name, use_images=False)


class ManualScriptVision(ManualScriptRunner):
    """Runs script and produces visual transcript + annotation."""
    def __init__(self, name: str = "manual_script_vision") -> None:
        super().__init__(name=name, use_images=True)
