"""
Visual multimodal agent with hypothesis tracking for AS66 game.
Extends the memory agent to use images + text diffs for hypothesis management.
"""
from __future__ import annotations
from typing import Any, List, Dict
import base64
import logging
import os
from pathlib import Path

from rllm.agents.agent import Action

from lucidgym.utils.grid_processing import downsample_4x4, render_grid_to_png_bytes
from lucidgym.prompts.visual_memory_prompts import (
    build_initial_hypotheses_system_prompt,
    build_initial_hypotheses_user_content,
    build_update_hypotheses_system_prompt,
    build_observation_system_prompt,
    build_action_selection_system_prompt,
    build_action_selection_user_prompt,
)

from .hypothesis_agent import AS66MemoryAgent

log = logging.getLogger(__name__)


class TurnData:
    """Stores complete data for a single turn to enable image regeneration."""

    def __init__(
        self,
        turn_number: int,
        action_str: str,
        before_grid: List[List[int]],
        after_grid: List[List[int]],
        diff_str: str,
        is_level_up: bool = False,
        is_game_over: bool = False
    ):
        self.turn_number = turn_number
        self.action_str = action_str
        self.before_grid = before_grid
        self.after_grid = after_grid
        self.diff_str = diff_str
        self.is_level_up = is_level_up
        self.is_game_over = is_game_over


class AS66VisualMemoryAgent(AS66MemoryAgent):
    """
    Agent that uses multimodal context (images + text diffs) to manage hypotheses.
    Implements interleaved image+text history that scales with number of turns.
    """

    IMAGE_DIR = Path("memory") / "images"

    def __init__(
        self,
        name: str = "as66_visual_memory_agent",
        model: str = "gpt-4o",
        reasoning_effort: str = "low",
        game_id: str | None = None,
        downsample: bool = True,
        include_text_diff: bool = True,
        context_length_limit: int = -1,
        image_detail_level: str = "low",
        pixels_per_cell: int = 24,
        representation: "RepresentationConfig | None" = None,
        use_general: bool = False,
    ) -> None:
        """
        Initialize the visual memory agent.

        Args:
            name: Agent name
            model: OpenAI vision model to use
            reasoning_effort: Reasoning effort level
            game_id: Game identifier
            downsample: Whether to downsample grids to 16x16
            include_text_diff: Whether to include text diffs
            context_length_limit: Max context tokens (-1 for unlimited)
            image_detail_level: OpenAI image detail level (low/high)
            pixels_per_cell: Pixels per grid cell for rendering
            representation: Grid representation configuration
            use_general: If True, use general learning prompts
        """
        from lucidgym.utils.representation import RepresentationConfig
        super().__init__(
            name=name,
            model=model,
            reasoning_effort=reasoning_effort,
            game_id=game_id,
            downsample=downsample,
            include_text_diff=include_text_diff,
            context_length_limit=context_length_limit,
            representation=representation or RepresentationConfig(downsample=downsample),
            use_general=use_general,
        )

        # Vision-specific settings
        self.image_detail_level = image_detail_level
        self.pixels_per_cell = pixels_per_cell

        # Ensure per-game image directory exists
        self.game_image_dir = self.IMAGE_DIR / self.game_id
        self.game_image_dir.mkdir(parents=True, exist_ok=True)

        # Store structured turn data for regenerating images
        self.turn_history: List[TurnData] = []

    def _get_grid_from_frame(self, frame_3d: List[List[List[int]]]) -> List[List[int]]:
        """Helper to get the correct grid (16x16 or 64x64) based on downsample setting."""
        if self.downsample:
            return downsample_4x4(frame_3d, take_last_grid=True, round_to_int=True)
        else:
            return frame_3d[-1] if frame_3d else []

    def _grid_to_base64(self, grid: List[List[int]]) -> str:
        """Convert a grid to base64 PNG data (without the data URL prefix)."""
        try:
            png_bytes = render_grid_to_png_bytes(grid, cell=self.pixels_per_cell)
            if not png_bytes:
                raise ValueError("Rendered empty PNG bytes")
            return base64.b64encode(png_bytes).decode('utf-8')
        except Exception as e:
            log.error(f"[{self.game_id}] Failed to generate image: {e}")
            return ""

    def _build_turn_multimodal_content(self, turn: TurnData) -> List[Dict[str, Any]]:
        """
        Builds interleaved multimodal content for a single turn.
        Returns a list of content items (text + images).
        """
        content = []

        # Turn header and action
        header_text = f"### Turn {turn.turn_number}\n\n**Action:** `{turn.action_str}`\n\n"

        if turn.is_level_up:
            header_text = f"> **LEVEL UP! A new level begins below.**\n>\n> {header_text}"
        elif turn.is_game_over:
            header_text = f"> **GAME OVER!**\n>\n> {header_text}"

        content.append({"type": "text", "text": header_text + "**State Before:**"})

        # Before image
        before_b64 = self._grid_to_base64(turn.before_grid)
        if before_b64:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{before_b64}",
                    "detail": self.image_detail_level
                }
            })
        else:
            content.append({"type": "text", "text": "\n*(Image generation failed)*"})

        content.append({"type": "text", "text": "\n\n**State After:**"})

        # After image
        after_b64 = self._grid_to_base64(turn.after_grid)
        if after_b64:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{after_b64}",
                    "detail": self.image_detail_level
                }
            })
        else:
            content.append({"type": "text", "text": "\n*(Image generation failed)*"})

        # Optional text diff
        if self.include_text_diff:
            content.append({
                "type": "text",
                "text": f"\n\n**Resulting Textual Diff:**\n```\n{turn.diff_str}\n```\n\n---\n\n"
            })
        else:
            content.append({"type": "text", "text": "\n\n---\n\n"})

        return content

    def _get_turns_within_context_limit(self) -> List[TurnData]:
        """
        Returns subset of turn_history that fits within context limit,
        preserving high-information turns (level ups, game overs).
        """
        if self.context_length_limit == -1:
            return self.turn_history

        if not self.turn_history:
            return []

        # Separate high-info vs regular turns
        high_info_turns = []
        regular_turns = []

        for turn in self.turn_history:
            if turn.is_level_up or turn.is_game_over:
                high_info_turns.append(turn)
            else:
                regular_turns.append(turn)

        # Estimate tokens per turn
        # Images with detail="low" are ~85 tokens each (OpenAI)
        # ~150 tokens text + 170 tokens for 2 images = ~320 tokens per turn
        TOKENS_PER_TURN = 320

        max_turns = max(1, self.context_length_limit // TOKENS_PER_TURN)

        # Always include high-info turns
        kept_turns = high_info_turns[:]
        remaining_slots = max_turns - len(high_info_turns)

        if remaining_slots > 0 and regular_turns:
            # Fill remaining slots with most recent regular turns
            kept_turns.extend(regular_turns[-remaining_slots:])

        # Sort by turn number to maintain chronological order
        kept_turns.sort(key=lambda t: t.turn_number)

        return kept_turns

    def _build_full_history_multimodal_content(self) -> List[Dict[str, Any]]:
        """
        Builds complete interleaved multimodal content for all turns within context limit.
        Returns a list ready to be included in user message content.
        """
        turns_to_include = self._get_turns_within_context_limit()

        if not turns_to_include:
            return [{"type": "text", "text": "## Move History\n\n(No moves recorded yet.)\n\n"}]

        content = [{"type": "text", "text": "## Move History\n\n"}]

        for turn in turns_to_include:
            content.extend(self._build_turn_multimodal_content(turn))

        return content

    def _initialize_memory(self, observation: dict) -> None:
        """Generate initial hypotheses using an image."""
        log.info(f"[{self.game_id}] Initializing visual memory and hypotheses...")

        grid_3d = observation.get("frame", [])
        grid = self._get_grid_from_frame(grid_3d)

        img_b64 = self._grid_to_base64(grid)
        if not img_b64:
            log.error(f"[{self.game_id}] Cannot initialize memory, image generation failed.")
            self.hypotheses_content = "## Hypotheses\n\n(ERROR: Initial image generation failed.)"
            self._is_initialized = True
            return

        sys_prompt = build_initial_hypotheses_system_prompt()
        user_content = build_initial_hypotheses_user_content(self.game_id, img_b64, detail=self.image_detail_level)

        # Build API call parameters
        api_params = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_content}
            ],
        }

        # Add reasoning_effort for GPT-5 series models
        if self.model.startswith("gpt-5"):
            api_params["reasoning_effort"] = self.reasoning_effort

        resp = self._client.chat.completions.create(**api_params)

        self.hypotheses_content = f"## Hypotheses\n\n{(resp.choices[0].message.content or '').strip()}"
        self._token_total += getattr(resp.usage, "total_tokens", 0) or 0

        # Don't store initial state in turn_history since it's just setup
        self.move_history_content = "## Move History\n\n"

        self._write_memory()
        self._is_initialized = True
        log.info(f"[{self.game_id}] Initial visual hypotheses generated.")

    def _update_memory_from_action(
        self,
        prev_observation: dict,
        action_dict: dict,
        new_observation: dict
    ) -> bool:
        """
        Updates memory with latest move and revises hypotheses using
        multimodal context (all past images + new images).
        """
        prev_grid_3d = prev_observation.get("frame", [])
        new_grid_3d = new_observation.get("frame", [])

        prev_grid = self._get_grid_from_frame(prev_grid_3d)
        new_grid = self._get_grid_from_frame(new_grid_3d)

        # Skip memory update if either grid is empty (e.g., NOT_PLAYED state)
        if not prev_grid or not new_grid:
            log.debug(f"[{self.game_id}] Skipping memory update: empty grid (prev={bool(prev_grid)}, new={bool(new_grid)})")
            return False

        state_hash = self._get_state_hash(prev_grid)

        # Build action identifier
        action_name = action_dict.get("name", "UNKNOWN")
        action_identifier = action_name

        state_action_tuple = (state_hash, action_identifier)

        # Check for repeated state-action
        if state_action_tuple in self.seen_state_actions:
            log.warning(f"[{self.game_id}] Repeated state-action pair detected: {action_identifier}. Applying penalty.")
            return True

        self.seen_state_actions.add(state_action_tuple)

        prev_score = prev_observation.get("score", 0)
        new_score = new_observation.get("score", 0)
        is_level_up = new_score > prev_score
        diff = self._calculate_diff(prev_grid, new_grid)

        # Store this turn's data
        turn_number = len(self.seen_state_actions)
        turn_data = TurnData(
            turn_number=turn_number,
            action_str=action_identifier,
            before_grid=prev_grid,
            after_grid=new_grid,
            diff_str=diff,
            is_level_up=is_level_up,
            is_game_over=(new_observation.get("state") == "GAME_OVER")
        )
        self.turn_history.append(turn_data)

        # Update text-based memory file (for disk persistence)
        self._update_text_memory(turn_data)

        # Build multimodal content for hypothesis update
        log.info(f"[{self.game_id}] Updating hypotheses based on visual move history...")
        sys_prompt = build_update_hypotheses_system_prompt()

        # Build user content: hypotheses text + full interleaved history
        user_content = [
            {"type": "text", "text": "Here is the game memory so far, including your prior hypotheses.\n\n"},
            {"type": "text", "text": self.hypotheses_content + "\n\n"},
        ]

        # Add full interleaved history
        user_content.extend(self._build_full_history_multimodal_content())

        if is_level_up:
            special_instruction = (
                "**IMPORTANT CONTEXT: A LEVEL UP just occurred.** The game board has changed for the new level. "
                "Your primary task now is to re-validate your current hypotheses against this new environment. "
                "Generate a new set of five hypotheses that reflect your understanding of this new level.\n\n"
            )
            user_content.insert(0, {"type": "text", "text": special_instruction})

        user_content.append({
            "type": "text",
            "text": "\nAnalyze this evidence and provide an updated list of five refined hypotheses."
        })

        # Build API call parameters
        api_params = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_content}
            ],
        }

        # Add reasoning_effort for GPT-5 series models
        if self.model.startswith("gpt-5"):
            api_params["reasoning_effort"] = self.reasoning_effort

        resp = self._client.chat.completions.create(**api_params
        )
        self._token_total += getattr(resp.usage, "total_tokens", 0) or 0

        new_hypotheses = (resp.choices[0].message.content or "").strip()
        self.hypotheses_content = "## Hypotheses\n\n" + (new_hypotheses if new_hypotheses else self.hypotheses_content.split("\n\n", 1)[1])

        self._write_memory()
        log.info(f"[{self.game_id}] Visual memory and hypotheses updated.")
        return False

    def _update_text_memory(self, turn: TurnData) -> None:
        """Updates the text-based memory file (for disk persistence)."""
        entry_header = f"### Turn {turn.turn_number}\n\n"
        entry_parts = [
            f"**Action:** `{turn.action_str}`\n\n",
            "**State Before (Image):** [Image stored]\n\n",
            "**State After (Image):** [Image stored]\n",
        ]

        if self.include_text_diff:
            entry_parts.append(f"\n**Resulting Textual Diff:**\n```\n{turn.diff_str}\n```\n")
        else:
            entry_parts.append("\n")

        entry_body = "".join(entry_parts)

        if turn.is_level_up:
            history_entry = f"> **LEVEL UP! A new level begins below.**\n>\n{'> '.join((entry_header + entry_body).splitlines(True))}\n---\n\n"
        elif turn.is_game_over:
            history_entry = f"> **GAME OVER!**\n>\n{'> '.join((entry_header + entry_body).splitlines(True))}\n---\n\n"
        else:
            history_entry = f"{entry_header}{entry_body}\n---\n\n"

        self.move_history_content += history_entry

    def _get_observation_text(self, memory_content: str, grid: List[List[int]], score: int, step: int) -> str:
        """
        Call the LLM with complete multimodal context (all past images + current image)
        to get a text observation.
        """
        # Generate current state image
        current_img_b64 = self._grid_to_base64(grid)
        if not current_img_b64:
            log.error(f"[{self.game_id}] Cannot get observation, image generation failed.")
            return "ERROR: Could not generate current state image for observation."

        sys_prompt = build_observation_system_prompt()

        # Build user content: status + current state + hypotheses + full history
        user_content = [
            {"type": "text", "text": f"**Current Game Status:**\n- Step: {step}\n- Score: {score}\n\n**Current Board State (Image):**"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{current_img_b64}", "detail": self.image_detail_level}},
            {"type": "text", "text": "\n\n" + self.hypotheses_content + "\n\n"},
        ]

        # Add full interleaved history (all past images + text)
        user_content.extend(self._build_full_history_multimodal_content())

        user_content.append({
            "type": "text",
            "text": "\nFollow your reasoning process and provide a detailed text analysis, concluding with your recommended action. Be precise with all coordinates."
        })

        # Store prompts for logging (text representation of multimodal content)
        user_text_parts = [item.get("text", "[IMAGE]") if item.get("type") == "text" else "[IMAGE]" for item in user_content]
        self._last_observation_prompt = f"SYSTEM:\n{sys_prompt}\n\nUSER:\n" + "\n".join(user_text_parts)

        # Build API call parameters
        api_params = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_content}
            ],
        }

        # Add reasoning_effort for GPT-5 series models
        if self.model.startswith("gpt-5"):
            api_params["reasoning_effort"] = self.reasoning_effort

        resp = self._client.chat.completions.create(**api_params)
        self._token_total += getattr(resp.usage, "total_tokens", 0) or 0

        observation = (resp.choices[0].message.content or "No observation generated.").strip()
        # Store response for logging
        self._last_observation_response = observation
        log.info(f"[{self.game_id} | Step {step}] Visual Observation Rationale generated.")
        return observation

    def update_from_model(self, action_payload: dict | None = None, **_: Any) -> dict:
        """
        Convert model response to Action.
        Main logic loop that calls the overridden multimodal methods.
        """
        # Reuse parent logic which now calls our overridden multimodal methods
        return super().update_from_model(action_payload, **_)
