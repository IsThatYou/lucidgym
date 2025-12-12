"""
LucidGym-specific Countdown environment customizations.

This module subclasses TextArena's ``CountdownEnv`` so LucidGym can tweak the
board observation formatting without vendoring the upstream file. Keeping the
logic here lets us track upstream bug fixes while still controlling how board
snapshots look to downstream adapters such as ``TextArenaEnv``.
"""
from __future__ import annotations

from typing import Any

import textarena as ta
from textarena.envs.Countdown.env import CountdownEnv as _BaseCountdownEnv



class LucidGymCountdownEnv(_BaseCountdownEnv):  # type: ignore[misc]
    """
    Countdown variant that removes the progress suffix from board observations.

    LucidGym's ``TextArenaEnv`` adapter expects raw observation tuples and
    handles its own aggregation. The stock Countdown board text always appends
    ``\"Current progress score: ...\"`` which adds noise to downstream prompts.
    Overriding ``_add_board_observation`` here keeps everything else aligned
    with upstream while trimming the board payload to just ``_render_board()``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def _add_board_observation(self) -> None:
        board_text = self._render_board()
        self.state.add_observation(  # type: ignore[attr-defined]
            message=board_text,
            observation_type=ta.ObservationType.GAME_BOARD,  # type: ignore[attr-defined]
        )


__all__ = ["LucidGymCountdownEnv"]

