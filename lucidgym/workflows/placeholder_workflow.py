"""
Placeholder workflow to prove end-to-end registration works.

The workflow simply resets its components and raises a TerminationEvent
to signal that it is not meant for actual rollouts.
"""
from __future__ import annotations

from typing import Any

from rllm.agents.agent import Episode
from rllm.workflows.workflow import TerminationEvent, TerminationReason, Workflow


class LucidGymPlaceholderWorkflow(Workflow):
    """Workflow that deliberately aborts execution."""

    async def run(self, task: dict, uid: str, **kwargs: Any) -> Episode | None:
        self.reset(task, uid)
        raise TerminationEvent(TerminationReason.ERROR)
