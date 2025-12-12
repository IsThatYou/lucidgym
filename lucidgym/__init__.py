"""
LucidGym
========

This package hosts extension modules that sit on top of the upstream
`rllm` agent/env/workflow stack. Nothing here is tightly coupled to
training infrastructure; instead we expose registry helpers so new
components can be imported and registered from CLI/config entry points.
"""

from .registry import register_lucidgym_components

__all__ = ["register_lucidgym_components"]
