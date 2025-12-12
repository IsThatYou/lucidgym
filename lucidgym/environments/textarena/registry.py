"""
Helpers that register LucidGym-owned TextArena environments.

We rely on TextArena's native registry so Hydra configs can reference LucidGym
variants via ``ta.make(env_id=...)`` without patching upstream packages.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from .countdown_env import LucidGymCountdownEnv  # noqa: F401 (import side-effect)

try:  # pragma: no cover - surfaced via helper
    import textarena as ta  # noqa: F401
    from textarena.envs import registration as ta_registration
except Exception as exc:  # pragma: no cover - exercised when deps missing
    ta_registration = None  # type: ignore[assignment]
    _TEXTARENA_IMPORT_ERROR = exc
else:
    _TEXTARENA_IMPORT_ERROR = None

WrapperSpec = Dict[str, List[Any] | None]


@dataclass(frozen=True)
class _LucidEnvOverride:
    """
    Declarative description of a LucidGym TextArena override.

    Setting ``base_env_id`` copies kwargs + wrapper variants from the upstream
    entry before applying optional overrides. Leaving it ``None`` requires
    callers to supply ``wrappers`` explicitly.
    """

    lucid_env_id: str
    entry_point: str
    base_env_id: str | None = None
    wrappers: WrapperSpec | None = None
    kwargs: Dict[str, Any] = field(default_factory=dict)


_LUCID_TEXTARENA_ENV_OVERRIDES: Sequence[_LucidEnvOverride] = (
    _LucidEnvOverride(
        base_env_id="Countdown-v0",
        lucid_env_id="LucidCountdown-v0",
        entry_point="lucidgym.environments.textarena.countdown_env:LucidGymCountdownEnv",
    ),
)


def _require_textarena_registry() -> None:
    if _TEXTARENA_IMPORT_ERROR is not None:  # pragma: no cover - gated by deps
        raise ImportError(
            "LucidGym's TextArena overrides require the `textarena` package. "
            "Install it via `pip install textarena` before registering environments."
        ) from _TEXTARENA_IMPORT_ERROR


def _get_env_spec(env_id: str):
    assert ta_registration is not None  # for type checkers
    spec = ta_registration.ENV_REGISTRY.get(env_id)
    if spec is None:
        raise KeyError(f"TextArena env '{env_id}' is not registered. Import textarena.envs before calling this helper.")
    return spec


def _collect_wrapper_variants(base_env_id: str) -> Dict[str, List[Any] | None]:
    """
    Return the wrapper presets for every suffix registered under ``base_env_id``.

    The keys mirror TextArena's ``register_with_versions`` contract: ``default``
    for the base env plus any suffixes such as ``-train``.
    """
    specs: Dict[str, List[Any] | None] = {}
    assert ta_registration is not None
    for env_id, spec in ta_registration.ENV_REGISTRY.items():
        if not env_id.startswith(base_env_id):
            continue
        suffix = env_id[len(base_env_id) :]
        key = "default" if suffix == "" else suffix
        specs[key] = spec.default_wrappers
    if "default" not in specs:
        raise KeyError(f"Unable to locate base env '{base_env_id}' in the TextArena registry.")
    return specs


def _discover_variant_ids(base_env_id: str) -> List[str]:
    """List every env ID that shares the same prefix as ``base_env_id``."""
    assert ta_registration is not None
    return sorted(env_id for env_id in ta_registration.ENV_REGISTRY if env_id.startswith(base_env_id))


def _ensure_base_envs_loaded() -> None:
    """
    Import TextArena's registry side-effects.

    TextArena registers all built-in games from ``textarena.envs``. Importing
    that module twice is harmless but guarantees ``ENV_REGISTRY`` is populated
    before LucidGym attempts to mirror those entries.
    """
    __import__("textarena.envs")  # noqa: F401


def _register_env_override(spec: _LucidEnvOverride, *, force: bool) -> List[str]:
    """Mirror one upstream TextArena env under a LucidGym-owned ID."""
    assert ta_registration is not None

    existing = _discover_variant_ids(spec.lucid_env_id)
    if existing and not force:
        return existing

    if existing:
        for env_id in existing:
            ta_registration.ENV_REGISTRY.pop(env_id, None)

    kwargs: Dict[str, Any] = dict(spec.kwargs)
    wrappers: WrapperSpec | None = spec.wrappers

    if spec.base_env_id is not None:
        base_spec = _get_env_spec(spec.base_env_id)
        inherited_kwargs = dict(base_spec.kwargs)
        inherited_kwargs.update(kwargs)
        kwargs = inherited_kwargs

        base_wrappers = _collect_wrapper_variants(spec.base_env_id)
        if wrappers is None:
            wrappers = base_wrappers

    if wrappers is None:
        raise ValueError(
            f"Env override '{spec.lucid_env_id}' must define wrappers or specify base_env_id."
        )

    ta_registration.register_with_versions(
        id=spec.lucid_env_id,
        entry_point=spec.entry_point,
        wrappers=wrappers,
        **kwargs,
    )

    return _discover_variant_ids(spec.lucid_env_id)


def register_lucidgym_textarena_envs(force: bool = False) -> List[str]:
    """
    Register LucidGym's TextArena env variants inside TextArena's registry.

    Args:
        force: When True, existing LucidGym entries are removed before re-registering.

    Returns:
        The sorted list of LucidGym env IDs that are now present in the registry.
    """

    _require_textarena_registry()
    _ensure_base_envs_loaded()
    assert ta_registration is not None

    registered_ids: List[str] = []
    for spec in _LUCID_TEXTARENA_ENV_OVERRIDES:
        registered_ids.extend(_register_env_override(spec, force=force))

    # Sort to keep responses deterministic regardless of registration order.
    return sorted(registered_ids)


__all__ = ["register_lucidgym_textarena_envs"]
