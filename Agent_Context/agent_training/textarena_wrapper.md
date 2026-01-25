# TextArena Wrapper Reference

This document summarizes every file under `textarena/wrappers`, clarifying what each wrapper does and how it differs from its neighbors.

## Top-Level Package

### textarena/wrappers/__init__.py
- Re-exports the most commonly used wrappers (simple renderer, action, and observation helpers) so callers can `from textarena.wrappers import ...` without drilling into subpackages.
- Acts purely as a convenience surface; it does not add new behavior.

## ActionWrappers

### ActionWrappers/action_formatting_wrapper.py — `ActionFormattingWrapper`
- Normalizes free-form agent text by wrapping it in square brackets when missing.
- Keeps semantics intact; it only touches formatting.
- Differs from the clipper wrappers because it never truncates or validates length—its sole job is enforcing the bracket contract some environments expect.

### ActionWrappers/clip_action_wrapper.py — `ClipWordsActionWrapper`
- Splits an action into whitespace-delimited tokens and keeps only the first *N* words.
- Useful when a prompt budget is counted in words; preserves original word order and truncates the tail.
- Unlike the character clipper, it respects word boundaries and leaves short actions untouched.

### ActionWrappers/clip_action_wrapper.py — `ClipCharactersActionWrapper`
- Limits the total character count (default 1,000) and slices from the end if the action is too long.
- Complementary to the word clipper: use it when transport or game rules care about characters rather than tokens.
- Different from `ClipWordsActionWrapper` in that it can cut through words and keeps the *last* portion of the text (useful when the answer payload is at the end).

## ObservationWrappers

### ObservationWrappers/llm_observation_wrapper.py — `LLMObservationWrapper`
- Builds a running chat transcript per player, naming senders via the environment’s `role_mapping` (or `GAME`).
- Returns a single string aggregated from all observations so far, making the stream LLM-friendly.
- Serves as the base behavior other observation wrappers specialize.

### ... same file — `DiplomacyObservationWrapper`
- Extends `LLMObservationWrapper` but formats output through `env.get_prompt`, which expects the current history.
- Drops the very first observation (usually the scenario prompt) when constructing the history so prompts are not duplicated.
- Purpose-built for Diplomacy-style games where the environment owns the exact prompt template.

### ... same file — `FirstLastObservationWrapper`
- Stores all observations but only surfaces the first (initial prompt) and the most recent message, followed by `"Next Action:"`.
- Helps long-horizon games stay within context limits by showing only the setup plus the latest update; unlike `LLMObservationWrapper`, it hides the middle of the conversation.

### ... same file — `GameBoardObservationWrapper`
- Requires `ObservationType.PROMPT` and the latest `ObservationType.GAME_BOARD`; raises if either is missing.
- Returns the original prompt plus the most recent board snapshot, making it ideal when the visual state matters more than textual chatter.
- Differs from `FirstLastObservationWrapper` by focusing on board state (ObservationType filtering) rather than chronological messages.

### ... same file — `GameMessagesObservationWrapper`
- Concatenates every message that is not a `PLAYER_ACTION` or `GAME_ADMIN`, so players see system/game updates without echoing their own actions.
- Provides only text; no board handling.
- Complementary to `GameBoardObservationWrapper`: one favors narration, the other structural board info.

### ... same file — `GameMessagesAndCurrentBoardObservationWrapper`
- Combines the prompt, filtered game messages (same exclusions as above), and the latest board state into one block.
- Maintains `self.full_observations` to pull whichever board entry arrived last and validates that both PROMPT and GAME_BOARD exist.
- Contains a duplicated `observation` method definition, but both copies perform the same logic.
- Use when agents need both narration and the current board without player action echoes.

### ... same file — `SingleTurnObservationWrapper`
- Ignores history entirely and returns only `observation[0][1]`, i.e., the text of the current event.
- Minimal wrapper that fits single-step tasks or environments that already package context per step.

### ... same file — `SettlersOfCatanObservationWrapper`
- Tracks all observations and renders them as `[Sender]\tmessage` lines, appending only the most recent board at the end.
- Ensures the last `ObservationType.GAME_BOARD` survives even if multiple boards were emitted (searches for final index before rendering).
- Contains a duplicated `observation` method (identical bodies), but behavior effectively mirrors `LLMObservationWrapper` with tailored formatting.
- Tailored for Settlers of Catan where alternating chat plus board state is key.

### ObservationWrappers/classical_reasoning_eval_observation_wrapper.py — `ClassicalReasoningEvalsObservationWrapper`
- Frames tasks for classical reasoning evals with a fixed system prefix: instructs the agent to reason step by step and puts the latest user query inside `<｜User｜>...` tokens.
- Stores all observations but only uses the newest entry when producing the formatted prompt.
- Unlike the other observation wrappers, it returns an empty string when `observation is None`, signaling that there is no cached view to reuse yet.

### ObservationWrappers/__init__.py
- Re-exports the observation wrappers listed above so callers can import them from `textarena.wrappers.ObservationWrappers` without referencing individual files.
- Also documents (via comments) wrappers that are planned but not yet implemented.

## RenderWrappers

### RenderWrappers/__init__.py
- Currently exposes only `SimpleRenderWrapper` under the render-wrappers namespace.

### RenderWrappers/SimpleRenderWrapper/render.py — `SimpleRenderWrapper`
- Uses `rich` to render either the board, chat panes, or both depending on `render_mode` (`standard`, `board`, `chat`, `multi`).
- Collects logs from `env.state.logs`, groups them by player, and truncates messages based on terminal dimensions so the layout stays within the screen.
- In `multi` mode it shows the board on top and the textual action that was just taken in a wide panel, letting viewers see their latest command alongside the board.
- `reset` enforces that `standard`/`chat` modes work only for two-player games and auto-fills missing player names from `role_mapping`.

### RenderWrappers/SimpleRenderWrapper/render copy.py — `SimpleRenderWrapper` (alternate draft)
- Duplicates most of `render.py` but differs in `multi` mode: instead of echoing the submitted `action`, it inspects `logs` to find the last non-game message and displays that entry.
- Appears to be an experimental or backup version (not imported anywhere); useful to know if you prefer multi-pane chat to reflect actual logs rather than the current action string.

## TrainingWrappers

### TrainingWrappers/check_if_valid.py
- Placeholder for a future training-time move validator; currently only contains a TODO comment about deep-copying state and verifying legality.

### TrainingWrappers/__init__.py
- Empty file so the `TrainingWrappers` directory is recognized as a package; no runtime behavior yet.

