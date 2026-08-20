from __future__ import annotations
 
import contextlib
import inspect
import io
import logging
import time
from typing import Any
 
from gepa.lm import LM
 
 
logger = logging.getLogger(__name__)
 
_RESERVED_NAMES = {"finish", "__builtins__"}
 
 
class Finished(Exception):
    """Raised by the injected `finish(result)` to end the episode, carrying
    the object the agent is submitting as its final answer."""
 
    def __init__(self, value: Any):
        self.value = value
 
 
class LimitsExceeded(Exception):
    """Raised internally when a configured step/cost/time budget is
    exhausted. Caught inside Agent.__call__ -- callers never see this
    exception, they see `Agent.stop_reason` and get something usable back
    either way."""
 
 
def _finish(result: Any) -> None:
    raise Finished(result)
 
 
def _safe_str(obj: Any, max_len: int = 1000) -> str:
    """str() that can't crash the harness and can't blow up the context
    window, no matter what `obj` is. str() falls back to __repr__ when
    __str__ isn't overridden, so this is the one call that covers both.
    Never returns more than `max_len` characters, suffix included."""
    try:
        text = str(obj)
    except Exception as exc:  # noqa: BLE001 - obj's __str__/__repr__ is arbitrary user code
        text = f"<str() failed: {exc!r}>"
    if len(text) <= max_len:
        return text
    suffix = "..."
    return text[: max(max_len - len(suffix), 0)] + suffix
 
 
def _has_custom_str(cls: type) -> bool:
    """True if cls overrides __repr__ or __str__ beyond the default from
    `object`. Used for two different things: (1) whether showing a preview
    is informative at all, and (2) whether string-diffing that preview
    between turns can be trusted -- an object with the default repr can
    mutate in place (e.g. a stateful bash-tool-like env) without its
    preview changing at all, since the default repr is just an address."""
    return cls.__repr__ is not object.__repr__ or cls.__str__ is not object.__str__
 
 
def _has_public_api(cls: type) -> bool:
    try:
        return any(
            not name.startswith("_")
            for name, _ in inspect.getmembers(cls, predicate=inspect.isroutine)
        )
    except Exception:  # noqa: BLE001
        return False
 
 
def render_api_stub(env_cls: type, minimal: bool = False) -> str:
    """Turn a type's public methods into a .pyi-style stub for the system
    prompt. Works for arbitrary types -- dataclasses, builtins (str, list),
    a thin wrapper around a persistent bash session, a graph object,
    anything -- and never raises: a broken or exotic class degrades to a
    stub that says introspection failed, rather than crashing the agent."""
    try:
        if minimal:
            return f"class {env_cls.__name__}:\n    ..."
 
        lines = [f"class {env_cls.__name__}:"]
        found = False
        for name, member in inspect.getmembers(env_cls, predicate=inspect.isroutine):
            if name.startswith("_"):
                continue
            found = True
            try:
                sig = inspect.signature(member)
            except (TypeError, ValueError):
                sig = "(...)"
            try:
                doc = inspect.getdoc(member)
            except Exception:  # noqa: BLE001
                doc = None
            lines.append(f"    def {name}{sig}:")
            if doc:
                lines.append(f'        """{doc.strip().splitlines()[0]}"""')
            lines.append("        ...")
            lines.append("")
        if not found:
            lines.append("    ...  # no public methods found")
        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001 - introspection must never crash the agent
        logger.warning("Failed to introspect %r: %r", env_cls, exc)
        name = getattr(env_cls, "__name__", "Environment")
        return f"class {name}:\n    ...  # introspection failed: {exc!r}"
 
 
 
class Agent:
    """A lightweight Agent that drives a Python namespace by generating code."""
 
    def __init__(
        self,
        model: str,
        step_limit: int = 0,
        cost_limit: float = 2.0,
        wall_time_limit_seconds: int = 0,
        **kwargs: Any,
    ):
        self.model = model
        self.lm = LM(self.model, **kwargs)
        self.step_limit = step_limit
        self.wall_time_limit_seconds = wall_time_limit_seconds
        self.cost_limit = cost_limit
      
        self._n_calls: int = 0
        self._start_time: float | None = None
        self._history: list[dict[str, Any]] | None = None
        self._tracked_names: set[str] | None = None
 
        # Always mirrors the last state successfully reached, so even a
        # stop condition this version didn't anticipate leaves the caller
        # able to recover `agent.state`.
        self.state: dict[str, Any] = {}
        # The object passed to finish(...), once the episode completes
        # normally. None until then (and stays None for other stop reasons).
        self.result: Any = None
        self.stop_reason: str | None = None
 
    # -- state bookkeeping -------------------------------------------------
 
    def _public_state(self, namespace: dict[str, Any]) -> dict[str, Any]:
        items = {k: v for k, v in namespace.items() if k not in _RESERVED_NAMES}
        if self._tracked_names is not None:
            items = {k: v for k, v in items.items() if k in self._tracked_names}
        return items
 
    def _snapshot(self, namespace: dict[str, Any]) -> dict[str, tuple[type, str]]:
        return {
            name: (type(value), _safe_str(value, 1000))
            for name, value in self._public_state(namespace).items()
        }
 
    def _describe_var(self, name: str, cls: type, preview: str, full_api: bool) -> str:
        line = f"- `{name}`: {cls.__name__} = {preview}"
        if full_api and _has_public_api(cls):
            stub = render_api_stub(cls)
            indented = "\n".join(("    " + l if l else l) for l in stub.splitlines())
            line += f"\n  Public API:\n{indented}"
        return line
 
    def _describe_state(self, namespace: dict[str, Any]) -> str:
        snap = self._snapshot(namespace)
        if not snap:
            return "(no variables provided)"
        return "\n".join(
            self._describe_var(name, cls, preview, full_api=True)
            for name, (cls, preview) in snap.items()
        )
 
    def _diff_description(self, prev_snap: dict[str, tuple[type, str]], namespace: dict[str, Any]) -> str:
        curr_snap = self._snapshot(namespace)
        lines = []
        for name, (cls, preview) in curr_snap.items():
            if name not in prev_snap:
                lines.append(self._describe_var(name, cls, preview, full_api=True))
                continue
            prev_cls, prev_preview = prev_snap[name]
            type_changed = cls is not prev_cls
            # An object with the default repr can mutate in place without
            # its preview changing at all (default repr is just an id) --
            # always resurface those rather than risk hiding a real change.
            maybe_hidden_mutation = not _has_custom_str(cls)
            if type_changed or preview != prev_preview or maybe_hidden_mutation:
                lines.append(self._describe_var(name, cls, preview, full_api=type_changed))
        for name in prev_snap:
            if name not in curr_snap:
                lines.append(f"- `{name}`: removed")
        return "\n".join(lines) if lines else "(no variables changed)"
 
    # -- prompting -------------------------------------------------------
 
    def _init_messages(self, namespace: dict[str, Any], prompt: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = list(prompt)
 
        system_prefix = (
            messages[0]["content"] + "\n### Environment\n"
            if messages and messages[0].get("role") == "system"
            else ""
        )
        state_desc = self._describe_state(namespace)
 
        system = (
            f"{system_prefix}"
            "You solve tasks by writing Python code. The following variables\n"
            "are already defined in your namespace:\n\n"
            f"{state_desc}\n\n"
            "You may reassign these variables, mutate objects in place, call\n"
            "their public methods, or define new variables -- whatever\n"
            "advances the task. Imports and helper variables persist across\n"
            "turns, just like a REPL.\n"
            "You may think step by step in plain text before you act, but your\n"
            "response must end with exactly one ```python fenced code block --\n"
            "never more than one, and this applies to every turn, including\n"
            "the turn where you call finish(...).\n"
            "Use `print(...)` to inspect values.\n"
            "Call `finish(result)` once the task is complete, passing the\n"
            "object that represents your final answer -- that object,\n"
            "exactly as passed, is what gets returned."
        )
 
        if messages and messages[0].get("role") == "system":
            messages = [{"role": "system", "content": system}] + messages[1:]
        else:
            messages = [{"role": "system", "content": system}] + messages
        return messages
 
    def _get_usage_info_string(self) -> str:
        try:
            steps = f"Steps: {self._n_calls}/{self.step_limit}, " if self.step_limit else ""
            return (
                f"{steps}"
                f"Tokens: {self.lm.total_tokens_in + self.lm.total_tokens_out}, "
                f"Budget: ${self.lm.total_cost:.4f}/${self.cost_limit:.2f}"
            )
        except Exception:  # pragma: no cover
            logger.debug("Failed to compute usage info", exc_info=True)
            return f"Steps: {self._n_calls}/{self.step_limit}"
 
    # -- execution ---------------------------------------------------------
 
    def _execute(self, namespace: dict[str, Any], code: str) -> tuple[str, bool, Any]:
        """Run generated code exactly as written against the persistent
        namespace -- no AST checks, no builtins restriction, no import
        allowlist; isolation is the caller-supplied variables' job, not
        the agent's. `namespace` is mutated in place by exec(), including
        up to the point an exception (or Finished) is raised, so state
        before the failure is never lost."""
        stdout = io.StringIO()
        finished = False
        finish_value: Any = None
        try:
            with contextlib.redirect_stdout(stdout):
                exec(code, namespace)
        except Finished as exc:
            finished = True
            finish_value = exc.value
        except Exception as exc:  # noqa: BLE001 - feed runtime errors back to the model
            observation = f"Execution error: {exc!r}\n{stdout.getvalue()}"
            return self._truncate(observation), False, None
 
        output = stdout.getvalue()
        observation = output if output else "[no output]"
        return self._truncate(observation), finished, finish_value
 
    def _truncate(self, text: str) -> str:
        if len(text) <= 1000:
            return text
        suffix = "... [truncated]"
        return text[: max(1000 - len(suffix), 0)] + suffix
 
    # -- limits & code parsing --------------------------------------------
 
    def _check_limits(self) -> None:
        if 0 < self.step_limit <= self._n_calls:
            raise LimitsExceeded(f"step_limit ({self.step_limit}) exceeded")
        try:
            cost = self.lm.total_cost
        except AttributeError:
            cost = None  # LM wrapper doesn't track cost -- skip this check rather than crash
        if cost is not None and 0 < self.cost_limit <= cost:
            raise LimitsExceeded(f"cost_limit (${self.cost_limit}) exceeded")
        if (
            self._start_time is not None
            and 0 < self.wall_time_limit_seconds <= time.time() - self._start_time
        ):
            raise LimitsExceeded(f"wall_time_limit_seconds ({self.wall_time_limit_seconds}) exceeded")
 
    def _extract_code(self, response_text: str) -> str | None:
        if "```python" not in response_text:
            return None
        _, tail = response_text.rsplit("```python", 1)
        if "```" not in tail:
            return None
        return tail.split("```", 1)[0]
 
    @staticmethod
    def _has_open_fence(response_text: str) -> bool:
        return "```python" in response_text
 
    # -- model call ----------------------------------------------------------
 
    def _add_message(self, message: dict[str, Any]) -> None:
        assert self._history is not None, "History is not initialized yet"
        print("> %s: %s" % (message.get("role", "user").upper(), message["content"]))
        self._history.append(message)
 
    # -- main loop -----------------------------------------------------------
 
    def __call__(self, prompt: str | list[dict[str, Any]], **state: Any) -> Any:
        namespace: dict[str, Any] = dict(state)
        namespace["finish"] = namespace.get("finish", _finish)
 
        self._tracked_names = set(state.keys())
        self._history = []
        for message in self._init_messages(namespace, prompt):
            self._add_message(message)
        self._n_calls = 0
        self._start_time = time.time()
        self.stop_reason = None
        self.result = None
        prev_snapshot = self._snapshot(namespace)
        self.state = self._public_state(namespace)
 
        while True:
            try:
                self._check_limits()
            except LimitsExceeded as exc:
                logger.info("Stopping: %s", exc)
                self.stop_reason = str(exc)
                break
 
            response_text = self.lm(self._history)
            if response_text is None:
                self.stop_reason = "lm_failure"
                break
            self._n_calls += 1
            self._add_message({"role": "assistant", "content": response_text})
 
            code = self._extract_code(response_text)
            if code is None:
                if self._has_open_fence(response_text):
                    feedback = "Your ```python code block was not closed with ```. Please resend a single, complete code block."
                else:
                    feedback = "Your response does not include a code block enclosed in ```python and ```. Call `finish(result)` once you completed the task."

                self._add_message({
                    "role": "user",
                    "content": feedback,
                })
                continue

 
            observation, finished, finish_value = self._execute(namespace, code)
            self.state = self._public_state(namespace)
            if finished:
                self.stop_reason = "finished"
                self.result = finish_value
                return finish_value
 
            usage_info = self._get_usage_info_string()
            diff = self._diff_description(prev_snapshot, namespace)
            prev_snapshot = self._snapshot(namespace)
            self._add_message({
                "role": "user",
                "content": (
                    f"Output:\n`{observation}`\n\n"
                    f"```text\nState changes:\n\n{diff}\n\n{usage_info}\n```\n"
                ),
            })
 
        # Reached only via a non-normal stop (limits/lm failure/no code) --
        # finish() was never called, so there's no submitted result. Return
        # the namespace as a best-effort fallback rather than nothing.
        return self._public_state(namespace)