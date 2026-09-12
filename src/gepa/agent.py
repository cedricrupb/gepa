from __future__ import annotations

import ast
import contextlib
import inspect
import io
import logging
import re
import reprlib
import signal
import threading
import time
import traceback
from typing import Any
 
from gepa.proposer.reflective_mutation.base import LanguageModel
 
 
logger = logging.getLogger(__name__)


class Finished(Exception):
    """Raised by the injected ``finish(result)`` to terminate an episode."""
 
    def __init__(self, value: Any):
        self.value = value

class CompactRequested(BaseException):
    """Raised by ``compact(summary)`` to ask the harness to rewrite history."""

    def __init__(self, summary: str):
        self.summary = summary
 

class StepTimeout(BaseException):
    """Raised internally when one Python action exceeds its execution budget."""
 
 
class LimitsExceeded(Exception):
    """Raised internally when a configured step/cost/time budget is exhausted."""
 
 
def _finish(result: Any) -> None:
    """Submit ``result`` as the final answer and terminate the current session."""
    raise Finished(result)


def _compact(summary: str) -> None:
    """Compact old transcript history using a concise continuation summary.

    Put durable task state in ``summary``: established facts, important named
    Python variables, completed work, unresolved issues, and the intended next
    step. Python objects themselves are not changed by compaction.
    """
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("compact(summary) requires a non-empty string summary")
    raise CompactRequested(summary.strip())


def make_litellm_llm(model_name, **kwargs) -> LanguageModel:
    from gepa.lm import LM
    return LM(model_name, **kwargs)


AGENT_PROMPT = """
You are a helpful assistant that solves tasks by writing Python code in a persistent Python REPL, observing the result, and reacting.

Use Python to interact with the environment and solve the user's task. You may use available objects and functions, create temporary helpers, and interact with the computer through Python.

## Turn format

Each turn, write exactly one executable Python code block, fenced with the delimiter alone on its own line:

```python
# code
```

The environment executes the block and returns its stdout, result, or traceback as your next observation.

## State

<initial_state>
{{INITIAL_STATE}}
</initial_state>

This lists what is bound in the namespace at the start of the session. Their docstrings describe what the available objects, functions, and tools do; inspect them when needed.

Anything else you define, including variables and helper functions, persists across turns for the rest of the session.

In a long session, older outputs in your visible history may be folded down to short stubs to save context. The underlying Python state is unaffected. If you need information from an earlier step again, inspect the corresponding Python object rather than relying on your memory of its previous representation.

Prefer assigning results you may need later to clearly named variables rather than only printing them, since Python state survives history folding.

## Finishing

`finish(result)` submits `result` to the user and ends the session. Nothing after it runs, and the namespace is discarded unless something in the initial state is explicitly persistent beyond the session.

Call `finish(...)` alone in its own code block.

Only finish once the information needed for the result has already been established through prior observations. Do not combine an unobserved environment interaction with `finish(...)`.

`result` may be any Python object. Return whatever best represents what the user actually needs, since nothing else from the session will be visible to them.
""".strip()

_PYTHON_BLOCK_RE = re.compile(
    r"^[ \t]*```python[ \t]*\r?\n(?P<code>.*?)^[ \t]*```[ \t]*(?:\r?\n|\Z)",
    re.MULTILINE | re.DOTALL,
)
_PYTHON_OPEN_RE = re.compile(r"^[ \t]*```python[ \t]*$", re.MULTILINE)


class Agent:
    """A lightweight CodeReAct agent backed by a persistent Python namespace."""
 
    def __init__(
        self,
        lm: LanguageModel | str,
        step_limit: int = 0,
        cost_limit: float = 2.0,
        wall_time_limit_seconds: int = 0,
        max_observation_chars: int = 30_000,
        max_environment_message_chars: int = 60_000,
        keep_recent_environment_outputs: int = 6,
        environment_pruning_threshold: int = 100_000,
        compaction_threshold_tokens: int = 175_000,
        compaction_warn_fraction: float = 0.85,
        compaction_keep_turns: int = 4,
        step_timeout_seconds: float = 60.0,
        no_out: bool = False,
        **kwargs: Any,
    ):
        if isinstance(lm, str):
            lm = make_litellm_llm(lm, **kwargs)
        else:
            from gepa.lm import TrackingLM

            if not hasattr(lm, "total_cost"):
                lm = TrackingLM(lm)

        self.lm = lm
        self.step_limit = step_limit
        self.wall_time_limit_seconds = wall_time_limit_seconds
        self.cost_limit = cost_limit
        self.max_observation_chars = max_observation_chars
        self.max_environment_message_chars = max_environment_message_chars
        self.environment_pruning_threshold = environment_pruning_threshold
        self.keep_recent_environment_outputs = keep_recent_environment_outputs
        self.compaction_threshold_tokens = compaction_threshold_tokens
        self.compaction_warn_fraction = compaction_warn_fraction
        self.compaction_keep_turns = compaction_keep_turns
        self.step_timeout_seconds = step_timeout_seconds
        self.no_out = no_out
        
        self._n_calls: int = 0
        self._start_time: float | None = None
        self._history: list[dict[str, Any]] | None = None
        self._history_kinds: list[str] | None = None
        self._history_prefix_len: int = 0
        self._tracked_names: set[str] | None = None
        self._last_input_tokens: int | None = None

        self._start_tokens_in = 0
        self._start_tokens_out = 0
        self._start_cost = 0.0
    
        # Mirrors the last state successfully reached so callers can recover
        # useful work even when an episode stops because of a budget or error.
        self.state: dict[str, Any] = {}
        self.result: Any = None
        self.stop_reason: str | None = None

    @property
    def total_cost(self) -> float:
        return getattr(self.lm, "total_cost", 0.0) - self._start_cost

    @property
    def total_tokens_in(self) -> int:
        return getattr(self.lm, "total_tokens_in", 0) - self._start_tokens_in

    @property
    def total_tokens_out(self) -> int:
        return getattr(self.lm, "total_tokens_out", 0) - self._start_tokens_out

    @property
    def history(self) -> list[dict[str, Any]]:
        return self._history or []

    # -- limits & code parsing --------------------------------------------
     
    def _check_limits(self) -> None:
        if 0 < self.step_limit <= self._n_calls:
            raise LimitsExceeded(f"step_limit ({self.step_limit}) exceeded")
        try:
            cost = self.total_cost
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
        """Return the single complete fenced Python block, otherwise ``None``.

        Prose outside the block is tolerated.  Multiple Python blocks are not:
        the turn protocol deliberately has one executable action per model turn.
        """
        if len(_PYTHON_OPEN_RE.findall(response_text)) != 1:
            return None
        match = _PYTHON_BLOCK_RE.search(response_text)
        if match is None:
            return None
        return match.group("code")
    
    def _code_feedback(self, response_text: str) -> str:
        n_open = len(_PYTHON_OPEN_RE.findall(response_text))
        n_closed = len(_PYTHON_BLOCK_RE.findall(response_text))
        if n_open == 0:
            return (
                "Your response does not include a Python code block fenced with "
                "```python and ```. Send exactly one executable Python block."
            )
        if n_open > 1:
            return "Send exactly one ```python code block per turn."
        if n_closed == 0:
            return (
                "Your ```python code block was not closed with ``` on its own line. "
                "Resend one complete block."
            )
        return "Send exactly one complete executable ```python code block per turn."

    # -- prompt / namespace description ----------------------------------

    @staticmethod
    def _type_name(value: Any) -> str:
        tp = type(value)
        if tp.__module__ == "builtins":
            return tp.__qualname__
        return f"{tp.__module__}.{tp.__qualname__}"

    @staticmethod
    def _short_doc(value: Any, max_chars: int = 700) -> str:
        try:
            doc = inspect.getdoc(value) or ""
        except BaseException:
            return ""
        doc = " ".join(doc.split())
        if len(doc) <= max_chars:
            return doc
        return doc[: max_chars - 1].rstrip() + "…"

    @staticmethod
    def _signature(value: Any) -> str | None:
        if not callable(value):
            return None
        try:
            return str(inspect.signature(value))
        except (TypeError, ValueError, BaseException):
            # Some extension/proxy callables do not expose a signature and
            # some user objects implement hostile/introspective hooks.
            return None

    @staticmethod
    def _summary_repr(value: Any, max_chars: int = 700) -> str:
        """Return a bounded representation suitable for the initial manifest.

        For user-defined objects we intentionally call their ``__repr__``: an
        environment object can use it to advertise useful current state.
        Built-in containers go through ``reprlib`` to avoid constructing an
        enormous representation just for the prompt.
        """
        try:
            if type(value).__module__ == "builtins" and isinstance(
                value, (list, tuple, set, frozenset, dict)
            ):
                printer = reprlib.Repr()
                printer.maxlevel = 3
                printer.maxlist = 8
                printer.maxtuple = 8
                printer.maxset = 8
                printer.maxfrozenset = 8
                printer.maxdict = 8
                printer.maxstring = 240
                printer.maxother = 240
                text = printer.repr(value)
            else:
                text = repr(value)
        except BaseException as exc:
            return f"<repr failed: {type(exc).__name__}: {exc}>"

        text = " ".join(str(text).split())
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 1].rstrip() + "…"

    def _public_methods(
        self,
        value: Any,
        max_methods: int = 12,
        max_doc_chars: int = 240,
    ) -> tuple[list[str], int]:
        """Describe public methods without eagerly walking instance properties.

        Static class inspection avoids executing arbitrary property getters.
        Binding a method can still fail for exotic descriptors, so each member
        is isolated behind exception handling.
        """
        methods: list[str] = []
        total = 0
        try:
            members = inspect.getmembers_static(type(value))
        except BaseException:
            return methods, total

        for name, static_member in members:
            if name.startswith("_") or isinstance(static_member, property):
                continue
            try:
                bound = getattr(value, name)
            except BaseException:
                continue
            if not callable(bound):
                continue
            total += 1
            if len(methods) >= max_methods:
                continue

            signature = self._signature(bound) or "(...)"
            line = f"    - `{name}{signature}`"
            doc = self._short_doc(bound, max_chars=max_doc_chars)
            if doc:
                line += f": {doc}"
            methods.append(line)

        return methods, total

    def _describe_namespace(self, namespace: dict[str, Any]) -> str:
        lines: list[str] = []
        for name in sorted(namespace):
            if name.startswith("__"):
                continue
            value = namespace[name]
            signature = self._signature(value)
            if signature is not None:
                lines.append(f"- `{name}{signature}`")
            else:
                lines.append(f"- `{name}`: {self._type_name(value)}")

            lines.append(f"  repr: {self._summary_repr(value)}")

            doc = self._short_doc(value)
            if doc:
                lines.append(f"  {doc}")

            # Functions already expose their interaction surface through their
            # own signature. For object-like capabilities, advertise a small
            # bounded public API as well.
            if signature is None:
                methods, total = self._public_methods(value)
                if methods:
                    lines.append("  public methods:")
                    lines.extend(methods)
                    if total > len(methods):
                        lines.append(f"    - … ({total - len(methods)} more)")

        return "\n".join(lines) if lines else "(no initial names)"

    # -- model call ----------------------------------------------------------
     
    def _init_messages(
        self,
        namespace: dict[str, Any],
        prompt: str | list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        initial_state = self._describe_namespace(namespace)
        system_prompt = AGENT_PROMPT.replace("{{INITIAL_STATE}}", initial_state)

        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            # Copy each dict because callers often reuse the input transcript.
            messages = [dict(message) for message in prompt]

        return [{"role": "system", "content": system_prompt}, *messages]


    def _add_message(self, message: dict[str, Any], *, kind: str = "other") -> None:
        assert self._history is not None, "History is not initialized yet"
        assert self._history_kinds is not None, "History kinds are not initialized yet"

        message = dict(message)
        # Only harness-generated environment messages are hard-bounded here.
        # System/user instructions are not safely reconstructable and must not
        # be silently truncated by the harness.
        if kind.startswith(("observation:", "feedback:")):
            message["content"] = self._truncate(
                str(message.get("content", "")),
                self.max_environment_message_chars,
            )

        if not self.no_out:
            print("> %s: %s" % (message.get("role", "user").upper(), message["content"]))
        self._history.append(message)
        self._history_kinds.append(kind)

    def _fold_old_environment_outputs(self) -> None:
        """Replace old environment observations with tiny deterministic stubs."""
        assert self._history is not None and self._history_kinds is not None
        if self._last_input_tokens and self._last_input_tokens < self.environment_pruning_threshold:
            return

        keep = self.keep_recent_environment_outputs
        if keep < 0:
            return

        full = [
            i for i, kind in enumerate(self._history_kinds)
            if kind.startswith("observation:")
        ]
        if len(full) <= keep:
            return

        for i in full[:-keep] if keep else full:
            kind = self._history_kinds[i]
            step = kind.partition(":")[2] or "?"
            self._history[i] = {
                "role": "user",
                "content": (
                    f"[Older environment output from step {step} was folded to save "
                    "context. Python state is unchanged; re-inspect named variables, "
                    "objects, or files if you need the details again.]"
                ),
            }
            self._history_kinds[i] = f"folded:{step}"

    def _apply_compaction(self, summary: str) -> None:
        """Keep the initial task, a semantic summary, and a few recent turns."""
        assert self._history is not None and self._history_kinds is not None

        prefix = [dict(m) for m in self._history[: self._history_prefix_len]]
        prefix_kinds = list(self._history_kinds[: self._history_prefix_len])
        runtime = list(zip(
            self._history[self._history_prefix_len :],
            self._history_kinds[self._history_prefix_len :],
        ))

        # The administrative compact(summary) action itself adds no useful
        # continuation context; remove it from the rebuilt transcript.
        if runtime and runtime[-1][1] == "assistant":
            runtime.pop()

        n_recent = max(0, 2 * self.compaction_keep_turns)
        recent = runtime[-n_recent:] if n_recent else []
        summary_message = {
            "role": "user",
            "content": (
                "[Compacted session summary. Treat this as context from earlier "
                "turns; the live Python namespace is unchanged.]\n\n" + summary.strip()
            ),
        }

        self._history = prefix + [summary_message] + [dict(m) for m, _ in recent]
        self._history_kinds = (
            prefix_kinds + ["compaction_summary"] + [kind for _, kind in recent]
        )
        self._last_input_tokens = None


    # -- execution --------------------------------------------------------

    @staticmethod
    def _is_plain_finish(tree: ast.Module) -> bool:
        if len(tree.body) != 1:
            return False
        stmt = tree.body[0]
        if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
            return False
        call = stmt.value
        return (
            isinstance(call.func, ast.Name)
            and call.func.id == "finish"
            and len(call.args) == 1
            and not call.keywords
        )

    @staticmethod
    def _mentions_finish(tree: ast.AST) -> bool:
        return any(isinstance(node, ast.Name) and node.id == "finish" for node in ast.walk(tree))

    @staticmethod
    def _is_plain_compact(tree: ast.Module) -> bool:
        if len(tree.body) != 1:
            return False
        stmt = tree.body[0]
        if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
            return False
        call = stmt.value
        return (
            isinstance(call.func, ast.Name)
            and call.func.id == "compact"
            and len(call.args) == 1
            and not call.keywords
        )

    @staticmethod
    def _mentions_compact(tree: ast.AST) -> bool:
        return any(isinstance(node, ast.Name) and node.id == "compact" for node in ast.walk(tree))

    @contextlib.contextmanager
    def _step_timeout(self):
        seconds = self.step_timeout_seconds
        if seconds <= 0:
            yield
            return

        # SIGALRM is the only practical way to interrupt arbitrary code while
        # keeping one in-process persistent namespace.  It is Unix/main-thread
        # only; on unsupported hosts we execute without a hard step timeout.
        if (
            not hasattr(signal, "SIGALRM")
            or not hasattr(signal, "setitimer")
            or threading.current_thread() is not threading.main_thread()
        ):
            logger.warning(
                "step_timeout_seconds is not enforceable outside a Unix main thread"
            )
            yield
            return

        old_handler = signal.getsignal(signal.SIGALRM)
        old_timer = signal.getitimer(signal.ITIMER_REAL)
        started = time.monotonic()

        def on_alarm(signum, frame):
            raise StepTimeout(
                f"Python action exceeded step_timeout_seconds ({seconds:g})"
            )

        signal.signal(signal.SIGALRM, on_alarm)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)
            if old_timer != (0.0, 0.0):
                elapsed = time.monotonic() - started
                remaining = max(0.0, old_timer[0] - elapsed)
                if remaining > 0 or old_timer[1] > 0:
                    signal.setitimer(signal.ITIMER_REAL, remaining, old_timer[1])

    @staticmethod
    def _format_exception(exc: BaseException) -> str:
        return "".join(traceback.TracebackException.from_exception(exc).format())

    @staticmethod
    def _result_repr(value: Any) -> str:
        printer = reprlib.Repr()
        printer.maxlevel = 6
        printer.maxlist = 20
        printer.maxtuple = 20
        printer.maxset = 20
        printer.maxfrozenset = 20
        printer.maxdict = 20
        printer.maxstring = 2000
        printer.maxother = 2000
        try:
            return printer.repr(value)
        except BaseException as exc:
            return f"<repr failed: {type(exc).__name__}: {exc}>"

    def _execute(
        self,
        namespace: dict[str, Any],
        code: str,
        finish_callable : Any = _finish,
    ) -> tuple[str, bool, Any]:
        """Execute one action with REPL-like final-expression semantics."""
        stdout = io.StringIO()
        stderr = io.StringIO()

        # The harness owns this primitive. Reinstalling it every turn also
        # recovers if earlier user code tried to shadow the name.
        namespace["finish"]  = finish_callable
        namespace["compact"] = _compact 
        namespace.setdefault("__name__", "__codereact__")

        try:
            tree = ast.parse(code, filename="<codereact>", mode="exec")
        except SyntaxError as exc:
            return self._format_exception(exc), False, None

        is_finish = self._is_plain_finish(tree)
        is_compact = self._is_plain_compact(tree)
        if self._mentions_finish(tree) and not is_finish:
            return (
                "ProtocolError: call finish(result) alone as the only statement "
                "in its Python block.",
                False,
                None,
            )

        if self._mentions_compact(tree) and not is_compact:
            return (
                "ProtocolError: call compact(summary) alone as the only statement "
                "in its Python block.",
                False,
                None,
            )

        result_sentinel = object()
        result: Any = result_sentinel

        try:
            with (
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
                self._step_timeout(),
            ):
                if is_finish or is_compact:
                    # Evaluate the call so a custom finish() may return feedback
                    # as the observation. The default finish() raises Finished.
                    expression = ast.Expression(tree.body[0].value)
                    result = eval(
                        compile(
                            ast.fix_missing_locations(expression),
                            "<codereact>",
                            "eval",
                        ),
                        namespace,
                        namespace,
                    )
                elif tree.body and isinstance(tree.body[-1], ast.Expr):
                    prefix = ast.Module(body=tree.body[:-1], type_ignores=tree.type_ignores)
                    if prefix.body:
                        exec(
                            compile(ast.fix_missing_locations(prefix), "<codereact>", "exec"),
                            namespace,
                            namespace,
                        )
                    expression = ast.Expression(tree.body[-1].value)
                    result = eval(
                        compile(
                            ast.fix_missing_locations(expression),
                            "<codereact>",
                            "eval",
                        ),
                        namespace,
                        namespace,
                    )
                else:
                    exec(compile(tree, "<codereact>", "exec"), namespace, namespace)
        except CompactRequested:
            namespace["compact"] = _compact
            namespace["finish"]  = finish_callable
            raise 
        except Finished as exc:
            namespace["finish"]  = finish_callable
            namespace["compact"] = _compact
            return self._compose_observation(stdout.getvalue(), stderr.getvalue()), True, exc.value
        except StepTimeout as exc:
            namespace["finish"]  = finish_callable
            namespace["compact"] = _compact
            observation = self._compose_observation(
                stdout.getvalue(),
                stderr.getvalue(),
                f"StepTimeout: {exc}",
            )
            return observation, False, None
        except BaseException as exc:
            namespace["finish"]  = finish_callable
            namespace["compact"] = _compact
            observation = self._compose_observation(
                stdout.getvalue(),
                stderr.getvalue(),
                self._format_exception(exc),
            )
            return observation, False, None
        finally:
            namespace["finish"]  = finish_callable
            namespace["compact"] = _compact

        result_text = None
        if result is not result_sentinel and result is not None:
            result_text = self._result_repr(result)
        return self._compose_observation(stdout.getvalue(), stderr.getvalue(), result_text), False, None

    @staticmethod
    def _compose_observation(
        stdout: str,
        stderr: str,
        final: str | None = None,
    ) -> str:
        parts: list[str] = []
        if stdout:
            parts.append(stdout.rstrip("\n"))
        if stderr:
            parts.append("stderr:\n" + stderr.rstrip("\n"))
        if final:
            parts.append(final.rstrip("\n"))
        return "\n".join(parts) if parts else "(no output)"

    # -- state snapshots / observations ----------------------------------

    @staticmethod
    def _is_public_name(name: str) -> bool:
        return not name.startswith("__") and name not in ["finish", "compact"]

    def _public_state(self, namespace: dict[str, Any]) -> dict[str, Any]:
        return {
            name: value
            for name, value in namespace.items()
            if self._is_public_name(name)
        }

    @staticmethod
    def _preview(value: Any) -> str:
        printer = reprlib.Repr()
        printer.maxlevel = 3
        printer.maxlist = 6
        printer.maxtuple = 6
        printer.maxset = 6
        printer.maxfrozenset = 6
        printer.maxdict = 6
        printer.maxstring = 240
        printer.maxother = 240
        try:
            return printer.repr(value)
        except BaseException as exc:
            return f"<repr failed: {type(exc).__name__}>"

    def _snapshot(self, namespace: dict[str, Any]) -> dict[str, tuple[int, str, str]]:
        """Take a cheap-ish diagnostic snapshot for user-visible state diffs.

        ``id`` catches rebinding and the compact repr catches many useful
        in-place mutations.  This is observability, not a correctness/security
        mechanism: arbitrary Python objects cannot be perfectly diffed.
        """
        return {
            name: (id(value), self._type_name(value), self._preview(value))
            for name, value in namespace.items()
            if self._is_public_name(name)
        }

    def _diff_description(
        self,
        previous: dict[str, tuple[int, str, str]],
        namespace: dict[str, Any],
    ) -> str:
        current = self._snapshot(namespace)
        lines: list[str] = []

        for name in sorted(previous.keys() - current.keys()):
            _, type_name, preview = previous[name]
            lines.append(f"- {name}: {type_name} = {preview}")

        for name in sorted(current.keys() - previous.keys()):
            _, type_name, preview = current[name]
            lines.append(f"+ {name}: {type_name} = {preview}")

        for name in sorted(previous.keys() & current.keys()):
            old_id, old_type, old_preview = previous[name]
            new_id, new_type, new_preview = current[name]
            if (old_id, old_type, old_preview) != (new_id, new_type, new_preview):
                if old_preview == new_preview and old_type == new_type:
                    lines.append(f"~ {name}: rebound ({new_type})")
                else:
                    lines.append(
                        f"~ {name}: {old_type} {old_preview} -> {new_type} {new_preview}"
                    )

        return "\n".join(lines) if lines else "(none)"

    def _truncate(self, text: str, limit: int = -1) -> str:
        if limit < 0:
            limit = self.max_observation_chars
        if len(text) <= limit:
            return text

        # Reserve enough room for a useful head and tail. For tiny configured
        # limits, a simple prefix is clearer than elaborate accounting.
        if limit < 120:
            return text[: max(0, limit - 1)] + "…"

        marker_template = "\n... <{} chars omitted> ...\n"
        marker = marker_template.format(0)
        available = limit - len(marker)
        head_len = int(available * 0.6)
        tail_len = available - head_len
        omitted = len(text) - head_len - tail_len
        marker = marker_template.format(omitted)

        # The actual digit count can change the marker length; adjust the head.
        overflow = head_len + tail_len + len(marker) - limit
        if overflow > 0:
            head_len = max(1, head_len - overflow)
            omitted = len(text) - head_len - tail_len
            marker = marker_template.format(omitted)

        return text[:head_len] + marker + text[-tail_len:]

    def _render_environment_message(
        self,
        observation: str,
        diff: str,
        usage_info: str,
    ) -> str:
        """Render one observation while preserving the status tail under a hard cap."""
        was_truncated = self.max_observation_chars > 0 and len(observation) > self.max_observation_chars
        rendered_observation = self._truncate(observation)
        label = " (middle truncated)" if was_truncated else ""

        status = f"Status:\n```text\n{usage_info}\n```"
        body = (
            f"Output{label}:\n```text\n{rendered_observation}\n```\n\n"
            f"State changes:\n```text\n{diff}\n```"
        )

        limit = self.max_environment_message_chars
        if limit <= 0:
            return body + "\n\n" + status

        separator = "\n\n"
        body_budget = max(0, limit - len(separator) - len(status))
        body = self._truncate(body, body_budget)
        rendered = body + separator + status
        # Defensive final bound (e.g. an unusually large status string).
        return self._truncate(rendered, limit)


    def _get_usage_info_string(self) -> str:
        fields = [f"Steps: {self._n_calls}/{self.step_limit}"] if self.step_limit else []
        if self._last_input_tokens is not None:
            if self.compaction_threshold_tokens > 0:
                threshold = self.compaction_threshold_tokens
                ratio = self._last_input_tokens / threshold
                hint = ""
                if ratio >= 1.0:
                    hint = "; compact(summary) recommended"
                elif ratio >= self.compaction_warn_fraction:
                    hint = "; approaching compaction threshold"
                fields.append(
                    f"Context: ~{self._last_input_tokens:,}/{threshold:,} input tokens{hint}"
                )
            else:
                fields.append(f"Context: ~{self._last_input_tokens:,} input tokens")
        try:
            fields.append(f"Total: {self.total_tokens_in + self.total_tokens_out} tokens")
        except (AttributeError, TypeError):
            pass
        try:
            fields.append(f"Budget: ${self.total_cost:.4f}/${self.cost_limit:.2f}")
        except (AttributeError, TypeError, ValueError):
            pass
        if self._start_time is not None:
            fields.append(f"Elapsed: {time.time() - self._start_time:.1f}s")
        return ", ".join(fields)
    
    # -- main loop --------------------------------------------------------

    def _reset_tracking(self) -> None:
        self._n_calls = 0
        self._start_time = time.time()
        self._start_cost += self.total_cost
        self._start_tokens_in += self.total_tokens_in
        self._start_tokens_out += self.total_tokens_out

    def __call__(self, prompt: str | list[dict[str, Any]], **state: Any) -> Any:
        namespace: dict[str, Any] = dict(state)
        finish_callable = namespace.get("finish", _finish)
        namespace["finish"] = finish_callable
        namespace.setdefault("__name__", "__codereact__")
        # ``compact`` controls this harness's transcript and is therefore owned
        # by the harness rather than by caller state.
        namespace["compact"] = _compact
        namespace.setdefault("__name__", "__codereact__")

        self._tracked_names = set(namespace.keys())
        self._history = []
        self._history_kinds = []
        for i, message in enumerate(self._init_messages(namespace, prompt)):
            self._add_message(message, kind="system" if i == 0 else "root")
        self._history_prefix_len = len(self._history)
        self._reset_tracking()
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

            self._fold_old_environment_outputs()
            try:
                tokens_before_call = getattr(self.lm, "total_tokens_in", None)
            except BaseException:
                tokens_before_call = None

            response_text = self.lm(self._history)

            try:
                tokens_after_call = getattr(self.lm, "total_tokens_in", None)
                if (
                    isinstance(tokens_before_call, (int, float))
                    and isinstance(tokens_after_call, (int, float))
                    and tokens_after_call >= tokens_before_call
                ):
                    self._last_input_tokens = int(tokens_after_call - tokens_before_call)
            except BaseException:
                pass

            if response_text is None:
                self.stop_reason = "lm_failure"
                break

            self._n_calls += 1
            self._add_message({"role": "assistant", "content": response_text}, kind = "assistant")

            code = self._extract_code(response_text)
            if code is None:
                self._add_message(
                    {
                        "role": "user",
                        "content": self._code_feedback(response_text),
                    },
                    kind=f"feedback:{self._n_calls}",
                )
                continue

            try:
                observation, finished, finish_value = self._execute(
                    namespace, code, finish_callable = finish_callable
                )
            except CompactRequested as exc:
                self._apply_compaction(exc.summary)
                continue

            self.state = self._public_state(namespace)
            if finished:
                self.stop_reason = "finished"
                self.result = finish_value
                return finish_value

            usage_info = self._get_usage_info_string()
            diff = self._diff_description(prev_snapshot, namespace)
            prev_snapshot = self._snapshot(namespace)
            self._add_message(
                {
                    "role": "user",
                    "content": self._render_environment_message(
                        observation,
                        diff,
                        usage_info
                    )
                },
                kind = f"observation:{self._n_calls}"
            )

        # A non-normal stop returns the recoverable REPL state. ``result``
        # remains None because finish(...) was never reached.
        return self._public_state(namespace)

    