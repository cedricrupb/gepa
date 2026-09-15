from __future__ import annotations

import io
import difflib
import copy
import contextlib
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Callable
from pathlib import Path

from collections import defaultdict

from gepa.core.state import GEPAState

from gepa.proposer.reflective_mutation.base import LanguageModel
from gepa.proposer.reflective_mutation.reflection_lm import ReflectionProposal

from gepa.agent import Agent, Finished


AGENTIC_TEMPLATE = """You're improving a component — a prompt, a piece of code, or another text artifact — of an assistant that performs a task for me.

Your goal this round is to improve the component's ability to solve the underlying task. Use observed behavior, diagnostic feedback, validation results, and evolutionary history as evidence what should change and why. 

Improve the underlying strategy or mechanism rather than memorizing individual examples or exploiting incidental properties of the evaluation environment.

**Parent:** `{{PARENT_ID}}` (Component: `{{PARENT_COMPONENT}}`)
**Parent validation score:** `{{PARENT_SCORE}}`
{{ALGORITHMIC_NOVELTY}}
## Recommended workflow

Work step-by-step so that observations from earlier steps can inform later decisions.

1. **Analyze the current state:**
    - `parent` - the parent component to evolve.
    - `dataset_with_feedback` - concrete examples of inputs, generated outputs, and feedback for the parent system.
    - `evolution_context` - the evolution history of what has been tried before.

    Understand the underlying task and inspect enough evidence to identify important successes, failures, and opportunities. 
    
    Produce a concise analysis report (idea introduced by the parent, observed impact, important failure modes or strengths, and takeaways) in a form you can refer back to during the rest of the run. 
    
    Then consider promising ways the system could work better, including approaches not directly suggested by the observed examples, and formulate a falsifiable hypothesis for the change you want to pursue.

2. **Prototype:** If `parent` exposes an `evaluate` function, use it when useful to test whether the central mechanism of your hypothesis behaves as intended before committing to the final edit.
    
    For each test:
      1. Use temporary variants of `parent` component when useful.
      2. Pull examples from `dataset_with_feedback` to test against.
      3. Compare alternatives and use the results to refine or reject your hypothesis.

3. **Implement the candidate:** Apply targeted edits to `parent` according to your hypothesis to create a new `component`.
    
    Then re-read the resulting component and check that:
        - the change actually implements the intended mechanism.
        - it addresses the underlying task rather than merely improving observed evaluation signals.
        - it is informed by the evidence and evolutionary history without being unnecessarily constrained by them.
        - it does not hardcode or memorize individual examples.
        - it remains valid and preserves useful existing behavior.

4. **Submit:** Submit your final candidate with `finish(component)`.

Call `finish(...)` alone in its own code block.
""".strip()

ALGORITHMIC_NOVELTY = """
### Algorithmic Novelty

Use `evolution_context` to understand what has already been explored and avoid needlessly repeating equivalent approaches.

When editing `parent`, prefer meaningful differences when they are supported by a promising hypothesis. Prioritize:

1. **Algorithmic differences**: Different approaches, logic, or strategies
2. **Structural changes**: Different data structures, control flow, or organization
3. **Functional improvements**: New features, optimizations, or capabilities
4. **Implementation variations**: Different ways of achieving the same goal that could lead to different performance characteristics
5. **Parameter and prompt changes**: Different parameters or prompt changes that could lead to different performance characteristics

Combining compatible ideas from previous candidates is valid, especially when they address different limitations.

Before submitting, briefly explain what mechanism changed relative to the most relevant previous approaches and why that difference is worth testing.

"""


DEFAULT_NOVELTY = """
### Novelty Criteria

Use `evolution_context` to understand what has already been explored and avoid needlessly repeating equivalent approaches. 

Prefer a meaningfully different hypothesis when there is evidence or a plausible reason to explore one. Combining compatible ideas from previous candidates is valid, especially when they address different limitations.

Before submitting, briefly explain what meaningfully changed relative to the most relevant previous approaches and why.
"""



def _finish(result: Any) -> None:
    assert isinstance(result, Component), f"Expected result to be a component, but got `{result}`"
    assert result.parent is not None, "Expected a changed component, but `result.parent` is None."
    raise Finished(result)


# Tools ---------------------------------------------------------------

class Component: 
 
    def __init__(self, current: str, parent: Component | None = None, doc: str | None = None):
        self._current = current
        self.parent = parent

        if doc is not None:
            self.__doc__ = doc

    @property
    def current(self):
        return self._current
 
    def view(self, start_line_idx: int = 0, end_line_idx: int = -1, raw: bool = False) -> str:
        """Return lines prefixed with line numbers N; line numbers are display only."""

        if raw:
            lines = self._current.splitlines()
            if end_line_idx == -1:
                end_line_idx = len(lines)
            rendered = lines[start_line_idx:end_line_idx]
        else:
            numbered_lines = list(enumerate(self._current.splitlines(), start=0))
    
            if end_line_idx == -1:
                end_line_idx = len(numbered_lines)
            selected = numbered_lines[start_line_idx:end_line_idx]
            rendered = [f"{line_no}\t{line}" for line_no, line in selected]
 
        return "\n".join(rendered)
 
    def str_replace(self, string_to_replace: str, replacement_str: str) -> Component:
        """Replace a unique substring and returns a new Component. Do not include line numbers."""
        occurrences = self._current.count(string_to_replace)
        if occurrences == 0:
            raise ValueError(
                f"string_to_replace not found in current component: {string_to_replace!r}"
            )
        if occurrences > 1:
            raise ValueError(
                f"string_to_replace is not unique ({occurrences} occurrences) -- "
                f"include more surrounding context to make it unique: {string_to_replace!r}"
            )
 
        return Component(
            self._current.replace(string_to_replace, replacement_str),
            parent=self,
            doc = self.__doc__,
        )
 
    def revert(self) -> Component:
        """Undo the last replace_str(), returning the previous Component -- or this one, unchanged, if there is no earlier version."""
        return self.parent or self

    def __len__(self):
        return len(self._current.splitlines())

    def __repr__(self) -> str:
        return f"Component(n_lines={len(self)})"
 
    def __str__(self):
        return self.view(raw = True)


class PythonHarness(Component):
     
    def str_replace(self, string_to_replace: str, replacement_str: str) -> PythonHarness:
        """Replace a unique substring and returns a new harness. Do not include line numbers."""
        component = super().str_replace(string_to_replace, replacement_str)
        return PythonHarness(
            component.current,
            parent=self,
            doc = self.__doc__,
        )

    def __repr__(self) -> str:
        return f"PythonHarness(n_lines={len(self)})"
    
    def evaluate(self, **namespace) -> str:
        """Run harness against the namespace and returns stdout; `namespace` is mutated in place by exec()."""
        stdout = io.StringIO()
   
        with contextlib.redirect_stdout(stdout):
            exec(self._current, namespace)

        output = stdout.getvalue()
        observation = output if output else "[no output]"
        return observation


def make_litellm_llm(model_name, **kwargs) -> LanguageModel:
    from gepa.lm import LM
    return LM(model_name, **kwargs)


class Instruction(Component):

    def __init__(self, current: str, lm: LanguageModel | str, parent: Component | None = None, **kwargs):
        super().__init__(current, parent)

        if isinstance(lm, str):
            lm = make_litellm_llm(lm, **kwargs)
        else:
            from gepa.lm import TrackingLM

            if not hasattr(lm, "total_cost"):
                lm = TrackingLM(lm)

        self._lm = lm

    def str_replace(self, string_to_replace: str, replacement_str: str) -> Instruction:
        """Replace a unique substring and returns a new harness. Do not include line numbers."""
        component = super().str_replace(string_to_replace, replacement_str)
        return Instruction(
            component._current,
            self._lm,
            parent = self,
            doc = self.__doc__,
        )

    def __repr__(self) -> str:
        return f"Instruction(n_lines={len(self)})"
        
    def evaluate(self, **kwargs) -> str:
        """Query LLM and return prompt result; `kwargs` is used to fill placeholders."""

        prompt = self._current
        for key, value in kwargs.items():
            prompt = prompt.replace("{%s}" % key, str(value))

        return self._lm(prompt)



class BashComponent(Component):
    """A file-backed Component.

    The component content is loaded from `path`. Read-only components may be
    inspected normally but cannot be edited. Call `copy(target_path)` to create
    a writeable working copy.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        writeable: bool = False,
        parent: BashComponent | None = None,
    ):
        self.path = Path(path).resolve()
        self.writeable = writeable

        if not self.path.is_file():
            raise ValueError(f"Expected a file, got: {self.path}")

        super().__init__(
            current=self.path.read_text(),
            parent=parent,
        )

    @property
    def current(self):
        return str(self.path)

    @classmethod
    def _from_current(
        cls,
        path: Path,
        current: str,
        *,
        writeable: bool,
        parent: BashComponent | None = None,
    ) -> BashComponent:
        """Construct without re-reading the file."""
        obj = cls.__new__(cls)
        obj.path = path
        obj.writeable = writeable
        Component.__init__(obj, current=current, parent=parent)
        return obj

    def _require_writeable(self) -> None:
        if not self.writeable:
            raise PermissionError(
                f"Component is read-only: {self.path}. "
                "Use copy(target_path) to obtain a writeable component."
            )

    def str_replace(
        self,
        string_to_replace: str,
        replacement_str: str,
    ) -> BashComponent:
        """Replace a unique substring and write the result to the backing file."""
        self._require_writeable()

        occurrences = self._current.count(string_to_replace)

        if occurrences == 0:
            raise ValueError(
                f"string_to_replace not found in current component: "
                f"{string_to_replace!r}"
            )

        if occurrences > 1:
            raise ValueError(
                f"string_to_replace is not unique ({occurrences} occurrences) -- "
                "include more surrounding context to make it unique: "
                f"{string_to_replace!r}"
            )

        current = self._current.replace(
            string_to_replace,
            replacement_str,
        )

        self.path.write_text(current)

        return self._from_current(
            self.path,
            current,
            writeable=True,
            parent=self,
        )

    def copy(self, target_path: str | Path) -> BashComponent:
        """Copy this component to `target_path` as a writeable component."""
        target_path = Path(target_path)

        if target_path.exists() and target_path.is_dir():
            raise ValueError(f"Target path is a directory: {target_path}")

        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(self._current)

        return BashComponent(
            target_path,
            writeable=True,
        )

    def __repr__(self) -> str:
        return (
            f"BashComponent("
            f"path={str(self.path)!r}, "
            f"writeable={self.writeable}, "
            f"lines={len(self._current.splitlines())}"
            f")"
        )
    

class Dataset:
    """A list of arbitrary-shaped reflection entries (agent notes, lessons
    learned, etc.) with three separate read paths for three audiences:
    `get` for code, `read` for one full entry shown to an LLM, `summary`
    for a token-efficient overview of everything at once.
    """
 
    PREVIEW_CHARS = 40     # per-entry preview length inside summary()'s table
 
    def __init__(self, entries: Sequence[Mapping[str, Any]], doc: str | None = None):
        self.entries = entries

        if doc:
            self.__doc__ = doc
 
    def get(self, idx: int) -> Mapping[str, Any]:
        """Return entry `idx` as the raw dict, use `read(idx)` to read the content of the dict."""
        return self.entries[idx]
 
    def read(self, idx: int, key: str) -> str:
        """Render `key` field of entry `idx` as one Markdown section."""
        value = self.entries[idx][key]
 
        def render_value(value: Any, level: int = 3) -> str:
            # level controls markdown header depth (###, ####, etc.)
            if isinstance(value, dict):
                s = ""
                for k, v in value.items():
                    s += f"{'#' * level} {k}\n"
                    s += render_value(v, min(level + 1, 6))
                if not value:
                    s += "\n"
                return s
            elif isinstance(value, list | tuple):
                s = ""
                for i, item in enumerate(value):
                    s += f"{'#' * level} Item {i + 1}\n"
                    s += render_value(item, min(level + 1, 6))
                if not value:
                    s += "\n"
                return s
            else:
                return f"{str(value).strip()}\n\n"

        return render_value(value, level = 3)
 
    def summary(self) -> str:
        """Render all entries as one Markdown table (one row per entry), with columns being the keys common to every entry -- call read(idx) for the full entry, or get(idx) for keys not shared by all entries."""
        if not self.entries:
            return "_(no entries)_"
 
        # Sorted, not a bare set: set iteration order isn't part of the
        # language guarantee, and the header row and every data row must
        # iterate in the SAME order for the table's columns to line up.
        keys = sorted(set.intersection(*(set(entry.keys()) for entry in self.entries)))
        if not keys:
            # No key is shared by every entry -- an intersection-based
            # table would have zero columns and silently show nothing.
            return "_(entries have no keys in common -- use get(idx)/read(idx) instead)_"
 
        rows = [f"| # | {' | '.join(keys)} |", f"|---|{'---|' * len(keys)}"]
        for idx, entry in enumerate(self.entries):
            column = []
            for key in keys:
                value = entry.get(key)
                if isinstance(value, str) and value.strip():
                    preview = value.strip()
                else:
                    preview = "[...]"

                preview = preview.replace("\n", " ").replace("\r", " ")
                preview = preview[:self.PREVIEW_CHARS] + "..." if len(preview) > self.PREVIEW_CHARS else preview
                column.append(preview)
            rows.append(f"| {idx} | {' | '.join(column)} |")
        return "\n".join(rows)

    def __repr__(self) -> str:
        keys = sorted(set.intersection(*(set(entry.keys()) for entry in self.entries)))
        return f"Dataset(n={len(self.entries)}, columns={keys})"
 
    def __str__(self):
        return self.summary()


class EvolutionContext:
    """Read-only evolution context providing information about the current candidate pool, the pareto frontier, trajectories,..."""

    def __init__(self, state: GEPAState[Any, Any], parent_idx: int):
        self._state = state
        self.parent_idx = parent_idx
        self._check_candidate_idx(parent_idx)

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._state.program_candidates)

    def __repr__(self) -> str:
        frontier_candidates = self._frontier_candidates()
        return (
            f"EvolutionContext(candidates={len(self)}, parent_idx={self.parent_idx}, "
            f"parent_score={self.score(self.parent_idx):.4f}, "
            f"pareto_candidates={len(frontier_candidates)}, "
            f"frontier_type={self._state.frontier_type!r})"
        )

    def fields(self) -> list[str]:
        """List readable public, non-callable attributes currently of the current evolution state"""
        result: list[str] = []
        for name in dir(self._state):
            if name.startswith("_"):
                continue
            try:
                value = getattr(self._state, name)
            except Exception:
                continue
            if not callable(value):
                result.append(name)
        return sorted(result)

    def state(self, name: str) -> Any:
        """Return a copy of one public field of the evolution state"""
        if not name or name.startswith("_"):
            raise ValueError("Only public fields can be read.")
        if not hasattr(self._state, name):
            raise KeyError(f"Evolution state has no field {name!r}")
        value = getattr(self._state, name)
        if callable(value):
            raise ValueError(f"{name!r} is callable, not a state field")
        try:
            return copy.deepcopy(value)
        except Exception as exc:
            raise TypeError(
                f"State field {name!r} could not be copied: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Candidate population

    def summary(
        self,
        *,
        limit: int | None = 30,
    ) -> str:
        """Compact overview of the accepted candidate population."""
        indices = list(range(len(self)))

        if limit is not None:
            indices = indices[: max(0, limit)]

        frontier = self._frontier_candidates()
        rows = [
            "| idx | score | n_val | pareto | parent | iteration |",
            "|---:|---:|---:|:---:|:---|:---|",
        ]
        for idx in indices:
            _, coverage = self._state.get_program_average_val_subset(idx)
            parents = [p for p in self._state.parent_program_for_candidate[idx] if p is not None]
            parent_text = ",".join(map(str, parents)) if parents else "-"
            marker = "*" if idx == self.parent_idx else ""
            pareto = "P" if idx in frontier else ""
            iteration_id = self._state.iteration_id_for_candidate_idx(idx) or "-"
            rows.append(
                f"| {idx}{marker} | {self.score(idx):.4f} | {coverage} | "
                f"{pareto} | {parent_text} | {iteration_id} |"
            )

        omitted = len(self) - len(indices)
        if omitted > 0:
            rows.append(f"\n_{omitted} additional accepted candidates omitted._")
        rows.append(
            f"\nActive parent: {self.parent_idx}. "
            f"Frontier type: {self._state.frontier_type}. "
            f"Accepted candidates: {len(self)}."
        )
        return "\n".join(rows)

    def candidate(self, candidate_idx: int, component: str | None = None) -> dict[str, str] | str:
        """Read an accepted candidate or one of its components."""
        self._check_candidate_idx(candidate_idx)
        candidate = self._state.program_candidates[candidate_idx]
        if component is None:
            return dict(candidate)
        if component not in candidate:
            raise KeyError(
                f"Candidate {candidate_idx} has no component {component!r}; "
                f"available: {sorted(candidate)}"
            )
        return candidate[component]

    def candidate_info(self, candidate_idx: int) -> dict[str, Any]:
        """Return compact metadata for one accepted candidate."""
        self._check_candidate_idx(candidate_idx)
        score, coverage = self._state.get_program_average_val_subset(candidate_idx)
        frontier_keys = self._frontier_keys_for(candidate_idx)
        parents = [
            p
            for p in self._state.parent_program_for_candidate[candidate_idx]
            if p is not None
        ]
        metric_calls = None
        if candidate_idx < len(self._state.num_metric_calls_by_discovery):
            metric_calls = self._state.num_metric_calls_by_discovery[candidate_idx]
        return {
            "candidate_idx": candidate_idx,
            "is_parent": candidate_idx == self.parent_idx,
            "score": score,
            "validation_coverage": coverage,
            "objective_scores": dict(self._state.prog_candidate_objective_scores[candidate_idx]),
            "parents": parents,
            "children": self.children(candidate_idx),
            "iteration_id": self._state.iteration_id_for_candidate_idx(candidate_idx),
            "pareto": bool(frontier_keys),
            "pareto_keys": frontier_keys,
            "metric_calls_at_discovery": metric_calls,
            "components": sorted(self._state.program_candidates[candidate_idx]),
        }

    def score(self, candidate_idx: int) -> float:
        """Return aggregate validation score for a candidate."""
        self._check_candidate_idx(candidate_idx)
        return self._state.program_full_scores_val_set[candidate_idx]

    def validation_scores(
        self,
        candidate_idx: int,
        *,
        descending: bool = False,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return the candidate's per-validation-example scores"""
        self._check_candidate_idx(candidate_idx)
        items = list(self._state.prog_candidate_val_subscores[candidate_idx].items())
        items.sort(key=lambda item: repr(item[0]), reverse=descending)
        if limit is not None:
            items = items[: max(0, limit)]
        return [{"val_id": val_id, "score": score} for val_id, score in items]

    def objective_scores(self, candidate_idx: int) -> dict[str, float]:
        """Return aggregate objective scores for one accepted candidate."""
        self._check_candidate_idx(candidate_idx)
        return dict(self._state.prog_candidate_objective_scores[candidate_idx])

    # ------------------------------------------------------------------
    # Comparisons

    def compare(self, a: int, b: int, *, limit: int = 10) -> dict[str, Any]:
        """Compare two candidates using their validation evidence; returns aggregate score delta, overlap coverage, counts of improvements
        and regressions, and the largest per-example changes. Positive deltas
        mean candidate ``b`` performed better than candidate ``a``.
        """
        self._check_candidate_idx(a)
        self._check_candidate_idx(b)
        sa = self._state.prog_candidate_val_subscores[a]
        sb = self._state.prog_candidate_val_subscores[b]
        shared = set(sa) & set(sb)

        deltas = [
            {
                "val_id": val_id,
                "a": sa[val_id],
                "b": sb[val_id],
                "delta": sb[val_id] - sa[val_id],
            }
            for val_id in shared
        ]
        improved = sorted(deltas, key=lambda x: x["delta"], reverse=True)
        regressed = sorted(deltas, key=lambda x: x["delta"])

        return {
            "a": a,
            "b": b,
            "score_a": self.score(a),
            "score_b": self.score(b),
            "score_delta": self.score(b) - self.score(a),
            "shared_validation_examples": len(shared),
            "only_a": sorted(set(sa) - set(sb), key=repr),
            "only_b": sorted(set(sb) - set(sa), key=repr),
            "num_improved": sum(d["delta"] > 0 for d in deltas),
            "num_regressed": sum(d["delta"] < 0 for d in deltas),
            "num_unchanged": sum(d["delta"] == 0 for d in deltas),
            "largest_improvements": [d for d in improved if d["delta"] > 0][:limit],
            "largest_regressions": [d for d in regressed if d["delta"] < 0][:limit],
            "changed_components": self.changed_components(a, b),
        }

    def changed_components(self, a: int, b: int) -> list[str]:
        """List component names whose text differs between two candidates."""
        self._check_candidate_idx(a)
        self._check_candidate_idx(b)
        ca = self._state.program_candidates[a]
        cb = self._state.program_candidates[b]
        names = set(ca) | set(cb)
        return sorted(name for name in names if ca.get(name) != cb.get(name))

    def diff(
        self,
        a: int,
        b: int,
        *,
        component: str | None = None,
        context_lines: int = 3,
        max_chars: int = 20_000,
    ) -> str:
        """Return a unified textual diff between two accepted candidates."""
        self._check_candidate_idx(a)
        self._check_candidate_idx(b)
        ca = self._state.program_candidates[a]
        cb = self._state.program_candidates[b]

        if component is not None:
            names = [component]
        else:
            names = self.changed_components(a, b)

        chunks: list[str] = []
        for name in names:
            if name not in ca and name not in cb:
                raise KeyError(f"Neither candidate has component {name!r}")
            before = ca.get(name, "").splitlines(keepends=True)
            after = cb.get(name, "").splitlines(keepends=True)
            diff = "".join(
                difflib.unified_diff(
                    before,
                    after,
                    fromfile=f"candidate_{a}/{name}",
                    tofile=f"candidate_{b}/{name}",
                    n=context_lines,
                )
            )
            if diff:
                chunks.append(diff)

        text = "\n".join(chunks) if chunks else "(no textual differences)"
        return self._middle_truncate(text, max_chars)

    # ------------------------------------------------------------------
    # Graph / Pareto structure

    def parents(self, candidate_idx: int) -> list[int]:
        """Return accepted parent candidate indices, excluding the seed's None."""
        self._check_candidate_idx(candidate_idx)
        return [
            p
            for p in self._state.parent_program_for_candidate[candidate_idx]
            if p is not None
        ]

    def children(self, candidate_idx: int) -> list[int]:
        """Return accepted candidates that directly descend from candidate_idx."""
        self._check_candidate_idx(candidate_idx)
        return [
            child_idx
            for child_idx, parents in enumerate(self._state.parent_program_for_candidate)
            if candidate_idx in parents
        ]


    def pareto(self, candidate_idx: int | None = None) -> dict[str, Any]:
        """Inspect the current Pareto frontier."""
        mapping = self._state.get_pareto_front_mapping()

        if candidate_idx is not None:
            self._check_candidate_idx(candidate_idx)
            keys = [key for key, programs in mapping.items() if candidate_idx in programs]
            return {
                "frontier_type": self._state.frontier_type,
                "candidate_idx": candidate_idx,
                "on_frontier": bool(keys),
                "keys": copy.deepcopy(keys),
                "num_keys": len(keys),
            }

        contributions: dict[int, int] = defaultdict(int)
        for programs in mapping.values():
            for idx in programs:
                contributions[idx] += 1
        return {
            "frontier_type": self._state.frontier_type,
            "mapping": copy.deepcopy(mapping),
            "candidate_contributions": dict(sorted(contributions.items())),
            "candidate_indices": sorted(contributions),
        }

    # ------------------------------------------------------------------
    # Historical attempts / trace

    def attempts(
        self,
        *,
        parent_idx: int | None = None,
        accepted: bool | None = None,
        limit: int | None = 50,
        newest_first: bool = True,
    ) -> list[dict[str, Any]]:
        """Summarize proposal attempts."""
        rows: list[dict[str, Any]] = []
        for trace_idx, entry in enumerate(self._state.full_program_trace):
            selected_parent = entry.get("selected_program_candidate")
            was_accepted = entry.get("proposal_accepted")
            if parent_idx is not None and selected_parent != parent_idx:
                continue
            if accepted is not None and bool(was_accepted) != accepted:
                continue

            before = entry.get("subsample_scores")
            after = entry.get("new_subsample_scores")
            rows.append(
                {
                    "trace_idx": trace_idx,
                    "iteration_id": entry.get("iteration_id"),
                    "parent_idx": selected_parent,
                    "accepted": was_accepted,
                    "new_candidate_idx": entry.get("new_program_idx"),
                    "new_candidate_indices": entry.get("new_program_indices"),
                    "subsample_ids": copy.deepcopy(entry.get("subsample_ids")),
                    "subsample_score_before": self._mean_or_none(before),
                    "subsample_score_after": self._mean_or_none(after),
                    "reject_reason": entry.get("reject_reason"),
                    "has_proposed_candidate": isinstance(entry.get("proposed_candidate"), dict),
                }
            )

        if newest_first:
            rows.reverse()
        if limit is not None:
            rows = rows[: max(0, limit)]
        return rows

    def trace(self, trace_idx: int) -> dict[str, Any]:
        """Return a raw execution trace."""
        if trace_idx < 0:
            trace_idx += len(self._state.full_program_trace)
        if not (0 <= trace_idx < len(self._state.full_program_trace)):
            raise IndexError(f"Trace index out of range: {trace_idx}")
        try:
            return copy.deepcopy(self._state.full_program_trace[trace_idx])
        except Exception as exc:
            raise TypeError(f"Trace entry {trace_idx} could not be copied: {exc}") from exc

    def best_outputs(self, val_id: Any | None = None) -> Any:
        """Read best validation outputs, if tracking is enabled."""
        outputs = self._state.best_outputs_valset
        if outputs is None:
            return None
        if val_id is None:
            return copy.deepcopy(outputs)
        if val_id not in outputs:
            raise KeyError(f"No tracked best output for validation id {val_id!r}")
        return copy.deepcopy(outputs[val_id])

    def run_info(self) -> dict[str, Any]:
        """Return high-level optimization-run counters and configuration."""
        return {
            "iteration": self._state.i,
            "total_num_evals": self._state.total_num_evals,
            "num_full_ds_evals": self._state.num_full_ds_evals,
            "frontier_type": self._state.frontier_type,
            "components": list(self._state.list_of_named_predictors),
            "num_candidates": len(self),
            "num_trace_entries": len(self._state.full_program_trace),
            "parent_idx": self.parent_idx,
            "parent_score": self.score(self.parent_idx),
        }

    # ------------------------------------------------------------------
    # Helpers

    def _check_candidate_idx(self, candidate_idx: int) -> None:
        if not isinstance(candidate_idx, int):
            raise TypeError("candidate_idx must be an int")
        if candidate_idx < 0 or candidate_idx >= len(self._state.program_candidates):
            raise IndexError(
                f"Candidate index {candidate_idx} out of range "
                f"[0, {len(self._state.program_candidates) - 1}]"
            )

    def _frontier_candidates(self) -> set[int]:
        result: set[int] = set()
        for programs in self._state.get_pareto_front_mapping().values():
            result.update(programs)
        return result

    def _frontier_keys_for(self, candidate_idx: int) -> list[Any]:
        return [
            copy.deepcopy(key)
            for key, programs in self._state.get_pareto_front_mapping().items()
            if candidate_idx in programs
        ]

    @staticmethod
    def _mean_or_none(values: Any) -> float | None:
        if isinstance(values, dict):
            values = list(values.values())
        if not isinstance(values, (list, tuple)) or not values:
            return None
        numeric = [float(x) for x in values if isinstance(x, (int, float))]
        return sum(numeric) / len(numeric) if numeric else None

    @staticmethod
    def _middle_truncate(text: str, limit: int) -> str:
        if limit <= 0 or len(text) <= limit:
            return text
        marker = "\n... <diff truncated> ...\n"
        available = max(2, limit - len(marker))
        head = available // 2
        tail = available - head
        return text[:head] + marker + text[-tail:]


class BashTool:
    """Bash tool to interact with the computer"""
    def __init__(self, target_dir: str | Path, doc : str | None = None):
        self.root = Path(target_dir).resolve()
        if not self.root.is_dir():
            raise ValueError(f"Not a directory: {self.root}")

        if doc:
            self.__doc__ = doc

    def _path(self, path: str | Path) -> Path:
        path = (self.root / path).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"Path escapes target directory: {path}")
        return path

    def read(self, path: str | Path) -> str:
        """Read a text file relative to the target directory."""
        return self._path(path).read_text()

    def write(self, path: str | Path, content: str) -> None:
        """Write a text file relative to the target directory."""
        path = self._path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def bash(self, command: str) -> str:
        """Run a bash command with the target directory as cwd."""
        result = subprocess.run(
            command,
            shell=True,
            executable="/bin/bash",
            cwd=self.root,
            text=True,
            capture_output=True,
        )

        output = result.stdout
        if result.stderr:
            output += ("\n" if output else "") + result.stderr

        if result.returncode != 0:
            raise RuntimeError(
                f"Command failed with exit code {result.returncode}:\n{output}"
            )

        return output

    def __repr__(self) -> str:
        return f"BashTool(root={str(self.root)!r})"


class Memory:
    """Persistent key-value memory for an agent.

    Use `set` to store information that should survive beyond the current
    REPL episode, `get` to retrieve the raw Python value, `read` to render
    a value for inspection, and `summary` for a compact overview.
    """

    PREVIEW_CHARS = 80

    def __init__(self, initial: Mapping[str, Any] | None = None, doc: str | None = None):
        self._data: dict[str, Any] = dict(initial or {})
        if doc:
            self.__doc__ = doc

    def get(self, key: str, default: Any = None) -> Any:
        """Return the raw value stored under `key`, or `default` if missing."""
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Store `value` under `key`, replacing any previous value."""
        self._data[key] = value

    def delete(self, key: str) -> bool:
        """Delete `key`. Return True if it existed, otherwise False."""
        if key not in self._data:
            return False
        del self._data[key]
        return True

    def contains(self, key: str) -> bool:
        """Return whether `key` exists in memory."""
        return key in self._data

    def keys(self) -> list[str]:
        """Return stored keys in sorted order."""
        return sorted(self._data)

    def items(self) -> list[tuple[str, Any]]:
        """Return stored key-value pairs sorted by key."""
        return [(key, self._data[key]) for key in sorted(self._data)]

    def clear(self) -> None:
        """Remove all stored entries."""
        self._data.clear()

    def read(self, key: str) -> str:
        """Render the complete value stored under `key` for inspection."""
        if key not in self._data:
            raise KeyError(key)
        return self._render_value(self._data[key])

    def summary(self) -> str:
        """Render a compact table of keys, value types, and previews."""
        if not self._data:
            return "_(memory is empty)_"

        rows = [
            "| key | type | preview |",
            "|---|---|---|",
        ]

        for key in sorted(self._data):
            value = self._data[key]

            try:
                preview = str(value).strip()
            except Exception:
                preview = "<unprintable>"

            preview = preview.replace("\n", " ").replace("\r", " ")
            if len(preview) > self.PREVIEW_CHARS:
                preview = preview[: self.PREVIEW_CHARS] + "..."

            # Keep Markdown tables intact.
            key_text = str(key).replace("|", "\\|")
            preview = preview.replace("|", "\\|")

            rows.append(
                f"| {key_text} | {type(value).__name__} | {preview} |"
            )

        return "\n".join(rows)

    @classmethod
    def _render_value(cls, value: Any, level: int = 3) -> str:
        """Render nested Python values as lightweight Markdown."""
        if isinstance(value, Mapping):
            if not value:
                return "_(empty mapping)_\n"

            parts: list[str] = []
            for key, item in value.items():
                parts.append(f"{'#' * level} {key}\n")
                parts.append(
                    cls._render_value(item, min(level + 1, 6))
                )
            return "".join(parts)

        if isinstance(value, (list, tuple)):
            if not value:
                return "_(empty sequence)_\n"

            parts = []
            for i, item in enumerate(value):
                parts.append(f"{'#' * level} Item {i + 1}\n")
                parts.append(
                    cls._render_value(item, min(level + 1, 6))
                )
            return "".join(parts)

        return f"{str(value).strip()}\n"

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __repr__(self) -> str:
        keys = sorted(self._data)
        return f"Memory(n={len(self._data)}, keys={keys!r})"

    def __str__(self) -> str:
        return self.summary()
    

# -------------------------------------------------------------------

def is_python_code(text: str) -> bool:
    import ast
    try:
        ast.parse(text, mode="exec")
        return True
    except SyntaxError:
        return False


class AgenticReflectionLM:

    def __init__(self, 
                lm: LanguageModel | str,
                logger: Any | None = None,
                adapter: Any | None = None,
                task_lm: LanguageModel | str | None = None,
                **agent_kwargs: Any,):
        self.agent = Agent(
            lm, **(agent_kwargs or {})
        )

        self.logger = logger
        self.adapter = adapter
        self.task_lm = task_lm

    def bind_logger(self, logger: Any) -> None:
        if self.logger is None:
            self.logger = logger

    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger.log(message)

    def bind_adapter(self, adapter: Any) -> None:
        if self.adapter is None:
            self.adapter = adapter

    def reflect(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> tuple[ReflectionProposal, AgenticReflectionLM]:
        context = {
            "parent_idx": None,
            "parent_validation_score": None,
            "population": tuple()
        }
        
        return self.reflect_with_context(candidate, reflective_dataset, components_to_update, context)


    def _component(self, candidate_component: str) -> Component:
        try:
            path = Path(candidate_component)
            if path.exists():
                return BashComponent(path)
        except Exception:
            pass

        if is_python_code(candidate_component):
            return PythonHarness(candidate_component)
        if self.task_lm:
            return Instruction(candidate_component, self.task_lm)

        return Component(candidate_component)


    def reflect_with_context(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
        context: Mapping[str, Any],
    ) -> tuple[ReflectionProposal, AgenticReflectionLM]:
            
        parent_idx    = context["parent_idx"]
        current_state = context["gepa_state"]

        evolution_context = EvolutionContext(current_state, parent_idx)
        parent_validation_score = evolution_context.score(parent_idx)

        proposal = ReflectionProposal(new_texts={}, prompts={}, raw_lm_outputs={}, metadata={})

        for comp in components_to_update:
            if comp not in reflective_dataset or not reflective_dataset.get(comp):
                self._log(f"Component '{comp}' is not in reflective dataset. Skipping.")
                continue
            if comp not in candidate:
                self._log(f"Component '{comp}' is missing from candidate. Skipping.")
                continue

            parent    = self._component(candidate[comp])
            dataset   = Dataset(reflective_dataset[comp])

            prompt = (
                AGENTIC_TEMPLATE
                    .replace("{{PARENT_ID}}", str(parent_idx))
                    .replace("{{PARENT_COMPONENT}}", str(comp))
                    .replace("{{PARENT_SCORE}}", f"{parent_validation_score:.4f}")
                    .replace("{{ALGORITHMIC_NOVELTY}}", ALGORITHMIC_NOVELTY if isinstance(parent, (PythonHarness, BashComponent)) else DEFAULT_NOVELTY)
            )

            result = self.agent(
                prompt = prompt,
                parent = parent,
                dataset_with_feedback = dataset,
                evolution_context = evolution_context,
                finish = _finish,
            )

            proposal.new_texts[comp] = result.current
            proposal.prompts[comp] = prompt
            proposal.raw_lm_outputs[comp] = result.current

        return proposal, self
            

    