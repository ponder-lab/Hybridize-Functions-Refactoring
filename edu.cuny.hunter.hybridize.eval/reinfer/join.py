"""Join one subject's strip records with the evaluator's output into outcome rows.

A stripped function's evaluator row is found by relative path and ``def`` line in the stripped tree
the evaluator analyzed, where the line is exact, and accepted only if the bare names agree (the
evaluator's qualified name can differ from the lexical one, #992). Within that function, the
evaluator's CSVs are then joined on their own key: relative path, function and definition ordinal.

Every function receives exactly one outcome:

- ``scored``: an inferred signature exists and is related to the removed one;
- ``not-reproduced:<reason>``: the function is in scope but the tool produced no spec. ``<reason>``
  is the evaluator's absence reason, or ``mapping`` when the removed spec holds a dict, a form the
  inference cannot express;
- ``no-call-site:in-tree`` / ``no-call-site:trimmed``: no evidence reached the function and no call
  to it was found, in the full checkout or in a trimmed one;
- ``evaluation-failed:<cause>``: the evaluator did not complete for the subject;
- ``relax-shapes``: a relaxation commit whose function the analysis reached, scored on its axes;
- ``not-reached:call-graph``: a relaxation commit whose function has in-tree callers the call graph did
  not connect, an in-scope miss (a relaxation commit with no caller at all scores ``no-call-site``);
- ``excluded:<reason>``: nothing can be scored (an unevaluable spec, a positional signature, a
  function the tool never considered).
"""

import ast
import csv
import os
import re

import spec

KEY = ("relative path", "function", "definition ordinal")

_IDENTITY = [
    "repo",
    "sha",
    "kind",
    "relative path",
    "function",
    "definition ordinal",
    "evaluator function",
    "evaluator ordinal",
    "library",
]
# The columns of each output, so that an output with no rows still has its header.
COLUMNS = {
    "functions": _IDENTITY
    + [
        "outcome",
        "also applies",
        "decorator",
        "other arguments",
        "relaxation",
        "removed spec",
        "removed source",
        "obtained by",
        "inferred spec",
        "absence reasons",
        "failed preconditions",
        "relation",
        "dtype relation",
        "shape relation",
        "arity removed",
        "arity inferred",
        "dimension rows",
        "resolved calls",
        "text callers",
        "trim",
    ],
    "parameters": _IDENTITY
    + [
        "outcome",
        "position",
        "param index",
        "param name",
        "removed spec",
        "removed leaves",
        "inferred spec",
        "relation",
        "dtype relation",
        "shape relation",
        "inferred ranks",
        "tensor types",
    ],
    "axes": _IDENTITY
    + [
        "position",
        "param index",
        "container position",
        "dim index",
        "extents",
        "classes",
        "verdict",
        "removed dim",
    ],
}

PRECONDITION_FAILURE_JAVA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "edu.cuny.hunter.hybridize.core",
    "src",
    "edu",
    "cuny",
    "hunter",
    "hybridize",
    "core",
    "analysis",
    "PreconditionFailure.java",
)


def precondition_names(path=PRECONDITION_FAILURE_JAVA):
    """Map each PreconditionFailure code to its constant name, read from the enum itself so the two
    cannot drift. ``NOT_ATTEMPTED`` says only that inference never ran; these say what stopped it.
    """
    with open(path) as f:
        return {
            int(code): name
            for name, code in re.findall(
                r"^\s+([A-Z_]+)\((\d+)\)", f.read(), re.MULTILINE
            )
        }


def read_rows(run_dir, name):
    path = os.path.join(run_dir, name)
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _key(row):
    return tuple(row[k] for k in KEY)


def _by_key(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(_key(row), []).append(row)
    return grouped


# --- The inferred side, from tensor_specs.csv ---------------------------------------------------


def _shape(text):
    """Parse the evaluator's shape rendering: ``(None, 128)``, ``(20,)``, ``()`` or ``None``."""
    value = ast.literal_eval(text)
    return None if value is None else list(value)


def inferred_tree(row):
    """Rebuild a parameter's inferred tree from its ``tensor_specs.csv`` row.

    A sequence renders its element shapes as ``[(2, 2), (3,)]`` and its dtypes as the DISTINCT element
    dtypes joined by ``|``. With one distinct dtype every element has it; with several, which element
    has which is not recoverable, and the elements' dtypes are left None with ``dtype_ambiguous`` set.
    """
    shape_text, dtype_text = row["shape"], row["dtype"]
    if shape_text.startswith("["):
        shapes = [None if s is None else list(s) for s in ast.literal_eval(shape_text)]
        dtypes = dtype_text.split("|")
        tree = spec.sequence(
            spec.leaf(dtypes[0] if len(dtypes) == 1 else None, s) for s in shapes
        )
        tree["dtype_ambiguous"] = len(dtypes) > 1
        return tree
    return spec.leaf(dtype_text, _shape(shape_text))


# --- Per-axis evidence, from parameter_dimensions.csv -------------------------------------------


def axis_evidence(dimension_rows):
    """Group a parameter's pre-consensus dimension rows by (container position, dim index).

    Each group carries the set of Constant extents and the set of raw dimension classes seen across
    the parameter's inferred types, plus the ranks those types have.
    """
    axes = {}
    ranks = set()
    for row in dimension_rows:
        if row.get("rank"):
            # A type of unknown rank reports its rank as TOP, which is evidence too, so it is kept.
            ranks.add(row["rank"])
        if row.get("dim index", "") == "":
            continue
        position = row.get("container position", "")
        dim_class = row["dim class"]
        name, _, value = dim_class.partition(",")
        axis = axes.setdefault(
            (position, int(row["dim index"])), {"extents": set(), "classes": set()}
        )
        axis["classes"].add(name)
        if name == "Constant":
            axis["extents"].add(int(value))
    return axes, ranks


def axis_verdict(axis):
    """IS's scoring rule. A wildcard is REQUIRED when the axis takes two or more distinct Constant
    extents or any member is Dynamic; UNDETERMINED when a member is Symbolic, Unresolved or TOP (or
    Ragged or Compound, which the rule does not yet name; Compound's components are not exported) and
    the Constants alone show no variation; otherwise UNNECESSARY, a single fixed extent. A scalar has no
    axes, so its rows carry no dim index and never reach here."""
    if len(axis["extents"]) > 1 or "Dynamic" in axis["classes"]:
        return "required"
    if axis["classes"] - {"Constant"}:
        return "undetermined"
    return "unnecessary"


# --- Call-site evidence -------------------------------------------------------------------------


def text_callers(checkout, qualname, def_file, def_line):
    """Count calls of the function's bare name across the checkout's ``.py`` files, excluding its own
    ``def`` line. A syntactic over-approximation, used only to tell "no caller exists" from "callers
    exist that the analysis did not connect"."""
    name = qualname.rsplit(".", 1)[-1]
    pattern = re.compile(r"(?<![\w])" + re.escape(name) + r"\s*\(")
    definition = re.compile(r"^\s*(async\s+)?def\s+" + re.escape(name) + r"\s*\(")
    count = 0
    for root, dirs, files in os.walk(checkout):
        dirs[:] = [d for d in dirs if d != ".git"]
        for file in files:
            if not file.endswith(".py"):
                continue
            path = os.path.join(root, file)
            with open(path, encoding="utf-8", errors="replace") as f:
                for number, line in enumerate(f, 1):
                    if definition.match(line) or line.lstrip().startswith("#"):
                        continue
                    if (
                        os.path.relpath(path, checkout) == def_file
                        and number == def_line
                    ):
                        continue
                    count += len(pattern.findall(line))
    return count


def resolved_calls(call_rows, qualname):
    """Count ``calls.csv`` rows whose ``callee`` (PyDev's fully qualified name of the call target) ends
    in this function's qualified name at a dot boundary. PyDev's module prefix is not compared, since
    it need not spell the module the way the evaluator's ``module`` column does."""
    return sum(
        1
        for row in call_rows
        if row["callee"] == qualname or row["callee"].endswith("." + qualname)
    )


def is_library(relative_path):
    parts = relative_path.replace(os.sep, "/").split("/")
    base = parts[-1]
    if base.startswith("test_") or base.endswith("_test.py") or base == "conftest.py":
        return False
    return not any(
        p
        in (
            "test",
            "tests",
            "examples",
            "example",
            "e2e",
            "integration_tests",
            "notebooks",
        )
        for p in parts[:-1]
    )


# --- The join -----------------------------------------------------------------------------------


def locate(record, by_location):
    """Find the evaluator's row for a stripped function by where it begins in the analyzed tree.

    The evaluator's name for a function can differ from the lexical qualified name (#992), so the row
    is found by relative path and beginning line (the evaluator's ``beginning line`` is the ``def`` line)
    in the very tree it analyzed, and accepted only if its bare name agrees.
    """
    row = by_location.get((record["file"], str(record["stripped_def_line"])))
    if row is None:
        return None
    if row["function"].rsplit(".", 1)[-1] != record["qualname"].rsplit(".", 1)[-1]:
        return None
    return row


UNDETERMINED = "UNDETERMINED"


def relate_entry(removed, inferred, axis):
    """``spec.relate_entry``, except where the inferred dtypes cannot be paired to positions.

    An inferred sequence whose elements have several dtypes carries ``dtype_ambiguous`` and no element
    dtypes; read as UNKNOWN, they would make every supplied dtype look TIGHTER. So its dtype relation is
    UNDETERMINED, and so is the combined one, unless the shapes alone are already INCOMPARABLE, which no
    dtype could change. The shape relation is unaffected.
    """
    if not inferred.get("dtype_ambiguous") or axis == "shape":
        return spec.relate_entry(removed, inferred, axis)
    if (
        axis == "both"
        and spec.relate_entry(removed, inferred, "shape") == spec.INCOMPARABLE
    ):
        return spec.INCOMPARABLE
    return UNDETERMINED


def relate_signature(removed, inferred, axis):
    """``spec.relate_signature`` over :func:`relate_entry`: INCOMPARABLE absorbs, then UNDETERMINED does."""
    if len(removed) != len(inferred):
        return spec.INCOMPARABLE
    relations = [relate_entry(r, i, axis) for r, i in zip(removed, inferred)]
    if any(r is None for r in relations):
        return None
    # The determined relations first: if they already combine to INCOMPARABLE, no value of an
    # undetermined one could change that.
    result = spec.AGREEMENT
    for r in relations:
        if r != UNDETERMINED:
            result = spec.combine(result, r)
    if result == spec.INCOMPARABLE:
        return spec.INCOMPARABLE
    return UNDETERMINED if UNDETERMINED in relations else result


def _relations(removed, inferred):
    return {
        axis: relate_entry(removed, inferred, axis)
        for axis in ("both", "dtype", "shape")
    }


def ground_truth_for(ground_truth, record, records):
    """The ground-truth entry for ``record``, and why there is none when one is keyed by its name.

    Entries are keyed by qualified name. An entry may also name its ``file`` and
    ``definition_ordinal``; each one it names must match. When what it names still leaves more than one
    stripped function by that name, the entry cannot say which one it describes, so none gets it.
    """
    entry = (ground_truth or {}).get(record["qualname"])
    if entry is None:
        return None, None
    selectors = [f for f in ("file", "definition_ordinal") if f in entry]
    if any(entry[f] != record[f] for f in selectors):
        return None, None
    candidates = sum(
        1
        for r in records
        if r["qualname"] == record["qualname"]
        and all(r[f] == entry[f] for f in selectors)
    )
    if candidates > 1:
        return None, (
            f"ground truth for {record['qualname']} does not say which of the {candidates} "
            "stripped functions that share that name it describes"
        )
    return entry, None


def removed_leaf(tree, container_position):
    """The leaf of the removed spec ``tree`` that an evaluator dimension row describes: the tree itself
    for a direct parameter (empty container position), or that element of a sequence."""
    if not tree:
        return None
    if container_position == "":
        return tree if tree["kind"] == "leaf" else None
    if tree["kind"] != "sequence":
        return None
    index = int(container_position)
    if index >= len(tree["elements"]):
        return None
    element = tree["elements"][index]
    return element if element["kind"] == "leaf" else None


def join_subject(
    subject, records, run_dir, checkout, ground_truth=None, trim=None, failure=None
):
    """Produce ``(function_rows, parameter_rows, axis_rows)`` for one subject."""
    functions = _by_key(read_rows(run_dir, "functions.csv"))
    specs = _by_key(
        r for r in read_rows(run_dir, "tensor_specs.csv") if r["source"] == "inferred"
    )
    absences = _by_key(
        r
        for r in read_rows(run_dir, "signature_absences.csv")
        if r["source"] == "absent"
    )
    parameters = _by_key(read_rows(run_dir, "parameters.csv"))
    dimensions = _by_key(read_rows(run_dir, "parameter_dimensions.csv"))
    calls = read_rows(run_dir, "calls.csv")
    failures = _by_key(read_rows(run_dir, "failed_preconditions.csv"))
    names = precondition_names()
    trim = trim or {}
    function_rows, parameter_rows, axis_rows = [], [], []

    by_location = {
        (rows[0]["relative path"], rows[0]["beginning line"]): rows[0]
        for rows in functions.values()
    }

    for record in records:
        evaluator_row = locate(record, by_location)
        key = _key(evaluator_row) if evaluator_row else None
        base = {
            "repo": subject["repo"],
            "sha": subject["sha"],
            "kind": subject["kind"],
            "relative path": record["file"],
            "function": record["qualname"],
            "definition ordinal": record["definition_ordinal"],
            "evaluator function": evaluator_row["function"] if evaluator_row else "",
            "evaluator ordinal": (
                evaluator_row["definition ordinal"] if evaluator_row else ""
            ),
            "library": is_library(record["file"]),
        }

        # The removed spec: a literal, an evaluated value, or nothing to score.
        removed, obtained_by, excluded = None, "", None
        if subject["kind"] == "relax_shapes":
            obtained_by = "none: a relaxation commit removes no spec"
        elif record["signature_form"] == "positional":
            excluded = "positional-signature"
        elif record["signature_form"] != "keyword":
            excluded = "no-signature-at-fix"
        else:
            resolved = record.get("resolved")
            try:
                if resolved:
                    removed = spec.parse_signature(resolved["source"])
                    obtained_by = f"literal, via the module-level name {resolved['name']} bound at line {resolved['line']}"
                else:
                    removed = spec.parse_signature(record["removed_source"])
                    obtained_by = "literal"
            except spec.NonLiteral as error:
                truth, ambiguity = ground_truth_for(ground_truth, record, records)
                if truth is None:
                    excluded = f"unevaluable: {ambiguity or error}"
                else:
                    removed = spec.signature_from_structure(truth["structure"])
                    obtained_by = "evaluated: " + ground_truth.get("obtained_by", "")

        present = key is not None
        non_self = sorted(
            (r for r in parameters.get(key, [])),
            key=lambda r: int(r["param index"]),
        )
        inferred_rows = sorted(
            # A row with no parameter index (an entry past the declared parameters) sorts last, so it
            # cannot shift the positional alignment of the indexed ones.
            specs.get(key, []),
            key=lambda r: (r["param index"] == "", int(r["param index"] or 0)),
        )
        inferred = [inferred_tree(r) for r in inferred_rows] if inferred_rows else None
        dimension_rows = dimensions.get(key, [])
        text = text_callers(
            checkout, record["qualname"], record["file"], record["def_line"]
        )
        resolved = resolved_calls(calls, record["qualname"])
        absence = sorted({r["absence reason"] for r in absences.get(key, [])})

        also = ""
        if excluded is None and not present and not failure:
            excluded = "not-considered-by-the-tool"
        if excluded is not None:
            outcome = f"excluded:{excluded}"
        elif failure:
            outcome = f"evaluation-failed:{failure}"
        elif subject["kind"] == "relax_shapes" and dimension_rows:
            outcome = "relax-shapes"
        elif subject["kind"] == "relax_shapes" and (resolved or text):
            # Callers exist in the tree, but no type evidence reached the function: an in-scope miss of
            # the call graph, distinct from a function the closed world puts out of reach (IS's ruling).
            outcome = "not-reached:call-graph"
        # PRECEDENCE: reachability before spec form. With no call site the tool has nothing to infer
        # from, whatever the spec's form, so a function the closed world puts out of reach is not
        # counted as a miss of the form; the form is kept in `also applies`.
        elif inferred is None and not dimension_rows and resolved == 0 and text == 0:
            outcome = "no-call-site:" + (
                "trimmed"
                if trim.get("sparse") and not trim.get("caller_scope_complete")
                else "in-tree"
            )
            if any(spec.contains_mapping(t) for t in removed or []):
                also = "not-reproduced:mapping"
        elif any(spec.contains_mapping(t) for t in removed or []):
            outcome = "not-reproduced:mapping"
        elif inferred is not None:
            outcome = "scored"
        else:
            outcome = "not-reproduced:" + ("|".join(absence) or "unknown")

        relation = {"both": None, "dtype": None, "shape": None}
        if outcome == "scored":
            relation = {
                axis: relate_signature(removed, inferred, axis) for axis in relation
            }

        function_rows.append(
            {
                **base,
                "outcome": outcome,
                "also applies": also,
                "decorator": record.get("decorator_callee") or "",
                "other arguments": "; ".join(record.get("other_arguments", [])),
                "relaxation": "; ".join(
                    f"{k}={v}" for k, v in record.get("relaxation", {}).items()
                ),
                "removed spec": (
                    ""
                    if removed is None
                    else " | ".join(spec.render(t) for t in removed)
                ),
                "removed source": record.get("removed_source") or "",
                "obtained by": obtained_by,
                "inferred spec": (
                    ""
                    if inferred is None
                    else " | ".join(spec.render(t) for t in inferred)
                ),
                "absence reasons": "|".join(absence),
                "failed preconditions": "|".join(
                    sorted(
                        {
                            names.get(int(r["code"]), r["code"])
                            for r in failures.get(key, [])
                            if r.get("code")
                        }
                    )
                ),
                "relation": relation["both"] or "",
                "dtype relation": relation["dtype"] or "",
                "shape relation": relation["shape"] or "",
                "arity removed": "" if removed is None else len(removed),
                "arity inferred": "" if inferred is None else len(inferred),
                "dimension rows": len(dimension_rows),
                "resolved calls": resolved,
                "text callers": text,
                "trim": trim.get("note", ""),
            }
        )

        # Per parameter, aligned positionally with the non-self parameters, as input_signature is.
        count = max(len(non_self), len(removed or []), len(inferred or []))
        for position in range(count):
            parameter = non_self[position] if position < len(non_self) else None
            index = parameter["param index"] if parameter else ""
            r_tree = removed[position] if removed and position < len(removed) else None
            i_tree = (
                inferred[position] if inferred and position < len(inferred) else None
            )
            rel = (
                _relations(r_tree, i_tree)
                if r_tree and i_tree
                else {"both": None, "dtype": None, "shape": None}
            )
            param_dims = (
                [d for d in dimension_rows if d["param index"] == index]
                if parameter
                else []
            )
            axes, ranks = axis_evidence(param_dims)
            parameter_rows.append(
                {
                    **base,
                    "outcome": outcome,
                    "position": position,
                    "param index": index,
                    "param name": parameter["param name"] if parameter else "",
                    "removed spec": spec.render(r_tree) if r_tree else "",
                    "removed leaves": (
                        "; ".join(
                            f"{p or '.'}={l['dtype'] or '?'} rank={'*' if l['shape'] is None else len(l['shape'])} axes={l['shape']}"
                            for p, l in spec.leaves(r_tree)
                        )
                        if r_tree
                        else ""
                    ),
                    "inferred spec": spec.render(i_tree) if i_tree else "",
                    "relation": rel["both"] or "",
                    "dtype relation": rel["dtype"] or "",
                    "shape relation": rel["shape"] or "",
                    "inferred ranks": "|".join(
                        sorted(
                            ranks,
                            key=lambda r: (
                                not r.isdigit(),
                                int(r) if r.isdigit() else 0,
                                r,
                            ),
                        )
                    ),
                    "tensor types": (
                        parameter.get("tensor types", "") if parameter else ""
                    ),
                }
            )
            for (container_position, dim), axis in sorted(axes.items()):
                removed_dim = ""
                r_leaf = removed_leaf(r_tree, container_position)
                if (
                    r_leaf
                    and r_leaf["shape"] is not None
                    and dim < len(r_leaf["shape"])
                ):
                    removed_dim = (
                        "None"
                        if r_leaf["shape"][dim] is None
                        else str(r_leaf["shape"][dim])
                    )
                axis_rows.append(
                    {
                        **base,
                        "position": position,
                        "param index": index,
                        "container position": container_position,
                        "dim index": dim,
                        "extents": "|".join(str(e) for e in sorted(axis["extents"])),
                        "classes": "|".join(sorted(axis["classes"])),
                        "verdict": axis_verdict(axis),
                        "removed dim": removed_dim,
                    }
                )
    for name, rows in zip(COLUMNS, (function_rows, parameter_rows, axis_rows)):
        for row in rows:
            if list(row) != COLUMNS[name]:
                raise AssertionError(
                    f"{name} row columns differ from COLUMNS: {list(row)}"
                )
    return function_rows, parameter_rows, axis_rows
