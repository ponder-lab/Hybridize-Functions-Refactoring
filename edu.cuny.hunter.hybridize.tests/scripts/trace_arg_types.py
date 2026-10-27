"""Trace concrete argument types for functions defined in a subject file.

Authoring-time companion to ponder-lab/Hybridize-Functions-Refactoring#810 and the
wala/ML#808 fixture idiom: execute a subject under python3.10 with no subject edits
and record, per function parameter, the set of concrete argument types observed at
runtime. Arguments exposing `shape` and `dtype` (TF tensors, variables, numpy arrays)
record (class, shape, dtype); everything else records the Python type name.

The report prints the raw per-parameter union plus, for all-tensor parameters of
consensus rank, the shape/dtype fold under the per-dimension singleton rule (a
disagreeing dimension generalizes to None). The fold is an authoring convenience;
the JUnit pin must still come from the tool's own emission.

Usage: python3.10 trace_arg_types.py <subject.py>
"""

import os
import runpy
import sys
from collections import defaultdict

# func name -> param name -> set of observation tuples.
OBSERVATIONS = defaultdict(lambda: defaultdict(set))


def describe(value):
    """A hashable description of one observed argument value."""
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None and dtype is not None:
        try:
            dims = tuple(shape.as_list()) if hasattr(shape, "as_list") else tuple(shape)
        except ValueError:  # unknown rank
            dims = None
        return (
            "tensor",
            type(value).__name__,
            dims,
            getattr(dtype, "name", str(dtype)),
        )
    return ("python", type(value).__name__)


def make_profiler(subject_path):
    def profiler(frame, event, arg):
        if event != "call":
            return
        code = frame.f_code
        if os.path.abspath(code.co_filename) != subject_path or code.co_name.startswith(
            "<"
        ):
            return
        names = code.co_varnames[: code.co_argcount + code.co_kwonlyargcount]
        for name in names:
            if name in frame.f_locals:
                OBSERVATIONS[code.co_name][name].add(describe(frame.f_locals[name]))

    return profiler


def fold(observed):
    """The per-dimension singleton fold of an all-tensor union, or None if inapplicable."""
    if not all(o[0] == "tensor" for o in observed):
        return None
    dtypes = {o[3] for o in observed}
    dtype = dtypes.pop() if len(dtypes) == 1 else None
    shapes = {o[2] for o in observed}
    if None in shapes or len({len(s) for s in shapes}) != 1:
        return (
            None,
            dtype,
        )  # unknown or disagreeing rank: shape degrades, dtype survives
    rank = len(next(iter(shapes)))
    dims = tuple(
        d.pop() if len(d := {s[j] for s in shapes}) == 1 else None for j in range(rank)
    )
    return (dims, dtype)


def main():
    subject_path = os.path.abspath(sys.argv[1])
    sys.setprofile(make_profiler(subject_path))
    try:
        runpy.run_path(subject_path, run_name="__main__")
    finally:
        sys.setprofile(None)

    for func in sorted(OBSERVATIONS):
        print(f"{func}:")
        for param, observed in OBSERVATIONS[func].items():
            print(f"  {param}: {sorted(observed)}")
            folded = fold(observed)
            if folded is not None:
                shape, dtype = folded
                print(f"    fold: shape={shape}, dtype={dtype}")


if __name__ == "__main__":
    main()
