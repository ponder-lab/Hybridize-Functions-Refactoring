"""Input-signature trees and the supplied-versus-inferred order that relates them.

A spec tree is one of three JSON-shaped dicts:

- ``{"kind": "leaf", "spec_type": "TensorSpec", "dtype": "float32" | None, "shape": [int | None, ...] | None}``
  where a ``None`` dtype is UNKNOWN and a ``None`` shape is unknown rank;
- ``{"kind": "sequence", "elements": [tree, ...]}`` for a list or tuple;
- ``{"kind": "mapping", "items": {key: tree, ...}}`` for a dict.

The order mirrors ``InputSignature.relate`` in edu.cuny.hunter.hybridize.core, clause for clause. It is
restated here rather than called because the harness runs outside the OSGi runtime; the two are held
together by ``relation_cases.txt``, which both this module's tests and the Java ``InputSignatureTest``
read, so a change to either side that the other does not share fails a build.
"""

import ast

AGREEMENT = "AGREEMENT"
SUPPLIED_TIGHTER = "SUPPLIED_TIGHTER"
SUPPLIED_BROADER = "SUPPLIED_BROADER"
INCOMPARABLE = "INCOMPARABLE"

SPEC_TYPES = ("TensorSpec", "RaggedTensorSpec", "SparseTensorSpec")


class NonLiteral(ValueError):
    """The removed expression is not a literal spec, so it has to be evaluated to be known."""


def leaf(dtype, shape, spec_type="TensorSpec"):
    return {
        "kind": "leaf",
        "spec_type": spec_type,
        "dtype": dtype,
        "shape": None if shape is None else list(shape),
    }


def sequence(elements):
    return {"kind": "sequence", "elements": list(elements)}


def mapping(items):
    return {"kind": "mapping", "items": dict(items)}


# --- The order, mirroring InputSignature --------------------------------------------------------


def combine(a, b):
    """AGREEMENT is the identity, INCOMPARABLE absorbs, and TIGHTER with BROADER is INCOMPARABLE."""
    if a == b:
        return a
    if a == AGREEMENT:
        return b
    if b == AGREEMENT:
        return a
    return INCOMPARABLE


def relate_dtype(supplied, inferred):
    if supplied == inferred:
        return AGREEMENT
    if supplied is None:
        return SUPPLIED_BROADER
    if inferred is None:
        return SUPPLIED_TIGHTER
    return INCOMPARABLE


def relate_dim(supplied, inferred):
    """A non-integer dimension (None, symbolic, ragged) is a wildcard; all wildcards are equal."""
    supplied_concrete = isinstance(supplied, int)
    inferred_concrete = isinstance(inferred, int)
    if not supplied_concrete and not inferred_concrete:
        return AGREEMENT
    if supplied_concrete and not inferred_concrete:
        return SUPPLIED_TIGHTER
    if not supplied_concrete and inferred_concrete:
        return SUPPLIED_BROADER
    return AGREEMENT if supplied == inferred else INCOMPARABLE


def relate_shape(supplied, inferred):
    if supplied is None and inferred is None:
        return AGREEMENT
    if supplied is None:
        return SUPPLIED_BROADER
    if inferred is None:
        return SUPPLIED_TIGHTER
    if len(supplied) != len(inferred):
        return INCOMPARABLE
    result = AGREEMENT
    for s, i in zip(supplied, inferred):
        result = combine(result, relate_dim(s, i))
    return result


def _relate_leaves(supplied, inferred, axis):
    if axis == "dtype":
        return relate_dtype(supplied["dtype"], inferred["dtype"])
    if axis == "shape":
        return relate_shape(supplied["shape"], inferred["shape"])
    return combine(
        relate_dtype(supplied["dtype"], inferred["dtype"]),
        relate_shape(supplied["shape"], inferred["shape"]),
    )


def relate_entry(supplied, inferred, axis="both"):
    """Relate one parameter's supplied tree to its inferred one.

    ``axis`` is ``"both"`` (InputSignature.relate's own relation), ``"dtype"`` or ``"shape"``: the
    same lattice restricted to one axis, so a dtype disagreement is kept apart from a shape one. A
    structural mismatch (a tensor against a sequence, sequences of different arity) is INCOMPARABLE on
    every axis, as it is in the Java order. A mapping on either side has no counterpart in the order and
    returns None; the caller records that outcome instead of a relation.
    """
    if supplied["kind"] == "mapping" or inferred["kind"] == "mapping":
        return None
    if supplied["kind"] == "leaf" and inferred["kind"] == "leaf":
        return _relate_leaves(supplied, inferred, axis)
    if supplied["kind"] == "sequence" and inferred["kind"] == "sequence":
        if len(supplied["elements"]) != len(inferred["elements"]):
            return INCOMPARABLE
        result = AGREEMENT
        for s, i in zip(supplied["elements"], inferred["elements"]):
            if s["kind"] != "leaf" or i["kind"] != "leaf":
                # InputSignature.Sequence holds tensor types only; a deeper nesting has no counterpart.
                return None
            result = combine(result, _relate_leaves(s, i, axis))
        return result
    return INCOMPARABLE


def relate_signature(supplied, inferred, axis="both"):
    """Relate whole signatures (lists of per-parameter trees), as InputSignature.relate does."""
    if len(supplied) != len(inferred):
        return INCOMPARABLE
    result = AGREEMENT
    for s, i in zip(supplied, inferred):
        r = relate_entry(s, i, axis)
        if r is None:
            return None
        result = combine(result, r)
    return result


def contains_mapping(tree):
    if tree["kind"] == "mapping":
        return True
    if tree["kind"] == "sequence":
        return any(contains_mapping(e) for e in tree["elements"])
    return False


def leaves(tree, path=""):
    """Yield ``(path, leaf)`` pairs, the path spelling the position: ``[0]``, ``.ids``, ``[1].length``."""
    if tree["kind"] == "leaf":
        yield path, tree
    elif tree["kind"] == "sequence":
        for index, element in enumerate(tree["elements"]):
            yield from leaves(element, f"{path}[{index}]")
    else:
        for key in sorted(tree["items"]):
            yield from leaves(tree["items"][key], f"{path}.{key}")


def render(tree):
    """A compact, stable spelling of a tree, e.g. ``float32[None,32]`` or ``(int32[], {ids: int64[None,None]})``."""
    if tree["kind"] == "leaf":
        dtype = tree["dtype"] or "?"
        shape = (
            "*"
            if tree["shape"] is None
            else ",".join("None" if d is None else str(d) for d in tree["shape"])
        )
        prefix = "" if tree["spec_type"] == "TensorSpec" else tree["spec_type"] + ":"
        return f"{prefix}{dtype}[{shape}]"
    if tree["kind"] == "sequence":
        return "(" + ", ".join(render(e) for e in tree["elements"]) + ")"
    return (
        "{"
        + ", ".join(f"{k}: {render(v)}" for k, v in sorted(tree["items"].items()))
        + "}"
    )


def parse_rendered(text):
    """Parse ``render``'s spelling back into a tree: ``float32[None,32]``, ``?[*]``, ``(int32[2], bool[])``.

    The inverse of ``render`` for leaves and sequences, which is all ``relation_cases.txt`` uses.
    """
    text = text.strip()
    if text.startswith("("):
        if not text.endswith(")"):
            raise ValueError(f"unbalanced sequence: {text}")
        inner, elements, depth, start = text[1:-1], [], 0, 0
        for index, char in enumerate(inner):
            if char in "([":
                depth += 1
            elif char in ")]":
                depth -= 1
            elif char == "," and depth == 0:
                elements.append(inner[start:index])
                start = index + 1
        if inner.strip():
            elements.append(inner[start:])
        return sequence(parse_rendered(e) for e in elements)
    dtype, _, rest = text.partition("[")
    if not rest.endswith("]"):
        raise ValueError(f"not a rendered leaf: {text}")
    body = rest[:-1]
    if body == "*":
        shape = None
    elif body == "":
        shape = []
    else:
        shape = [None if d.strip() == "None" else int(d) for d in body.split(",")]
    return leaf(None if dtype == "?" else dtype, shape)


def parse_rendered_signature(text):
    """A signature is its entries' renderings joined by `` | ``."""
    return [parse_rendered(e) for e in text.split(" | ")]


# --- Parsing a removed literal ------------------------------------------------------------------


def _name_tail(node):
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


def _dtype(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Constant) and node.value is None:
        return None
    tail = _name_tail(node)
    if tail is None:
        raise NonLiteral(f"dtype is not a literal: {ast.unparse(node)}")
    return tail


def _shape(node):
    if isinstance(node, ast.Constant) and node.value is None:
        return None
    if isinstance(node, (ast.List, ast.Tuple)):
        dims = []
        for element in node.elts:
            if isinstance(element, ast.Constant) and (
                element.value is None or type(element.value) is int
            ):
                dims.append(element.value)
            else:
                raise NonLiteral(
                    f"shape dimension is not a literal: {ast.unparse(element)}"
                )
        return dims
    if (
        isinstance(node, ast.Call)
        and _name_tail(node.func) == "TensorShape"
        and len(node.args) == 1
    ):
        return _shape(node.args[0])
    raise NonLiteral(f"shape is not a literal: {ast.unparse(node)}")


def _spec_call(node):
    spec_type = _name_tail(node.func)
    arguments = {k.arg: k.value for k in node.keywords}
    positional = ("shape", "dtype", "name")
    for index, value in enumerate(node.args):
        if index >= len(positional):
            raise NonLiteral(
                f"unexpected positional argument to {spec_type}: {ast.unparse(value)}"
            )
        arguments.setdefault(positional[index], value)
    shape = _shape(arguments["shape"]) if "shape" in arguments else None
    dtype = (
        _dtype(arguments["dtype"]) if "dtype" in arguments else "float32"
    )  # TensorSpec's own default.
    return leaf(dtype, shape, spec_type)


def parse_node(node):
    if isinstance(node, (ast.List, ast.Tuple)):
        return sequence(parse_node(e) for e in node.elts)
    if isinstance(node, ast.Dict):
        items = {}
        for key, value in zip(node.keys, node.values):
            if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
                raise NonLiteral("mapping key is not a string literal")
            items[key.value] = parse_node(value)
        return mapping(items)
    if isinstance(node, ast.Call) and _name_tail(node.func) in SPEC_TYPES:
        return _spec_call(node)
    raise NonLiteral(f"not a literal spec: {ast.unparse(node)}")


def parse_signature(source):
    """Parse a removed ``input_signature`` expression into one tree per parameter.

    The outer list or tuple is the signature itself, one element per non-``self`` parameter. Raises
    NonLiteral for anything that has to be evaluated to be known, such as ``dataset.element_spec``.
    """
    node = ast.parse(source.strip(), mode="eval").body
    if not isinstance(node, (ast.List, ast.Tuple)):
        raise NonLiteral(f"the signature is not a list or tuple: {ast.unparse(node)}")
    return [parse_node(e) for e in node.elts]


# --- Reading an evaluated value -----------------------------------------------------------------


def from_structure(structure):
    """Convert a ground-truth ``structure`` (``_list``/``_tuple``/dict/leaf) into a tree."""
    if "_list" in structure or "_tuple" in structure:
        return sequence(
            from_structure(e) for e in structure.get("_list", structure.get("_tuple"))
        )
    if "dtype" in structure and "shape" in structure:
        return leaf(
            structure["dtype"],
            structure["shape"],
            structure.get("spec_type", "TensorSpec"),
        )
    return mapping((k, from_structure(v)) for k, v in structure.items())


def signature_from_structure(structure):
    tree = from_structure(structure)
    if tree["kind"] != "sequence":
        raise ValueError("an evaluated signature must be a list or tuple")
    return tree["elements"]
