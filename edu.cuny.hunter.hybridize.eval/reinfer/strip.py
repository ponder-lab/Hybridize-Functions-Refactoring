"""Strip exactly the ``input_signature`` argument from named ``tf.function`` decorators.

Functions are named the way the evaluator names them: the full lexical qualified name (classes and
enclosing functions) and the definition ordinal, counted per qualified name by the line of the ``def``
(the rule ``Function.getDefinitionOrdinal`` documents in edu.cuny.hunter.hybridize.core). The beginning line is never
the key, because removing a multi-line signature moves every line below it.

The edit is made on the source text, cutting the keyword's exact span, so formatting, comments and
every other decorator argument survive. It is then verified on the AST: the original module with that
one keyword deleted must equal the edited module, node for node.
"""

import ast
import collections
import os
import warnings

SIGNATURE_KEYWORD = "input_signature"
RELAXATION_KEYWORDS = ("experimental_relax_shapes", "reduce_retracing")


def parse(source):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.parse(source)


def qualified_definitions(tree):
    """Map each lexically qualified name to its ``def`` nodes in source order (clauses 1 to 6)."""
    definitions = collections.defaultdict(list)

    def walk(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                walk(child, prefix + child.name + ".")
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                definitions[prefix + child.name].append(child)
                walk(child, prefix + child.name + ".")
            else:
                walk(child, prefix)

    walk(tree, "")
    return {
        name: sorted(nodes, key=lambda n: n.lineno)
        for name, nodes in definitions.items()
    }


def find_definition(tree, qualname, line=None, ordinal=None):
    """Return ``(ordinal, node)`` for a definition named by ``qualname`` and either its ordinal or a line.

    A line matches the ``def`` line or any line of a decorator above it, since a mined diff hunk may
    point at either. Raises LookupError when nothing, or more than one definition, matches.
    """
    nodes = qualified_definitions(tree).get(qualname, [])
    if ordinal is not None:
        if not 1 <= ordinal <= len(nodes):
            raise LookupError(
                f"{qualname} has no definition ordinal {ordinal} ({len(nodes)} found)"
            )
        return ordinal, nodes[ordinal - 1]
    matches = []
    for index, node in enumerate(nodes):
        first = min([node.lineno] + [d.lineno for d in node.decorator_list])
        # The header runs from the first decorator to the line before the body, so a line inside a
        # multi-line parameter list still names this definition.
        last = max(node.lineno, node.body[0].lineno - 1) if node.body else node.lineno
        if line is None or first <= line <= last:
            matches.append((index + 1, node))
    if len(matches) != 1:
        raise LookupError(
            f"{qualname} at line {line}: {len(matches)} of {len(nodes)} definitions match"
        )
    return matches[0]


def _tail(node):
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


def signature_decorator(node):
    """Find the ``*.function(...)`` decorator of ``node`` and describe its signature argument.

    Returns ``(index, call, form)``, where ``form`` is ``"keyword"`` (``input_signature=...``),
    ``"positional"`` (the second positional argument of ``function(func, input_signature)``) or
    ``"none"``. ``(None, None, "none")`` when no ``*.function`` decorator is present.
    """
    for index, decorator in enumerate(node.decorator_list):
        call = decorator if isinstance(decorator, ast.Call) else None
        func = call.func if call else decorator
        if _tail(func) != "function":
            continue
        if call is None:
            return index, None, "none"
        if any(k.arg == SIGNATURE_KEYWORD for k in call.keywords):
            return index, call, "keyword"
        if len(call.args) >= 2:
            return index, call, "positional"
        return index, call, "none"
    return None, None, "none"


class _Offsets:
    """Byte offsets for AST positions, which count UTF-8 bytes within a line."""

    def __init__(self, data):
        self.starts = [0]
        for line in data.splitlines(keepends=True):
            self.starts.append(self.starts[-1] + len(line))

    def at(self, lineno, col):
        return self.starts[lineno - 1] + col


def _start(node):
    return node.lineno, node.col_offset


def _end(node):
    return node.end_lineno, node.end_col_offset


_SPACE = (b" ", b"\t")


def _comma_after(data, position):
    """The offset just past the comma that follows ``position``, skipping whitespace and comments, or
    None if something else comes first."""
    while position < len(data):
        char = data[position : position + 1]
        if char in _SPACE or char in (b"\n", b"\r"):
            position += 1
        elif char == b"#":
            while position < len(data) and data[position : position + 1] not in (
                b"\n",
                b"\r",
            ):
                position += 1
        elif char == b",":
            return position + 1
        else:
            return None
    return None


def _comma_before(data, position):
    """The offset of the comma that precedes ``position``, skipping whitespace (not comments), or None."""
    position -= 1
    while position >= 0 and data[position : position + 1] in _SPACE + (b"\n", b"\r"):
        position -= 1
    return position if position >= 0 and data[position : position + 1] == b"," else None


def strip_keyword(data, call, keyword):
    """Return ``data`` (bytes) with ``keyword`` and its separating comma cut out of ``call``.

    The separating comma is found in the source, not inferred from the neighbouring nodes' extents: a
    node's extent excludes enclosing parentheses, so cutting from the previous argument's end would eat
    the `)` of `(f), input_signature=...`. Whitespace after the comma on the same line goes with it, and
    anything after a line break, comments included, stays.
    """
    offsets = _Offsets(data)
    elements = sorted(list(call.args) + list(call.keywords), key=_start)
    index = elements.index(keyword)
    start, finish = offsets.at(*_start(keyword)), offsets.at(*_end(keyword))
    if index + 1 < len(elements):
        comma = _comma_after(data, finish)
        if comma is None:
            raise ValueError("no comma after the keyword")
        end = comma
        while end < len(data) and data[end : end + 1] in _SPACE:
            end += 1
        if data.rfind(b"#", finish, comma) != -1:
            end = comma  # a comment between keyword and comma: cut only up to the comma, not beyond.
        begin = start
    elif index > 0:
        comma = _comma_before(data, start)
        if comma is None:
            raise ValueError("no comma before the keyword")
        begin, end = comma, finish
    else:
        begin, end = offsets.at(*_start(keyword)), offsets.at(*_end(keyword))
        rest = end
        while rest < len(data) and data[rest : rest + 1] in (b" ", b"\t", b"\n", b"\r"):
            rest += 1
        if data[rest : rest + 1] == b",":
            end = (
                rest + 1
            )  # a sole argument's trailing comma would leave `function(,)`.
    return data[:begin] + data[end:]


def _source_of(data, node):
    offsets = _Offsets(data)
    return data[offsets.at(*_start(node)) : offsets.at(*_end(node))].decode("utf-8")


def resolve_module_name(data, tree, value):
    """Resolve a signature written as a bare name to the one module-level assignment that binds it.

    This is the one level of indirection the evaluator's own supplied-signature model follows (a name
    whose referent is itself a name is not chased). Returns ``{"name", "line", "source"}``, or None when
    the value is not a bare name, or the module binds that name other than exactly once.
    """
    if not isinstance(value, ast.Name):
        return None
    bindings = [
        statement
        for statement in tree.body
        if isinstance(statement, (ast.Assign, ast.AnnAssign))
        and any(
            isinstance(target, ast.Name) and target.id == value.id
            for target in (
                statement.targets
                if isinstance(statement, ast.Assign)
                else [statement.target]
            )
        )
    ]
    if len(bindings) != 1 or bindings[0].value is None:
        return None
    return {
        "name": value.id,
        "line": bindings[0].lineno,
        "source": _source_of(data, bindings[0].value),
    }


def strip_function(data, qualname, line=None, ordinal=None):
    """Strip one function's signature keyword. Returns ``(new_data, record)``.

    The record names the function by qualified name and ordinal, and carries the removed expression's
    source text, the decorator's callee and every other argument, and the form of the signature. For a
    positional or absent signature nothing is removed and ``new_data`` is ``data``.
    """
    tree = parse(data)
    ordinal, node = find_definition(tree, qualname, line=line, ordinal=ordinal)
    index, call, form = signature_decorator(node)
    record = {
        "qualname": qualname,
        "definition_ordinal": ordinal,
        "def_line": node.lineno,
        "decorator_index": index,
        "decorator_callee": (
            ast.unparse(call.func if call else node.decorator_list[index])
            if index is not None
            else None
        ),
        "signature_form": form,
        "removed_source": None,
        "other_arguments": [],
        "relaxation": {},
    }
    if call is not None:
        record["other_arguments"] = [ast.unparse(a) for a in call.args] + [
            f"{k.arg}={ast.unparse(k.value)}"
            for k in call.keywords
            if k.arg != SIGNATURE_KEYWORD
        ]
        record["relaxation"] = {
            k.arg: ast.unparse(k.value)
            for k in call.keywords
            if k.arg in RELAXATION_KEYWORDS
        }
    if form == "positional":
        record["removed_source"] = _source_of(data, call.args[1])
    if form != "keyword":
        return data, record

    keyword = next(k for k in call.keywords if k.arg == SIGNATURE_KEYWORD)
    record["removed_source"] = _source_of(data, keyword.value)
    record["resolved"] = resolve_module_name(data, tree, keyword.value)
    new_data = strip_keyword(data, call, keyword)

    # Verify: deleting that one keyword from the original AST must give the edited file's AST.
    call.keywords.remove(keyword)
    expected = ast.dump(tree, include_attributes=False)
    actual = ast.dump(parse(new_data), include_attributes=False)
    if expected != actual:
        raise AssertionError(
            f"stripping {qualname} changed more than its {SIGNATURE_KEYWORD} argument"
        )
    return new_data, record


def strip_file(path, functions):
    """Strip each ``{"qualname", "line"}`` in ``functions`` from the file at ``path``, in place.

    Functions are located and stripped one at a time against the file as it stands, keyed by ordinal
    after the first lookup, so an earlier strip moving lines cannot misdirect a later one.
    """
    if os.path.islink(path):
        # Writing through a link could reach outside the copy, into the checkout it was made from.
        raise ValueError(f"{path} is a symbolic link; refusing to strip through it")
    with open(path, "rb") as f:
        data = f.read()
    located = []
    tree = parse(data)
    for function in functions:
        ordinal, _ = find_definition(
            tree, function["qualname"], line=function.get("line")
        )
        if any(
            o == ordinal and f["qualname"] == function["qualname"] for f, o in located
        ):
            raise ValueError(
                f"{function['qualname']} (ordinal {ordinal}) is listed twice"
            )
        located.append((function, ordinal))
    records = []
    for function, ordinal in located:
        data, record = strip_function(data, function["qualname"], ordinal=ordinal)
        records.append(record)
    # Where each function begins in the file as the evaluator will read it: its first decorator line, or
    # its def line. The evaluator's own name for a function can differ from the lexical one (#992), so
    # the join finds its row by this line in the same tree, never across trees.
    final = parse(data)
    for record in records:
        _, node = find_definition(
            final, record["qualname"], ordinal=record["definition_ordinal"]
        )
        record["stripped_first_line"] = min(
            [node.lineno] + [d.lineno for d in node.decorator_list]
        )
        record["stripped_def_line"] = node.lineno
    with open(path, "wb") as f:
        f.write(data)
    return records
