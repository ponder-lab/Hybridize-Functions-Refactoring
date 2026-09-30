import csv
import os
import tempfile
import unittest

import join
import spec

PRIMARY = [
    "subject",
    "function",
    "module",
    "relative path",
    "beginning line",
    "definition ordinal",
]


def write(directory, name, columns, rows):
    with open(os.path.join(directory, name), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(PRIMARY + columns if columns[0] != "subject" else columns)
        writer.writerows(rows)


# Where each synthetic function begins; the join finds a row by its location in the analyzed tree.
LINES = {"scored": 1, "called": 4, "lonely": 8, "mapped": 20, "serving_fn": 30}


def key(function, path="m.py", ordinal=1):
    return ["s", function, "m", path, LINES.get(function, 99), ordinal]


class AxisVerdictTest(unittest.TestCase):
    def verdict(self, *classes):
        rows = [
            {
                "param index": "0",
                "container position": "",
                "type ordinal": str(n),
                "rank": "1",
                "dim index": "0",
                "dim class": c,
            }
            for n, c in enumerate(classes)
        ]
        axes, _ = join.axis_evidence(rows)
        return join.axis_verdict(axes[("", 0)])

    def test_one_extent_is_unnecessary(self):
        self.assertEqual(self.verdict("Constant,4", "Constant,4"), "unnecessary")

    def test_two_extents_are_required(self):
        self.assertEqual(self.verdict("Constant,4", "Constant,8"), "required")

    def test_dynamic_is_required(self):
        self.assertEqual(self.verdict("Constant,4", "Dynamic"), "required")

    def test_unresolved_is_undetermined(self):
        self.assertEqual(self.verdict("Constant,4", "Unresolved"), "undetermined")

    def test_varying_extents_win_over_an_unresolved_member(self):
        self.assertEqual(
            self.verdict("Constant,4", "Constant,8", "Unresolved"), "required"
        )

    def test_an_unnamed_class_is_undetermined(self):
        self.assertEqual(self.verdict("Symbolic"), "undetermined")


class InferredTreeTest(unittest.TestCase):
    def test_leaf(self):
        self.assertEqual(
            join.inferred_tree({"dtype": "float32", "shape": "(None, 128)"}),
            spec.leaf("float32", [None, 128]),
        )
        self.assertEqual(
            join.inferred_tree({"dtype": "int32", "shape": "()"}),
            spec.leaf("int32", []),
        )
        self.assertEqual(
            join.inferred_tree({"dtype": "int32", "shape": "(20,)"}),
            spec.leaf("int32", [20]),
        )
        self.assertEqual(
            join.inferred_tree({"dtype": "int32", "shape": "None"}),
            spec.leaf("int32", None),
        )

    def test_sequence(self):
        tree = join.inferred_tree({"dtype": "float32", "shape": "[(2, 2), (3,)]"})
        self.assertEqual(
            tree["elements"], [spec.leaf("float32", [2, 2]), spec.leaf("float32", [3])]
        )
        self.assertFalse(tree["dtype_ambiguous"])

    def test_a_mixed_dtype_sequence_does_not_guess_positions(self):
        tree = join.inferred_tree({"dtype": "float32|int32", "shape": "[(2,), (3,)]"})
        self.assertTrue(tree["dtype_ambiguous"])
        self.assertEqual([e["dtype"] for e in tree["elements"]], [None, None])


class LibraryTest(unittest.TestCase):
    def test_paths(self):
        self.assertTrue(join.is_library("tf_image/core/bboxes/resize.py"))
        self.assertFalse(join.is_library("tests/test_resize.py"))
        self.assertFalse(
            join.is_library("integrations/tensorflow/e2e/matrix_ops_test.py")
        )
        self.assertFalse(
            join.is_library("examples/library/custom_transformer_training.py")
        )


class JoinSubjectTest(unittest.TestCase):
    """One subject, one function per outcome, against a synthetic evaluator output."""

    SUBJECT = {"repo": "r", "sha": "s", "kind": "signature"}

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.run = os.path.join(self.directory.name, "run")
        self.checkout = os.path.join(self.directory.name, "checkout")
        os.makedirs(self.run)
        os.makedirs(self.checkout)
        with open(os.path.join(self.checkout, "m.py"), "w") as f:
            f.write(
                "def scored(x):\n    pass\n\ndef called(x):\n    pass\n\ncalled(1)\n\ndef lonely(x):\n    pass\n"
            )
        functions = ["scored", "mapped", "called", "lonely", "serving_fn"]
        write(
            self.run,
            "functions.csv",
            ["method reference"],
            [key(f) + [""] for f in functions],
        )
        write(
            self.run,
            "parameters.csv",
            ["param index", "param name", "tensor types"],
            [key(f) + ["0", "x", ""] for f in functions],
        )
        write(
            self.run,
            "tensor_specs.csv",
            ["param index", "source", "dtype", "shape"],
            [key("scored") + ["0", "inferred", "float32", "(None, 3)"]],
        )
        write(
            self.run,
            "signature_absences.csv",
            ["param index", "source", "absence reason"],
            [
                key("called") + ["0", "absent", "UNKNOWN_DTYPE"],
                key("lonely") + ["", "absent", "NOT_ATTEMPTED"],
            ],
        )
        write(
            self.run,
            "parameter_dimensions.csv",
            [
                "param index",
                "param name",
                "is container",
                "container position",
                "type ordinal",
                "rank",
                "dim index",
                "dim class",
                "dtype",
                "dtype top",
            ],
            [
                key("scored")
                + [
                    "0",
                    "x",
                    "false",
                    "",
                    "0",
                    "2",
                    "0",
                    "Constant,2",
                    "FLOAT32",
                    "false",
                ],
                key("scored")
                + [
                    "0",
                    "x",
                    "false",
                    "",
                    "1",
                    "2",
                    "0",
                    "Constant,5",
                    "FLOAT32",
                    "false",
                ],
                key("scored")
                + [
                    "0",
                    "x",
                    "false",
                    "",
                    "0",
                    "2",
                    "1",
                    "Constant,3",
                    "FLOAT32",
                    "false",
                ],
                key("scored")
                + [
                    "0",
                    "x",
                    "false",
                    "",
                    "1",
                    "2",
                    "1",
                    "Constant,3",
                    "FLOAT32",
                    "false",
                ],
            ],
        )
        write(self.run, "calls.csv", ["subject", "callee", "expr"], [])
        write(
            self.run,
            "failed_preconditions.csv",
            ["refactoring", "severity", "code", "message"],
            [key("lonely") + ["", "3", "10", "m"]],
        )

    def tearDown(self):
        self.directory.cleanup()

    def record(self, qualname, removed, line):
        return {
            "file": "m.py",
            "qualname": qualname,
            "definition_ordinal": 1,
            "def_line": line,
            "stripped_first_line": line,
            "stripped_def_line": line,
            "signature_form": "keyword",
            "removed_source": removed,
            "decorator_callee": "tf.function",
            "other_arguments": [],
            "relaxation": {},
        }

    def test_outcomes(self):
        records = [
            self.record("scored", "[tf.TensorSpec([2, 3], tf.float32)]", 1),
            self.record("mapped", "[{'a': tf.TensorSpec([2])}]", 20),
            self.record("called", "[tf.TensorSpec([2])]", 4),
            self.record("lonely", "[tf.TensorSpec([2])]", 8),
            self.record("absent", "[tf.TensorSpec([2])]", 0),
            self.record("dynamic", "dataset.element_spec", 0),
        ]
        records[4]["qualname"] = "never_considered"
        functions, parameters, axes = join.join_subject(
            self.SUBJECT, records, self.run, self.checkout, trim={"sparse": True}
        )
        outcome = {r["function"]: r for r in functions}

        self.assertEqual(outcome["scored"]["outcome"], "scored")
        self.assertEqual(outcome["scored"]["relation"], spec.SUPPLIED_TIGHTER)
        self.assertEqual(outcome["scored"]["dtype relation"], spec.AGREEMENT)
        self.assertEqual(outcome["scored"]["shape relation"], spec.SUPPLIED_TIGHTER)
        self.assertEqual(outcome["mapped"]["outcome"], "not-reproduced:mapping")
        self.assertEqual(outcome["called"]["outcome"], "not-reproduced:UNKNOWN_DTYPE")
        self.assertEqual(outcome["called"]["text callers"], 1)
        # A trimmed checkout whose caller search is not known to be complete says so.
        self.assertEqual(outcome["lonely"]["outcome"], "no-call-site:trimmed")
        self.assertEqual(
            outcome["lonely"]["failed preconditions"], "UNDETERMINABLE_TENSOR_PARAMETER"
        )
        self.assertEqual(
            outcome["never_considered"]["outcome"],
            "excluded:not-considered-by-the-tool",
        )
        self.assertTrue(
            outcome["dynamic"]["outcome"].startswith("excluded:unevaluable")
        )

        by_dim = {(a["function"], a["dim index"]): a for a in axes}
        self.assertEqual(by_dim[("scored", 0)]["verdict"], "required")
        self.assertEqual(by_dim[("scored", 0)]["extents"], "2|5")
        self.assertEqual(by_dim[("scored", 0)]["removed dim"], "2")
        self.assertEqual(by_dim[("scored", 1)]["verdict"], "unnecessary")
        self.assertEqual(
            [p["param name"] for p in parameters if p["function"] == "scored"], ["x"]
        )

    def test_the_evaluator_name_is_found_by_location(self):
        # The evaluator does not qualify a function nested in a block within a function (#992).
        functions, _, _ = join.join_subject(
            self.SUBJECT,
            [self.record("main.serving_fn", "[tf.TensorSpec([2])]", 30)],
            self.run,
            self.checkout,
        )
        self.assertEqual(functions[0]["evaluator function"], "serving_fn")
        self.assertNotEqual(
            functions[0]["outcome"], "excluded:not-considered-by-the-tool"
        )

    def test_a_different_function_at_the_line_is_not_taken(self):
        functions, _, _ = join.join_subject(
            self.SUBJECT,
            [self.record("other", "[tf.TensorSpec([2])]", 30)],
            self.run,
            self.checkout,
        )
        self.assertEqual(functions[0]["outcome"], "excluded:not-considered-by-the-tool")

    def test_complete_scope_is_in_tree(self):
        records = [self.record("lonely", "[tf.TensorSpec([2])]", 8)]
        functions, _, _ = join.join_subject(
            self.SUBJECT,
            records,
            self.run,
            self.checkout,
            trim={"sparse": True, "caller_scope_complete": True},
        )
        self.assertEqual(functions[0]["outcome"], "no-call-site:in-tree")

    def test_ground_truth_replaces_a_non_literal(self):
        records = [self.record("scored", "dataset.element_spec", 1)]
        truth = {
            "scored": {
                "structure": {"_list": [{"dtype": "float32", "shape": [None, 3]}]}
            },
            "obtained_by": "ran it",
        }
        functions, _, _ = join.join_subject(
            self.SUBJECT, records, self.run, self.checkout, ground_truth=truth
        )
        self.assertEqual(functions[0]["outcome"], "scored")
        self.assertEqual(functions[0]["relation"], spec.AGREEMENT)
        self.assertEqual(functions[0]["obtained by"], "evaluated: ran it")


class PreconditionNamesTest(unittest.TestCase):
    def test_read_from_the_enum(self):
        names = join.precondition_names()
        self.assertEqual(names[10], "UNDETERMINABLE_TENSOR_PARAMETER")
        self.assertEqual(names[3], "UNDETERMINABLE_SIDE_EFFECTS")


if __name__ == "__main__":
    unittest.main()
