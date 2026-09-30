import os
import unittest

import spec

CASES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "relation_cases.txt")


def cases():
    with open(CASES) as f:
        for number, line in enumerate(f, 1):
            if line.strip() and not line.startswith("#"):
                yield number, [field.strip() for field in line.split(";")]


class RelationCasesTest(unittest.TestCase):
    """The shared table that InputSignatureTest also reads, so the two orders cannot drift apart."""

    def test_table_is_not_empty(self):
        self.assertGreater(len(list(cases())), 10)

    def test_each_axis(self):
        for number, (supplied, inferred, both, dtype, shape) in cases():
            s, i = spec.parse_rendered_signature(
                supplied
            ), spec.parse_rendered_signature(inferred)
            with self.subTest(line=number):
                self.assertEqual(spec.relate_signature(s, i, "both"), both)
                self.assertEqual(spec.relate_signature(s, i, "dtype"), dtype)
                self.assertEqual(spec.relate_signature(s, i, "shape"), shape)

    def test_axes_combine_to_the_relation(self):
        for number, (_, _, both, dtype, shape) in cases():
            with self.subTest(line=number):
                self.assertEqual(spec.combine(dtype, shape), both)

    def test_rendering_round_trips(self):
        for number, (supplied, inferred, *_) in cases():
            for text in (supplied, inferred):
                with self.subTest(line=number, text=text):
                    rendered = " | ".join(
                        spec.render(t) for t in spec.parse_rendered_signature(text)
                    )
                    self.assertEqual(rendered, text)


class CombineTest(unittest.TestCase):
    def test_lattice(self):
        A, T, B, I = (
            spec.AGREEMENT,
            spec.SUPPLIED_TIGHTER,
            spec.SUPPLIED_BROADER,
            spec.INCOMPARABLE,
        )
        for x in (A, T, B, I):
            self.assertEqual(spec.combine(A, x), x)
            self.assertEqual(spec.combine(x, x), x)
            self.assertEqual(spec.combine(I, x), I)
        self.assertEqual(spec.combine(T, B), I)
        self.assertEqual(spec.combine(B, T), I)


class MappingTest(unittest.TestCase):
    def test_a_mapping_has_no_relation(self):
        removed = spec.mapping({"ids": spec.leaf("int64", [None, None])})
        self.assertIsNone(spec.relate_entry(removed, spec.leaf("int64", [None, None])))
        self.assertIsNone(
            spec.relate_signature([removed], [spec.leaf("int64", [None, None])])
        )
        self.assertTrue(spec.contains_mapping(spec.sequence([removed])))

    def test_leaves_spell_their_paths(self):
        tree = spec.sequence(
            [
                spec.mapping(
                    {"b": spec.leaf("int32", []), "a": spec.leaf("string", [None])}
                )
            ]
        )
        self.assertEqual([p for p, _ in spec.leaves(tree)], ["[0].a", "[0].b"])


class ParseSignatureTest(unittest.TestCase):
    def test_positional_and_keyword_arguments(self):
        parsed = spec.parse_signature(
            "[tf.TensorSpec([None, 4], tf.float32), tf.TensorSpec(shape=(), dtype=tf.int32, name='n')]"
        )
        self.assertEqual(
            parsed, [spec.leaf("float32", [None, 4]), spec.leaf("int32", [])]
        )

    def test_defaults_and_unknown_rank(self):
        self.assertEqual(
            spec.parse_signature("[tf.TensorSpec(None)]"), [spec.leaf("float32", None)]
        )
        self.assertEqual(
            spec.parse_signature("[tf.TensorSpec(shape=[None, ])]"),
            [spec.leaf("float32", [None])],
        )

    def test_tensor_shape_and_string_dtype(self):
        parsed = spec.parse_signature(
            "(TensorSpec(tf.TensorShape([2, None]), 'bool'),)"
        )
        self.assertEqual(parsed, [spec.leaf("bool", [2, None])])

    def test_ragged_and_nested(self):
        parsed = spec.parse_signature(
            "[tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64), [tf.TensorSpec([2])]]"
        )
        self.assertEqual(parsed[0]["spec_type"], "RaggedTensorSpec")
        self.assertEqual(parsed[1], spec.sequence([spec.leaf("float32", [2])]))

    def test_dict(self):
        parsed = spec.parse_signature(
            "[{'image': tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32)}]"
        )
        self.assertEqual(
            parsed, [spec.mapping({"image": spec.leaf("float32", [None, None, 3])})]
        )

    def test_non_literals(self):
        for source in (
            "dataset.element_spec",
            "(dataset.element_spec,)",
            "[spec]",
            "[tf.TensorSpec(shape=s)]",
            "[tf.TensorSpec([n])]",
        ):
            with self.subTest(source=source):
                with self.assertRaises(spec.NonLiteral):
                    spec.parse_signature(source)

    def test_a_negative_dimension_is_not_read_as_a_wildcard(self):
        with self.assertRaises(spec.NonLiteral):
            spec.parse_signature("[tf.TensorSpec([-1, 3])]")


class StructureTest(unittest.TestCase):
    def test_evaluated_value(self):
        structure = {
            "_tuple": [{"ids": {"dtype": "int64", "shape": [None, None], "name": None}}]
        }
        self.assertEqual(
            spec.signature_from_structure(structure),
            [spec.mapping({"ids": spec.leaf("int64", [None, None])})],
        )

    def test_evaluated_list_of_leaves(self):
        structure = {
            "_list": [{"dtype": "uint8", "shape": [1, None, None, 3], "name": "images"}]
        }
        self.assertEqual(
            spec.signature_from_structure(structure),
            [spec.leaf("uint8", [1, None, None, 3])],
        )


if __name__ == "__main__":
    unittest.main()
