import os
import tempfile
import textwrap
import unittest

import strip


def source(text):
    return textwrap.dedent(text).lstrip("\n").encode("utf-8")


class StripKeywordTest(unittest.TestCase):
    def strip(self, text, qualname="f", **kwargs):
        return strip.strip_function(source(text), qualname, **kwargs)

    def test_sole_argument(self):
        data, record = self.strip("""
            @tf.function(input_signature=[tf.TensorSpec([None], tf.float32)])
            def f(x):
                return x
            """)
        self.assertEqual(data, source("@tf.function()\ndef f(x):\n    return x\n"))
        self.assertEqual(
            record["removed_source"], "[tf.TensorSpec([None], tf.float32)]"
        )
        self.assertEqual(record["signature_form"], "keyword")

    def test_sole_argument_with_a_trailing_comma(self):
        data, _ = self.strip("""
            @tf.function(
                input_signature=[tf.TensorSpec([2])],
            )
            def f(x):
                return x
            """)
        self.assertNotIn(b"input_signature", data)
        self.assertIn(b"@tf.function(\n    \n)", data)

    def test_first_of_several_keeps_the_rest(self):
        data, record = self.strip("""
            @tf.function(input_signature=[tf.TensorSpec([2])], experimental_relax_shapes=True)
            def f(x):
                return x
            """)
        self.assertIn(b"@tf.function(experimental_relax_shapes=True)", data)
        self.assertEqual(record["relaxation"], {"experimental_relax_shapes": "True"})
        self.assertEqual(record["other_arguments"], ["experimental_relax_shapes=True"])

    def test_last_of_several_across_lines(self):
        data, _ = self.strip("""
            @tf.function(jit_compile=True,
                         input_signature=[tf.TensorSpec([None, None], dtype=tf.int32),
                                          tf.TensorSpec([None, ], dtype=tf.int32)])
            def f(x, y):
                return x
            """)
        self.assertIn(b"@tf.function(jit_compile=True)\ndef f(x, y):", data)

    def test_middle_keyword(self):
        data, _ = self.strip("""
            @tf.function(autograph=False, input_signature=[], jit_compile=True)
            def f():
                return 1
            """)
        self.assertIn(b"@tf.function(autograph=False, jit_compile=True)", data)

    def test_other_decorators_and_callees_survive(self):
        data, record = self.strip("""
            @convert_type
            @def_function.function(input_signature=[tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32)])
            def f(x):
                return x
            """)
        self.assertIn(b"@convert_type\n@def_function.function()\n", data)
        self.assertEqual(record["decorator_callee"], "def_function.function")
        self.assertEqual(record["decorator_index"], 1)

    def test_multibyte_text_before_the_keyword(self):
        # AST column offsets count UTF-8 bytes, so a non-ASCII string earlier on the line must not shift the cut.
        data, _ = self.strip("""
            @tf.function(experimental_implements="é∑", input_signature=[tf.TensorSpec([1])])
            def f(x):
                return x
            """)
        self.assertIn(
            '@tf.function(experimental_implements="é∑")'.encode("utf-8"), data
        )

    def test_positional_signature_is_recorded_not_stripped(self):
        text = """
            @tf.function(None, [tf.TensorSpec([2])])
            def f(x):
                return x
            """
        data, record = self.strip(text)
        self.assertEqual(data, source(text))
        self.assertEqual(record["signature_form"], "positional")
        self.assertEqual(record["removed_source"], "[tf.TensorSpec([2])]")

    def test_no_signature(self):
        text = """
            @tf.function(experimental_relax_shapes=True)
            def f(x):
                return x
            """
        data, record = self.strip(text)
        self.assertEqual(data, source(text))
        self.assertEqual(record["signature_form"], "none")
        self.assertIsNone(record["removed_source"])

    def test_bare_decorator(self):
        _, record = self.strip("@tf.function\ndef f(x):\n    return x\n")
        self.assertEqual(record["signature_form"], "none")
        self.assertEqual(record["decorator_callee"], "tf.function")


class StripEdgeTest(unittest.TestCase):
    def strip(self, text, qualname="f", **kwargs):
        return strip.strip_function(source(text), qualname, **kwargs)

    def test_a_parenthesized_previous_argument_keeps_its_parenthesis(self):
        data, _ = self.strip(
            """
            @tf.function((f), input_signature=[tf.TensorSpec([2])])
            def g(x):
                return x
            """,
            "g",
        )
        self.assertIn(b"@tf.function((f))\n", data)

    def test_a_comment_after_the_keyword_survives(self):
        data, _ = self.strip("""
            @tf.function(input_signature=[tf.TensorSpec([2])],  # the signature
                         jit_compile=True)
            def f(x):
                return x
            """)
        self.assertIn(b"# the signature", data)
        self.assertIn(b"jit_compile=True)", data)
        self.assertNotIn(b"input_signature", data)

    def test_a_line_inside_a_multiline_header_names_the_definition(self):
        tree = strip.parse(source("@tf.function\ndef f(a,\n      b):\n    return a\n"))
        self.assertEqual(strip.find_definition(tree, "f", line=3)[0], 1)

    def test_the_verification_catches_a_wrong_edit(self):
        real = strip.strip_keyword
        try:
            strip.strip_keyword = lambda data, call, keyword: real(
                data, call, keyword
            ).replace(b"return x", b"return y")
            with self.assertRaises(AssertionError):
                self.strip(
                    "@tf.function(input_signature=[tf.TensorSpec([2])])\ndef f(x):\n    return x\n"
                )
        finally:
            strip.strip_keyword = real


class StripFileGuardTest(unittest.TestCase):
    TEXT = source(
        "@tf.function(input_signature=[tf.TensorSpec([2])])\ndef f(x):\n    return x\n"
    )

    def test_a_symbolic_link_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            target, link = os.path.join(directory, "t.py"), os.path.join(
                directory, "l.py"
            )
            with open(target, "wb") as f:
                f.write(self.TEXT)
            os.symlink(target, link)
            with self.assertRaises(ValueError):
                strip.strip_file(link, [{"qualname": "f", "line": 2}])
            with open(target, "rb") as f:
                self.assertEqual(f.read(), self.TEXT)

    def test_a_function_listed_twice_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "m.py")
            with open(path, "wb") as f:
                f.write(self.TEXT)
            with self.assertRaises(ValueError):
                strip.strip_file(
                    path, [{"qualname": "f", "line": 2}, {"qualname": "f", "line": 2}]
                )


class ResolveNameTest(unittest.TestCase):
    def test_a_unique_module_binding_resolves(self):
        data, record = strip.strip_function(
            source("""
                sig = [tf.TensorSpec(shape=(None, None), dtype=tf.int32, name="Inputs")]

                @tf.function(input_signature=sig)
                def f(x):
                    return x
                """),
            "f",
        )
        self.assertEqual(record["removed_source"], "sig")
        self.assertEqual(record["resolved"]["line"], 1)
        self.assertIn("TensorSpec(shape=(None, None)", record["resolved"]["source"])

    def test_a_name_bound_twice_does_not_resolve(self):
        _, record = strip.strip_function(
            source("""
                sig = [tf.TensorSpec([1])]
                sig = [tf.TensorSpec([2])]

                @tf.function(input_signature=sig)
                def f(x):
                    return x
                """),
            "f",
        )
        self.assertIsNone(record["resolved"])

    def test_an_attribute_does_not_resolve(self):
        _, record = strip.strip_function(
            source("""
                @tf.function(input_signature=dataset.element_spec)
                def f(x):
                    return x
                """),
            "f",
        )
        self.assertIsNone(record["resolved"])


class LookupTest(unittest.TestCase):
    TWINS = source("""
        class C:
            def m(self):
                class Model:
                    @tf.function(input_signature=[tf.TensorSpec([1])])
                    def update(self, v):
                        pass

        def outer():
            @tf.function(input_signature=[tf.TensorSpec([2])])
            def inner(x):
                return x
            return inner

        if flag:
            def f(x):
                return x
        else:
            @tf.function(input_signature=[tf.TensorSpec([3])])
            def f(x):
                return x
        """)

    def test_lexical_qualified_names(self):
        names = strip.qualified_definitions(strip.parse(self.TWINS))
        self.assertIn("C.m.Model.update", names)
        self.assertIn("outer.inner", names)
        self.assertEqual(len(names["f"]), 2)

    def test_ordinal_selects_a_twin(self):
        data, record = strip.strip_function(self.TWINS, "f", ordinal=2)
        self.assertEqual(record["definition_ordinal"], 2)
        self.assertEqual(record["removed_source"], "[tf.TensorSpec([3])]")
        self.assertIn(b"TensorSpec([1])", data)
        self.assertIn(b"TensorSpec([2])", data)

    def test_a_decorator_line_finds_its_definition(self):
        ordinal, node = strip.find_definition(
            strip.parse(self.TWINS), "C.m.Model.update", line=4
        )
        self.assertEqual((ordinal, node.lineno), (1, 5))

    def test_an_ambiguous_or_missing_name_fails(self):
        with self.assertRaises(LookupError):
            strip.find_definition(strip.parse(self.TWINS), "missing")
        with self.assertRaises(LookupError):
            strip.find_definition(strip.parse(self.TWINS), "f", ordinal=3)


class StripFileTest(unittest.TestCase):
    def test_two_functions_in_one_file(self):
        # Stripping the first multi-line signature moves the second function up; it must still be found.
        text = source("""
            @tf.function(input_signature=[
                tf.TensorSpec([None, 2]),
                tf.TensorSpec([None, 2]),
            ])
            def a(x, y):
                return x

            @tf.function(input_signature=[tf.TensorSpec([3])])
            def b(x):
                return x
            """)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "m.py")
            with open(path, "wb") as f:
                f.write(text)
            records = strip.strip_file(
                path, [{"qualname": "a", "line": 5}, {"qualname": "b", "line": 9}]
            )
            with open(path, "rb") as f:
                stripped = f.read()
        self.assertEqual([r["qualname"] for r in records], ["a", "b"])
        self.assertEqual(records[1]["removed_source"], "[tf.TensorSpec([3])]")
        self.assertNotIn(b"input_signature", stripped)
        self.assertEqual(stripped.count(b"@tf.function()"), 2)


if __name__ == "__main__":
    unittest.main()
