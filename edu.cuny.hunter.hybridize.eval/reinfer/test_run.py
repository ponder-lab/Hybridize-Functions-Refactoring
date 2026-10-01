import os
import shutil
import subprocess
import tempfile
import unittest

import run


class WriteCsvTest(unittest.TestCase):
    def test_no_rows_still_writes_the_header(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "reinfer_axes.csv")
            run.write_csv(path, [], run.join.COLUMNS["axes"])
            with open(path) as f:
                self.assertEqual(f.read().strip(), ",".join(run.join.COLUMNS["axes"]))


class FailureOfTest(unittest.TestCase):
    def test_the_completion_line_is_the_witness(self):
        self.assertIsNone(run.failure_of("...\n!MESSAGE " + run.SUCCESS_LINE + "\n", 1))

    def test_out_of_memory(self):
        self.assertEqual(
            run.failure_of("java.lang.OutOfMemoryError: Java heap space\n", 13),
            "OutOfMemoryError",
        )

    def test_stack_overflow(self):
        self.assertEqual(
            run.failure_of(
                "!MESSAGE Application error\njava.lang.StackOverflowError\n", 13
            ),
            "StackOverflowError",
        )

    def test_a_skipped_project_names_its_exception(self):
        log = (
            "!MESSAGE Evaluation completed: 0 of 1 project(s) succeeded, 1 failed and were skipped: "
            "p (CoreException: Project p has a script, a.py, that none of its PYTHONPATH entries cover)\n"
        )
        self.assertEqual(run.failure_of(log, 1), "CoreException")

    def test_an_unexplained_stop(self):
        self.assertEqual(
            run.failure_of("nothing useful\n", 7), "did-not-complete (launcher exit 7)"
        )


class ProjectNameTest(unittest.TestCase):
    def test_a_committed_project_file_names_the_project(self):
        with tempfile.TemporaryDirectory() as d:
            work = os.path.join(d, "gpt-2-tensorflow2.0-88dfc145b")
            os.makedirs(work)
            with open(os.path.join(work, ".project"), "w") as f:
                f.write(
                    "<projectDescription>\n\t<name>gpt-2-tensorflow2.0</name>\n</projectDescription>\n"
                )
            self.assertEqual(run.project_name(work), "gpt-2-tensorflow2.0")

    def test_without_one_the_directory_names_it(self):
        with tempfile.TemporaryDirectory() as d:
            work = os.path.join(d, "tf-image-15977b3c6")
            os.makedirs(work)
            self.assertEqual(run.project_name(work), "tf-image-15977b3c6")


class CopySparseTest(unittest.TestCase):
    """The selection is git's own non-cone sparse checkout, directory matches included."""

    FILES = [
        "a/x.py",
        "a/y.txt",
        "a/b/x.py",
        "a/b/c/x.py",
        "a/d.py/inner.txt",
        "ab.py",
        "x.py",
        "b/x.py",
    ]

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.source = os.path.join(self.directory.name, "source")
        for path in self.FILES:
            os.makedirs(
                os.path.dirname(os.path.join(self.source, path)) or self.source,
                exist_ok=True,
            )
            with open(os.path.join(self.source, path), "w") as f:
                f.write("x = 1\n")
        with open(os.path.join(self.source, ".gitattributes"), "w") as f:
            f.write("a/x.py export-ignore\n")
        git = ["git", "-C", self.source, "-c", "user.name=t", "-c", "user.email=t@t"]
        subprocess.run(["git", "init", "-q", self.source], check=True)
        subprocess.run(git + ["add", "."], check=True)
        subprocess.run(git + ["commit", "-q", "-m", "c"], check=True)

    def tearDown(self):
        self.directory.cleanup()

    def copy(self, patterns):
        work = os.path.join(self.directory.name, "work")
        shutil.rmtree(work, ignore_errors=True)
        run.copy_sparse(self.source, self.source, work, patterns)
        return sorted(
            os.path.relpath(os.path.join(d, f), work)
            for d, _, fs in os.walk(work)
            for f in fs
            if f != ".gitattributes"
        )

    def test_selections_follow_git(self):
        # Each expectation is what `git sparse-checkout set --no-cone` materializes: a pattern that
        # matches a directory takes everything under it, and an export rule changes nothing.
        cases = {
            ("/a/**",): [
                "a/b/c/x.py",
                "a/b/x.py",
                "a/d.py/inner.txt",
                "a/x.py",
                "a/y.txt",
            ],
            ("/a/*.py",): ["a/d.py/inner.txt", "a/x.py"],
            ("/a/b",): ["a/b/c/x.py", "a/b/x.py"],
            ("/x.py", "/a/b"): ["a/b/c/x.py", "a/b/x.py", "x.py"],
            ("/a/**/*.py",): ["a/b/c/x.py", "a/b/x.py", "a/d.py/inner.txt", "a/x.py"],
        }
        for patterns, expected in cases.items():
            with self.subTest(patterns=patterns):
                self.assertEqual(self.copy(list(patterns)), expected)

    def test_a_selection_of_nothing_is_refused(self):
        with self.assertRaises(RuntimeError):
            self.copy(["/missing.py"])

    def test_a_subdirectory_of_a_repository_is_refused(self):
        with self.assertRaises(RuntimeError):
            run.copy_sparse(
                self.source,
                os.path.join(self.source, "a"),
                os.path.join(self.directory.name, "w"),
                ["/x.py"],
            )


class TrimTest(unittest.TestCase):
    SUBJECT = {
        "path": "p",
        "notes": "the subject's own notes",
        "sparse": ["/a.py"],
        "caller_scope_complete": True,
        "trim_note": "why",
    }

    def test_the_trim_is_read_from_the_manifest_entry(self):
        self.assertEqual(
            run.trim_of(self.SUBJECT, {}),
            {"sparse": ["/a.py"], "caller_scope_complete": True, "note": "why"},
        )

    def test_a_local_source_is_added(self):
        self.assertEqual(run.trim_of(self.SUBJECT, {"source": "/s"})["source"], "/s")

    def test_a_malformed_trim_is_refused(self):
        for field, value in (
            ("sparse", True),
            ("sparse", "/a.py"),
            ("sparse", []),
            ("exclude", "tensorflow1"),
            ("caller_scope_complete", "yes"),
            ("trim_note", 1),
        ):
            with self.subTest(field=field, value=value):
                with self.assertRaises(RuntimeError):
                    run.trim_of({**self.SUBJECT, field: value}, {})

    def test_a_trim_field_in_the_local_config_is_refused(self):
        for local in ({"sparse": ["/b.py"]}, {"exclude": ["x"]}, {"note": "n"}):
            with self.subTest(local=local):
                with self.assertRaises(RuntimeError):
                    run.trim_of(self.SUBJECT, local)


class PrepareTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.source = os.path.join(self.directory.name, "source")
        for path in ("keep/a.py", "tensorflow1/b.py", "top.py"):
            os.makedirs(os.path.dirname(os.path.join(self.source, path)), exist_ok=True)
            with open(os.path.join(self.source, path), "w") as f:
                f.write("x = 1\n")
        git = ["git", "-C", self.source, "-c", "user.name=t", "-c", "user.email=t@t"]
        subprocess.run(["git", "init", "-q", self.source], check=True)
        subprocess.run(git + ["add", "."], check=True)
        subprocess.run(git + ["commit", "-q", "-m", "c"], check=True)
        head = subprocess.run(
            git + ["rev-parse", "HEAD"], check=True, capture_output=True, text=True
        )
        self.subject = {"sha": head.stdout.strip()}

    def tearDown(self):
        self.directory.cleanup()

    def test_an_excluded_tree_is_left_out(self):
        work = os.path.join(self.directory.name, "work")
        run.prepare(self.subject, self.source, work, exclude=["tensorflow1"])
        self.assertTrue(os.path.exists(os.path.join(work, "keep", "a.py")))
        self.assertTrue(os.path.exists(os.path.join(work, "top.py")))
        self.assertFalse(os.path.exists(os.path.join(work, "tensorflow1")))
        self.assertFalse(os.path.exists(os.path.join(work, ".git")))

    def test_an_ignored_file_is_refused(self):
        with open(os.path.join(self.source, ".gitignore"), "w") as f:
            f.write("build/\n")
        git = ["git", "-C", self.source, "-c", "user.name=t", "-c", "user.email=t@t"]
        subprocess.run(git + ["add", ".gitignore"], check=True)
        subprocess.run(git + ["commit", "-q", "-m", "ignore"], check=True)
        head = subprocess.run(
            git + ["rev-parse", "HEAD"], check=True, capture_output=True, text=True
        )
        os.makedirs(os.path.join(self.source, "build"))
        with open(os.path.join(self.source, "build", "gen.py"), "w") as f:
            f.write("y = 2\n")
        with self.assertRaises(RuntimeError):
            run.prepare(
                {"sha": head.stdout.strip()},
                self.source,
                os.path.join(self.directory.name, "w"),
            )

    def files(self, work):
        return sorted(
            os.path.relpath(os.path.join(d, f), work)
            for d, _, fs in os.walk(work)
            for f in fs
        )

    def test_a_sparse_trim_copies_only_the_matching_files_of_the_commit(self):
        # A file missing from the working tree is still copied: the trim reads the commit, not the
        # checkout, so how the local checkout happens to be sparse does not matter.
        os.remove(os.path.join(self.source, "keep", "a.py"))
        subprocess.run(
            ["git", "-C", self.source, "update-index", "--skip-worktree", "keep/a.py"],
            check=True,
        )
        work = os.path.join(self.directory.name, "work")
        run.prepare(self.subject, self.source, work, sparse=["/keep/*.py"])
        self.assertEqual(self.files(work), ["keep/a.py"])

    def test_sparse_and_exclude_together_are_refused(self):
        with self.assertRaises(RuntimeError):
            run.prepare(
                {**self.subject, "path": "p"},
                self.source,
                os.path.join(self.directory.name, "w"),
                exclude=["tensorflow1"],
                sparse=["/top.py"],
            )

    def test_a_sparse_trim_matching_nothing_is_refused(self):
        with self.assertRaises(RuntimeError):
            run.prepare(
                self.subject,
                self.source,
                os.path.join(self.directory.name, "w"),
                sparse=["/missing.py"],
            )

    def test_a_different_head_is_refused(self):
        with self.assertRaises(RuntimeError):
            run.prepare(
                {"sha": "0" * 40}, self.source, os.path.join(self.directory.name, "w")
            )


if __name__ == "__main__":
    unittest.main()
