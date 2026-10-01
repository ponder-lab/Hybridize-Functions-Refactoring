import os
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


class SparseMatcherTest(unittest.TestCase):
    def matches(self, pattern, path):
        return bool(run.sparse_matcher(pattern).fullmatch(path))

    def test_a_star_stays_within_one_directory(self):
        self.assertTrue(self.matches("/a/*.py", "a/x.py"))
        self.assertFalse(self.matches("/a/*.py", "a/b/x.py"))

    def test_a_double_star_spans_directories(self):
        for path in ("a/x.py", "a/b/x.py", "a/b/c/x.py"):
            with self.subTest(path=path):
                self.assertTrue(self.matches("/a/**/*.py", path))
        self.assertFalse(self.matches("/a/**/*.py", "b/x.py"))

    def test_a_pattern_is_anchored_at_the_root(self):
        self.assertTrue(self.matches("/x.py", "x.py"))
        self.assertFalse(self.matches("/x.py", "a/x.py"))

    def test_patterns_git_reads_differently_are_refused(self):
        for pattern in ("x.py", "!/x.py", "/a/", "/[ab].py"):
            with self.subTest(pattern=pattern):
                with self.assertRaises(RuntimeError):
                    run.sparse_matcher(pattern)


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
