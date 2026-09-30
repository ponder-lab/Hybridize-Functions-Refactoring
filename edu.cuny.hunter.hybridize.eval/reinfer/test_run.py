import os
import subprocess
import tempfile
import unittest

import run


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

    def test_a_different_head_is_refused(self):
        with self.assertRaises(RuntimeError):
            run.prepare(
                {"sha": "0" * 40}, self.source, os.path.join(self.directory.name, "w")
            )


if __name__ == "__main__":
    unittest.main()
