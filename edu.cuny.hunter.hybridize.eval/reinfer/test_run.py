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


if __name__ == "__main__":
    unittest.main()
