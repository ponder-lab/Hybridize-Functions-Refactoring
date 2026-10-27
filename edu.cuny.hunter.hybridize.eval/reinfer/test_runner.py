import os
import subprocess
import tempfile
import unittest

RUNNER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, "run-headless-evaluator.sh"
)


class RunnerExitCodeTest(unittest.TestCase):
    """The runner exits with the launcher's exit code, which callers read to tell
    a completed run from a stopped one (issue 1121)."""

    def run_with_launcher_exiting(self, code):
        with tempfile.TemporaryDirectory() as directory:
            launcher = os.path.join(directory, "launcher")
            with open(launcher, "w") as f:
                f.write(f"#!/bin/sh\nexit {code}\n")
            os.chmod(launcher, 0o755)
            env = dict(
                os.environ,
                ECLIPSE=launcher,
                WORKSPACE=os.path.join(directory, "workspace"),
                OUTDIR=os.path.join(directory, "out"),
            )
            return subprocess.run(
                ["bash", RUNNER], env=env, capture_output=True
            ).returncode

    def test_a_lost_pydev_answer_exits_7(self):
        self.assertEqual(self.run_with_launcher_exiting(7), 7)

    def test_the_other_codes_pass_through(self):
        for code in (0, 1, 6):
            with self.subTest(code=code):
                self.assertEqual(self.run_with_launcher_exiting(code), code)


if __name__ == "__main__":
    unittest.main()
