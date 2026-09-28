"""Run the pytest suite through colcon's unittest test runner."""

import os
import subprocess
import sys
import unittest
from pathlib import Path


class PytestSuite(unittest.TestCase):
    def test_pytest_suite(self):
        root = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
        env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
        result = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"],
                                cwd=root, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
