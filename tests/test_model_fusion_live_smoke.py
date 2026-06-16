import os
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_SCRIPT = REPO_ROOT / "scripts" / "model_fusion_live_smoke.py"
RUN_FLAG = "MLX_LM_RUN_LIVE_SMOKE"


class TestModelFusionLiveSmokeGate(unittest.TestCase):
    def test_smoke_gate_skips_by_default(self):
        env = os.environ.copy()
        env.pop(RUN_FLAG, None)

        result = run_smoke(env)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("SKIP:", result.stdout)

    @unittest.skipUnless(
        os.environ.get(RUN_FLAG) == "1",
        f"set {RUN_FLAG}=1 to run the Apple Silicon/MLX live smoke",
    )
    def test_smoke_gate_runs_when_enabled(self):
        result = run_smoke(os.environ.copy(), timeout=240)

        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )


def run_smoke(env, timeout=30):
    env["PYTHONPATH"] = str(REPO_ROOT)
    return subprocess.run(
        [sys.executable, str(SMOKE_SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


if __name__ == "__main__":
    unittest.main()
