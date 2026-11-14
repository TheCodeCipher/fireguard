"""The live routes never load the ML stack: inference stays offline (api_contract.md 12)."""

import subprocess
import sys


def test_live_route_imports_neither_torch_nor_fireguard_ml():
    code = (
        "import sys\n"
        "import fireguard.main, fireguard.api.routes.live\n"
        "bad = [m for m in sys.modules\n"
        "       if m == 'torch' or m.startswith(('torch.', 'fireguard_ml'))]\n"
        "assert not bad, bad\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
