"""Export the FastAPI OpenAPI document for frontend contract review."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fireguard.main import app

output = Path(__file__).resolve().parents[2] / "docs" / "openapi.json"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(app.openapi(), indent=2), encoding="utf-8")
print(f"Wrote {output}")
