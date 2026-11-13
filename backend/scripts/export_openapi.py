"""Export the FastAPI OpenAPI document for frontend contract review."""
import json
from pathlib import Path

from fireguard.main import app

output = Path(__file__).resolve().parents[2] / "docs" / "openapi.json"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(app.openapi(), indent=2), encoding="utf-8")
print(f"Wrote {output}")
