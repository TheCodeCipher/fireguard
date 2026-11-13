"""
WSTS all-TIFF raster grid audit -- no pandas required.

Run from the FireGuard repository:
    python scripts\wsts_raster_grid_audit.py

Requires rasterio in the active Python environment.
Outputs:
    data/manifests/wsts_raster_grid_audit.csv
    data/manifests/wsts_event_grid_consistency.csv
    data/manifests/wsts_event_spatial_metadata.csv
"""
from pathlib import Path
import csv
import json
import math
import rasterio

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(r"G:\My Drive\FireGuard\data\extracted")
EVENT_METADATA = REPO_ROOT / "data" / "manifests" / "wsts_event_metadata.csv"
OUT_DIR = REPO_ROOT / "data" / "manifests"

GRID_FIELDS = (
    "width", "height", "band_count", "crs",
    "resolution_x", "resolution_y",
    "transform", "left", "bottom", "right", "top",
)
AUDIT_FIELDS = (
    "year", "event_id", "relative_file_path", "file_name",
    "inspection_status", "inspection_error",
    *GRID_FIELDS, "dtype_list", "nodata_values",
)

def read_csv(path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])

def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

def same_value(a, b, tol=1e-8):
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=0, abs_tol=tol)
    return str(a) == str(b)

def inspect_tiff(path, event_id, year):
    row = {
        "year": str(year),
        "event_id": str(event_id),
        "relative_file_path": path.relative_to(DATA_ROOT).as_posix(),
        "file_name": path.name,
        "inspection_status": "OK",
        "inspection_error": "",
    }
    try:
        with rasterio.open(path) as ds:
            b, t = ds.bounds, ds.transform
            row.update({
                "width": ds.width,
                "height": ds.height,
                "band_count": ds.count,
                "crs": ds.crs.to_string() if ds.crs else "",
                "resolution_x": float(ds.res[0]),
                "resolution_y": float(ds.res[1]),
                "transform": json.dumps(list(t)[:6]),
                "left": float(b.left),
                "bottom": float(b.bottom),
                "right": float(b.right),
                "top": float(b.top),
                "dtype_list": ",".join(ds.dtypes),
                "nodata_values": ",".join(
                    "None" if v is None else str(v) for v in ds.nodatavals
                ),
            })
    except Exception as exc:
        row["inspection_status"] = "ERROR"
        row["inspection_error"] = f"{type(exc).__name__}: {exc}"
        for field in GRID_FIELDS:
            row[field] = ""
    return row

def main():
    if not DATA_ROOT.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {DATA_ROOT}")
    if not EVENT_METADATA.is_file():
        raise FileNotFoundError(f"Event metadata not found: {EVENT_METADATA}")

    events, event_fields = read_csv(EVENT_METADATA)
    required = {"year", "event_id", "tiff_count"}
    missing = required - set(event_fields)
    if missing:
        raise ValueError(f"Event metadata missing columns: {sorted(missing)}")

    audit = []
    for i, event in enumerate(events, start=1):
        year, event_id = event["year"], event["event_id"]
        event_dir = DATA_ROOT / str(year) / str(event_id)
        if not event_dir.is_dir():
            audit.append({
                "year": year, "event_id": event_id,
                "relative_file_path": "", "file_name": "",
                "inspection_status": "EVENT_DIRECTORY_MISSING",
                "inspection_error": str(event_dir),
            })
        else:
            paths = sorted(
                p for p in event_dir.iterdir()
                if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}
            )
            if not paths:
                audit.append({
                    "year": year, "event_id": event_id,
                    "relative_file_path": "", "file_name": "",
                    "inspection_status": "NO_TIFF_FILES",
                    "inspection_error": "",
                })
            for path in paths:
                audit.append(inspect_tiff(path, event_id, year))
        if i % 25 == 0 or i == len(events):
            print(f"Processed event records: {i}/{len(events)}")

    write_csv(OUT_DIR / "wsts_raster_grid_audit.csv", AUDIT_FIELDS, audit)

    summary = []
    for event in events:
        year, event_id = event["year"], event["event_id"]
        group = [r for r in audit if r["event_id"] == event_id]
        good = [r for r in group if r["inspection_status"] == "OK"]
        result = {
            "year": year,
            "event_id": event_id,
            "manifest_tiff_count": event["tiff_count"],
            "discovered_tiff_count": sum(bool(r["file_name"]) for r in group),
            "inspected_tiff_count": len(good),
            "error_or_missing_records": len(group) - len(good),
        }
        for field in GRID_FIELDS:
            values = [r.get(field, "") for r in good]
            unique = set(values)
            result[field + "_unique_values"] = len(unique)
            result[field + "_consistent"] = (
                bool(good) and all(same_value(v, values[0]) for v in values)
            )
        result["grid_consistent_all_fields"] = (
            bool(good)
            and len(good) == len(group)
            and result["discovered_tiff_count"] == int(event["tiff_count"])
            and all(result[field + "_consistent"] for field in GRID_FIELDS)
        )
        summary.append(result)

    summary_fields = list(summary[0]) if summary else []
    write_csv(OUT_DIR / "wsts_event_grid_consistency.csv", summary_fields, summary)

    extended = []
    for event in events:
        match = next(r for r in summary if r["event_id"] == event["event_id"])
        row = dict(event)
        for key in (
            "discovered_tiff_count", "inspected_tiff_count",
            "error_or_missing_records", "grid_consistent_all_fields",
        ):
            row[key] = match[key]
        row["metadata_scope"] = "first_tiff_plus_all_tiff_grid_audit"
        row["actual_fire_perimeter_status"] = "UNKNOWN"
        row["fire_footprint_proxy_status"] = "NOT_DERIVED"
        extended.append(row)

    extended_fields = list(event_fields)
    for key in (
        "discovered_tiff_count", "inspected_tiff_count",
        "error_or_missing_records", "grid_consistent_all_fields",
        "metadata_scope", "actual_fire_perimeter_status",
        "fire_footprint_proxy_status",
    ):
        if key not in extended_fields:
            extended_fields.append(key)
    write_csv(OUT_DIR / "wsts_event_spatial_metadata.csv", extended_fields, extended)

    success = sum(r["inspection_status"] == "OK" for r in audit)
    failures = len(audit) - success
    consistent = sum(bool(r["grid_consistent_all_fields"]) for r in summary)
    print("\nAUDIT COMPLETE")
    print("Audit records:", len(audit))
    print("TIFFs successfully inspected:", success)
    print("Error/missing records:", failures)
    print("Events with consistent grids:", consistent, "/", len(summary))
    print("Output directory:", OUT_DIR)

if __name__ == "__main__":
    main()
