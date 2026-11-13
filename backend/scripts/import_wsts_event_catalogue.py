"""Import a curated, approved event catalogue CSV into wildfire_events.

Expected columns: event_id, year. Optional columns: name, region, start_date,
end_date, complex_group_id, p1_fold_id, p2_fold_id, manifest_sha256.
Raw rasters, target paths and label data are intentionally not imported.
"""
import argparse
import csv
from datetime import date
from pathlib import Path

from fireguard.core.database import SessionLocal
from fireguard.models import WildfireEvent


def optional_date(value: str | None):
    return date.fromisoformat(value) if value else None


def import_catalogue(path: Path) -> tuple[int, int]:
    inserted = updated = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle, SessionLocal() as db:
        reader = csv.DictReader(handle)
        if not {"event_id", "year"}.issubset(set(reader.fieldnames or [])):
            raise ValueError("Catalogue CSV must contain event_id and year columns")
        for row in reader:
            event_id = (row.get("event_id") or "").strip()
            if not event_id:
                raise ValueError("Empty event_id found in event catalogue")
            year = int(row["year"])
            existing = db.query(WildfireEvent).filter_by(event_key=event_id).one_or_none()
            values = {
                "name": row.get("name") or None,
                "region": row.get("region") or None,
                "year": year,
                "start_date": optional_date(row.get("start_date") or None),
                "end_date": optional_date(row.get("end_date") or None),
                "complex_group_id": row.get("complex_group_id") or None,
                "p1_fold_id": row.get("p1_fold_id") or None,
                "p2_fold_id": row.get("p2_fold_id") or None,
                "manifest_sha256": row.get("manifest_sha256") or None,
                "source_dataset": "WSTS",
            }
            if existing is None:
                db.add(WildfireEvent(event_key=event_id, **values))
                inserted += 1
            else:
                for key, value in values.items():
                    setattr(existing, key, value)
                updated += 1
        db.commit()
    return inserted, updated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    args = parser.parse_args()
    inserted, updated = import_catalogue(args.csv_path)
    print(f"Imported catalogue: inserted={inserted}, updated={updated}")


if __name__ == "__main__":
    main()
