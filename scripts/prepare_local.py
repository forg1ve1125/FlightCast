"""Prepare local paths and metadata DB without training or changing model weights.

Default: inspect only. --apply: copy the verified source CSV, back up and rebase
the active metadata's model_path values, then build flight_meta.db if absent.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "flight_delay_model"
MODELS = CODE / "models"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def active_metadata():
    baseline = read_json(MODELS / "latest_run.json")["run_id"]
    buckets = read_json(MODELS / "bucket_latest.json")["run_id"]
    paths = list((MODELS / baseline).glob("*__metadata.json"))
    paths += list((MODELS / buckets).glob("*catboost__metadata.json"))
    registry = read_json(MODELS / "wx_serving.json")
    for slot in ("cls", "bucket", "reg", "cond_reg"):
        run, filename = registry.get(slot + "_run"), registry.get(slot + "_file")
        if run:
            if not filename:
                raise ValueError(f"Missing {slot}_file in model registry")
            paths.append(MODELS / run / (Path(filename).stem + "__metadata.json"))
    calibration = MODELS / registry["bucket_run"] / "isotonic_full2022.json"
    if not calibration.is_file():
        raise FileNotFoundError(calibration)
    if not paths:
        raise ValueError("No active model metadata found")
    return sorted(set(paths))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Apply local setup")
    parser.add_argument("--source-csv", type=Path, help="Original flights_sample_3m.csv")
    args = parser.parse_args()
    active = []
    for path in active_metadata():
        meta = read_json(path)
        binary = path.parent / meta["model_file"]
        if not binary.is_file():
            raise FileNotFoundError(binary)
        active.append((path, meta, binary))
    hashes = {str(p.relative_to(ROOT)): digest(p) for _, _, p in active}
    print(f"Active model files verified: {len(active)}; calibration JSON present.")
    stale = sum(not Path(m["model_path"]).is_file() for _, m, _ in active)
    print(f"Metadata paths needing local setup: {stale}")

    destination = ROOT / "Dataset" / "flights_sample_3m.csv"
    source = destination if destination.is_file() else args.source_csv
    if source is None:
        print("Original CSV missing. Pass --source-csv to use an existing copy.")
        if args.apply:
            raise FileNotFoundError(destination)
    else:
        source = source.resolve(strict=True)
        expected = read_json(ROOT / "Dataset" / "data_manifest.json")["original_md5"]
        actual = digest(source, "md5")
        if actual.lower() != expected.lower():
            raise ValueError(f"CSV MD5 mismatch: {actual}; expected {expected}")
        print(f"Original CSV MD5 verified: {actual}")

    if not args.apply:
        print("Inspection only: no files changed, no dependencies installed.")
        return

    local = ROOT / ".local"
    local.mkdir(exist_ok=True)
    checksum_file = local / "model-sha256.json"
    if checksum_file.exists() and read_json(checksum_file) != hashes:
        raise ValueError("Model files differ from the previous setup; inspect before continuing")
    checksum_file.write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    if not destination.exists():
        shutil.copy2(source, destination)
        if digest(destination, "md5") != expected.lower():
            raise ValueError("Copied CSV failed checksum verification")
        print(f"Copied original CSV to {destination}")

    for path, meta, binary in active:
        replacement = str(binary.resolve())
        if meta["model_path"] == replacement:
            continue
        backup = local / "metadata-backup" / path.relative_to(ROOT)
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            shutil.copy2(path, backup)
        # Replace only this JSON value; retain every other metadata field.
        text = path.read_text(encoding="utf-8")
        updated, count = re.subn(
            r'("model_path"\s*:\s*)"(?:[^"\\]|\\.)*"',
            lambda match: match[1] + json.dumps(replacement, ensure_ascii=False),
            text,
        )
        if count != 1:
            raise ValueError(f"Expected exactly one model_path: {path}")
        expected_meta = dict(meta, model_path=replacement)
        if json.loads(updated) != expected_meta:
            raise ValueError(f"Unexpected metadata change: {path}")
        path.write_text(updated, encoding="utf-8")
    if any(digest(ROOT / relative) != value for relative, value in hashes.items()):
        raise ValueError("Model weights changed unexpectedly")
    print("Local paths prepared. Model weights and registry unchanged.")

    database = CODE / "server" / "flight_meta.db"
    if database.exists():
        print("Existing flight_meta.db kept; no database overwritten.")
    else:
        subprocess.run([sys.executable, str(CODE / "server" / "build_meta_db.py")],
                       cwd=CODE, check=True)
    print("Local data preparation complete. Run dev.ps1 -Target check next.")


if __name__ == "__main__":
    main()
