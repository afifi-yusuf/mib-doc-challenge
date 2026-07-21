"""Batch runner for classical MIB solution."""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .classical import predict_pdf
from .constants import FIELDNAMES


def _process_one(pdf_path: str) -> dict:
    rec = predict_pdf(pdf_path)
    return {k: rec[k] for k in FIELDNAMES}


def run_pipeline(input_dir: Path, output_path: Path, workers: int = 4) -> int:
    pdfs = sorted(Path(input_dir).glob("*.pdf"))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    done = 0
    if workers <= 1 or len(pdfs) <= 1:
        for pdf in pdfs:
            rec = _process_one(str(pdf))
            results[rec["case_id"]] = rec
            done += 1
            if done % 25 == 0:
                print(f"progress {done}/{len(pdfs)}", file=sys.stderr, flush=True)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_process_one, str(pdf)): pdf for pdf in pdfs}
            for fut in as_completed(futures):
                try:
                    rec = fut.result()
                    results[rec["case_id"]] = rec
                except Exception as exc:  # noqa: BLE001
                    pdf = futures[fut]
                    print(f"WARN: failed {pdf.name}: {exc}", file=sys.stderr)
                done += 1
                if done % 25 == 0:
                    print(f"progress {done}/{len(pdfs)}", file=sys.stderr, flush=True)

    with output_path.open("w", encoding="utf-8") as f:
        for case_id in sorted(results):
            row = {**results[case_id], "confidence": float(results[case_id]["confidence"])}
            f.write(json.dumps(row, sort_keys=True) + "\n")
    return len(results)


def main(argv: list[str] | None = None) -> None:
    import os

    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2:
        raise SystemExit("usage: solution.py <input_pdf_dir> <output_predictions_path>")
    workers = int(os.environ.get("MIB_WORKERS", "2"))
    n = run_pipeline(Path(argv[0]), Path(argv[1]), workers=workers)
    print(f"wrote {n} predictions to {argv[1]}", file=sys.stderr)


if __name__ == "__main__":
    main()
