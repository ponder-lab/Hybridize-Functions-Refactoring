#!/usr/bin/env python3
"""Diff a headless-evaluator run's evidence populations against the recorded baselines.

The wala/ML#765 regression sat undetected for thirteen releases because nothing compared
the consumer-visible evidence populations across releases (wala/ML#767). This script
institutionalizes the comparison the wala/ML#689 re-measure did by hand: it groups each
evidence CSV's rows by subject, compares the run's rows against the per-subject baseline
under `baselines/`, and classifies additions and losses. Additions are reviewed and folded
into the baseline (`--update`); losses block (nonzero exit) until dispositioned with an
issue link.

Usage:
  ./diff-evidence-baselines.py --run /path/to/run/output/dir
  ./diff-evidence-baselines.py --run /path/to/run/output/dir --update

The run directory is wherever the headless evaluator wrote its result CSVs (its working
directory). Only subjects present in the run are compared, so a single-subject smoke run
(e.g. PROJECTS=NLPGNN) diffs that subject alone.
"""

import argparse
import csv
import sys
from pathlib import Path

# The evidence populations under baseline: per-function statuses and per-parameter rows.
# Volatile outputs (timings, results.csv) are deliberately not baselined.
EVIDENCE_FILES = ["statuses.csv", "parameters.csv"]

SUBJECT_COLUMN = "subject"


def read_rows_by_subject(path):
    """Read an evidence CSV and group its data rows by subject.

    :param path: The CSV file path.
    :return: A pair of the header row and a dict from subject to the set of that
        subject's rows, each row rendered back to a canonical comma-joined string.
    """
    by_subject = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return None, by_subject
        try:
            subject_index = header.index(SUBJECT_COLUMN)
        except ValueError:
            subject_index = None
        for row in reader:
            if not row:
                continue
            subject = row[subject_index] if subject_index is not None else ""
            by_subject.setdefault(subject, set()).add(",".join(row))
    return header, by_subject


def baseline_path(baselines_dir, subject, filename):
    """Compute the baseline file path for a subject's evidence file.

    :param baselines_dir: The baselines root directory.
    :param subject: The subject name.
    :param filename: The evidence file name.
    :return: The baseline file path.
    """
    return baselines_dir / subject / filename


def read_baseline(path):
    """Read a normalized baseline file into its header and row set.

    :param path: The baseline file path.
    :return: A pair of the header row and the set of data-row strings, or (None, None)
        when no baseline is recorded.
    """
    if not path.exists():
        return None, None
    with open(path, newline="", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]
    if not lines:
        return None, None
    return lines[0], set(line for line in lines[1:] if line)


def write_baseline(path, header, rows):
    """Write a subject's normalized (sorted, unique) baseline file.

    :param path: The baseline file path.
    :param header: The CSV header row (a list of column names).
    :param rows: The set of data-row strings.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in sorted(rows):
            f.write(row + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        required=True,
        type=Path,
        help="Directory holding the run's result CSVs (the evaluator's working directory).",
    )
    parser.add_argument(
        "--baselines",
        type=Path,
        default=Path(__file__).resolve().parent / "baselines",
        help="Baselines root directory (default: baselines/ beside this script).",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Fold the run's rows into the baselines instead of just reporting.",
    )
    args = parser.parse_args()

    losses = 0
    additions = 0
    for filename in EVIDENCE_FILES:
        run_file = args.run / filename
        if not run_file.exists():
            print(f"{filename}: not present in run directory; skipped.")
            continue
        header, by_subject = read_rows_by_subject(run_file)
        if header is None:
            print(f"{filename}: empty; skipped.")
            continue
        for subject in sorted(by_subject):
            rows = by_subject[subject]
            base_file = baseline_path(args.baselines, subject or "GLOBAL", filename)
            base_header, base_rows = read_baseline(base_file)
            if base_rows is None:
                print(
                    f"{filename} [{subject}]: no baseline recorded"
                    f" ({len(rows)} row(s) in run)."
                )
                if args.update:
                    write_baseline(base_file, header, rows)
                    print(f"{filename} [{subject}]: baseline seeded.")
                else:
                    additions += len(rows)
                continue
            if base_header != ",".join(header):
                print(
                    f"{filename} [{subject}]: header changed;"
                    " re-seed the baseline deliberately."
                )
                losses += 1
                continue
            added = rows - base_rows
            lost = base_rows - rows
            for row in sorted(added):
                print(f"{filename} [{subject}] ADDED: {row}")
            for row in sorted(lost):
                print(f"{filename} [{subject}] LOST: {row}")
            additions += len(added)
            losses += len(lost)
            if args.update and (added or lost):
                write_baseline(base_file, header, rows)
                print(f"{filename} [{subject}]: baseline updated.")
            if not added and not lost:
                print(
                    f"{filename} [{subject}]: identical to baseline ({len(rows)} row(s))."
                )

    print(f"Total: {additions} addition(s), {losses} loss(es).")
    if losses:
        print(
            "Losses block until dispositioned: cite the disposition issue in the"
            " baseline-update pull request (wala/ML#767)."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
