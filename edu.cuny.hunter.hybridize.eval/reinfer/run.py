"""Strip-and-re-infer: strip each fix commit's developer signature, re-infer it, and score the two.

For every subject in the manifest this copies the checkout (never writing to it), strips only the
``input_signature`` argument of the listed functions, imports the copy headlessly with the manifest's
source roots, runs the evaluator annotation-free in the standard configuration with inference on, and
joins the output into three CSVs: one row per function, per parameter, and per parameter axis.

Usage::

    python3 run.py --manifest SUBJECTS/manifest.json --out OUT \\
        --eclipse <hybridize-evaluator launcher> --runner <run-headless-evaluator.sh> \\
        --python <python3.10 with TensorFlow> [--config config.json] [--only PATH ...]

``--config`` supplies per-subject trims for sparse checkouts: a source directory to copy instead of
the manifest's, whether the caller search is complete by lexical scope, and a note for the output.
"""

import argparse
import csv
import datetime
import json
import os
import shutil
import subprocess
import sys

import join
import strip

# The reference configuration (wala/ML#765), with inference on and no type-annotation sidecar.
EVALUATOR_ENVIRONMENT = {
    "PERFORM_ANALYSIS": "true",
    "PERFORM_CHANGE": "false",
    "TEST_ENTRYPOINTS": "true",
    "FOLLOW_TYPE_HINTS": "true",
    "SPECULATIVE": "true",
    "INFER_INPUT_SIGNATURES": "true",
    "CHECK_SIDE_EFFECTS": "false",
    "CHECK_RECURSION": "false",
    "CHECK_TENSOR_COMPUTATION": "false",
    "CHECK_EAGER_ONLY_CALLS": "false",
    "CHECK_NUMPY_CALLS": "false",
    "PROCESS_IN_PARALLEL": "false",
    "OUTPUT_CALLS": "true",
    "MAX_HEAP": "30480m",
}


SUCCESS_LINE = "Evaluation completed: all 1 project(s) succeeded."


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def git(directory, *arguments):
    result = subprocess.run(
        ["git", "-C", directory, *arguments], capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else None


def repository_of(directory):
    """The checkout's git directory: the path itself, or (saArbabi) the one repository beneath it."""
    if git(directory, "rev-parse", "--show-toplevel") == os.path.realpath(directory):
        return directory
    children = [
        os.path.join(directory, c)
        for c in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, c, ".git"))
    ]
    return children[0] if len(children) == 1 else directory


def prepare(subject, source, work):
    """Copy ``source`` to ``work`` without ``.git``, after checking HEAD is the manifest's SHA."""
    repository = repository_of(source)
    head = git(repository, "rev-parse", "HEAD")
    if head != subject["sha"]:
        raise RuntimeError(
            f"{source}: HEAD is {head}, not the manifest's {subject['sha']}"
        )
    if git(repository, "status", "--porcelain"):
        raise RuntimeError(f"{source}: the working tree is not clean")
    if os.path.exists(work):
        raise RuntimeError(f"{work} exists; use a fresh --out")
    shutil.copytree(source, work, ignore=shutil.ignore_patterns(".git"), symlinks=True)


def strip_subject(subject, work):
    records = []
    by_file = {}
    for function in subject["functions"]:
        by_file.setdefault(function["file"], []).append(function)
    for file, functions in sorted(by_file.items()):
        for record in strip.strip_file(os.path.join(work, file), functions):
            record["file"] = file
            records.append(record)
    return records


def evaluate(subject, work, out, arguments):
    project = os.path.basename(work)
    entry = work + "=" + ":".join(subject["roots"])
    run_dir = os.path.join(out, "run")
    workspace = os.path.join(out, "workspace")
    os.makedirs(workspace)
    environment = dict(os.environ)
    environment.update(EVALUATOR_ENVIRONMENT)
    environment.update(
        {
            "ECLIPSE": arguments.eclipse,
            "WORKSPACE": workspace,
            "OUTDIR": run_dir,
            "IMPORT_PROJECTS": entry,
            "PROJECTS": project,
            "PYTHON_INTERPRETER": arguments.python,
            "JAVA_TOOL_OPTIONS": "-ea:edu.cuny.hunter.hybridize...",
        }
    )
    log_path = os.path.join(out, "run.log")
    with open(log_path, "w") as log:
        code = subprocess.run(
            ["bash", arguments.runner],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        ).returncode
    # The product launcher exits nonzero even after a successful evaluation, so the exit code is not a
    # witness. The evaluator's own completion line is.
    with open(log_path, errors="replace") as log:
        succeeded = SUCCESS_LINE in log.read()
    return run_dir, code, succeeded


def taken_under(arguments, subject, source, trim):
    plugins = os.path.join(os.path.dirname(arguments.eclipse), "plugins")
    jars = (
        sorted(
            p
            for p in os.listdir(plugins)
            if "cast.python.ml_" in p or "hybridize.core_" in p
        )
        if os.path.isdir(plugins)
        else []
    )
    harness = os.path.dirname(os.path.abspath(__file__))
    return {
        "taken": datetime.datetime.now().isoformat(timespec="seconds"),
        "subject": {
            "repo": subject["repo"],
            "sha": subject["sha"],
            "roots": subject["roots"],
            "source": source,
        },
        "evaluator": {
            "launcher": arguments.eclipse,
            "bundles": jars,
            "consumer": arguments.consumer,
        },
        "harness": {
            "commit": git(harness, "rev-parse", "HEAD"),
            "dirty": bool(git(harness, "status", "--porcelain", ".")),
        },
        "configuration": {**EVALUATOR_ENVIRONMENT, "annotations": "none"},
        "trim": trim,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--eclipse", required=True)
    parser.add_argument("--runner", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument(
        "--consumer",
        required=True,
        help="the Hybridize commit the evaluator product was built from",
    )
    parser.add_argument("--config", help="per-subject trims, keyed by manifest path")
    parser.add_argument("--only", nargs="*", help="manifest paths to run, default all")
    parser.add_argument(
        "--strip-only",
        action="store_true",
        help="strip and record, without running the evaluator",
    )
    arguments = parser.parse_args(argv)

    with open(arguments.manifest) as f:
        manifest = json.load(f)
    config = {}
    if arguments.config:
        with open(arguments.config) as f:
            config = json.load(f)
    subjects_dir = os.path.dirname(os.path.abspath(arguments.manifest))
    os.makedirs(arguments.out, exist_ok=True)
    combined = {"functions": [], "parameters": [], "axes": []}
    status = 0

    for subject in manifest["subjects"]:
        if arguments.only and subject["path"] not in arguments.only:
            continue
        trim = config.get(subject["path"], {})
        source = os.path.expanduser(
            trim.get("source", os.path.join(subjects_dir, subject["path"]))
        )
        out = os.path.join(arguments.out, subject["path"])
        work = os.path.join(out, subject["path"])
        os.makedirs(out)
        prepare(subject, source, work)
        records = strip_subject(subject, work)
        with open(os.path.join(out, "strip.json"), "w") as f:
            json.dump(records, f, indent=1)
        with open(os.path.join(out, "taken-under.json"), "w") as f:
            json.dump(taken_under(arguments, subject, source, trim), f, indent=1)
        if arguments.strip_only:
            print(
                f"{subject['path']}: stripped {sum(1 for r in records if r['signature_form'] == 'keyword')} of {len(records)}"
            )
            continue

        run_dir, code, succeeded = evaluate(subject, work, out, arguments)
        if not succeeded:
            print(
                f"{subject['path']}: EVALUATION DID NOT COMPLETE (exit {code}); see {out}/run.log"
            )
            status = 1
            continue
        truth_path = os.path.join(
            subjects_dir, "_ground-truth", subject["path"], "element_spec.json"
        )
        truth = None
        if os.path.exists(truth_path):
            with open(truth_path) as f:
                truth = json.load(f)
        rows = join.join_subject(
            subject, records, run_dir, work, ground_truth=truth, trim=trim
        )
        for name, part in zip(("functions", "parameters", "axes"), rows):
            write_csv(os.path.join(out, f"reinfer_{name}.csv"), part)
            combined[name].extend(part)
        outcomes = ", ".join(f"{r['function']}={r['outcome']}" for r in rows[0])
        print(f"{subject['path']}: evaluated (launcher exit {code}); {outcomes}")

    for name, rows in combined.items():
        write_csv(os.path.join(arguments.out, f"reinfer_{name}.csv"), rows)
    return status


if __name__ == "__main__":
    sys.exit(main())
