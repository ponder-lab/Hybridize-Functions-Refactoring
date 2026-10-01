"""Strip-and-re-infer: strip each fix commit's developer signature, re-infer it, and score the two.

For every subject in the manifest this copies the checkout (never writing to it), strips only the
``input_signature`` argument of the listed functions, imports the copy headlessly with the manifest's
source roots, runs the evaluator annotation-free in the standard configuration with inference on, and
joins the output into three CSVs: one row per function, per parameter, and per parameter axis.

Usage::

    python3 run.py --manifest SUBJECTS/manifest.json --out OUT \\
        --eclipse <hybridize-evaluator launcher> --runner <run-headless-evaluator.sh> \\
        --python <python3.10 with TensorFlow> [--subjects-dir DIR] [--ground-truth DIR] \\
        [--config config.json] [--only PATH ...]

A subject's trim is part of its manifest entry: ``sparse`` (the paths of the fix commit to analyze,
for a repository too large to analyze whole), ``caller_scope_complete`` (whether that trim is complete
for caller search), ``exclude`` (top-level paths to leave out) and a ``trim_note``. ``--config`` holds only
what is local to one machine, keyed by manifest path: a ``source`` checkout to copy in place of the
one under ``--subjects-dir``.
"""

import argparse
import csv
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

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


def write_csv(path, rows, columns):
    """Write ``rows`` under the header ``columns``; with no rows, the file holds the header alone, so an
    empty result is told apart from a run that wrote nothing."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
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


def prepare(subject, source, work, exclude=(), sparse=()):
    """Copy ``source`` to ``work`` without ``.git``, after checking HEAD is the manifest's SHA.

    ``exclude`` names top-level paths of the checkout to leave out of the analyzed tree, for a
    repository that holds a second, separate program (VaDER's ``tensorflow1`` beside
    ``tensorflow2``). ``sparse``, when given, is a list of git sparse-checkout patterns, and only
    the files of the commit that git's own non-cone sparse checkout selects are copied (see
    ``copy_sparse``). Either trim is recorded with the run.
    """
    repository = repository_of(source)
    head = git(repository, "rev-parse", "HEAD")
    if head != subject["sha"]:
        raise RuntimeError(
            f"{source}: HEAD is {head}, not the manifest's {subject['sha']}"
        )
    # --ignored too: ignored files (build output, a virtualenv, generated sources) are copied into the
    # analyzed tree like any other, so a checkout carrying them is not the commit it claims to be.
    if git(repository, "--no-optional-locks", "status", "--porcelain", "--ignored"):
        raise RuntimeError(
            f"{source}: the working tree is not clean, or carries ignored files"
        )
    if os.path.exists(work):
        raise RuntimeError(f"{work} exists; use a fresh --out")
    if sparse:
        if exclude:
            raise RuntimeError(f"{subject['path']}: both sparse and exclude are given")
        copy_sparse(repository, source, work, sparse)
        return
    excluded = {os.path.normpath(os.path.join(source, e)) for e in exclude}

    def ignore(directory, names):
        return {
            n
            for n in names
            if n == ".git" or os.path.normpath(os.path.join(directory, n)) in excluded
        }

    shutil.copytree(source, work, ignore=ignore, symlinks=True)


def copy_sparse(repository, source, work, patterns):
    """Write the files of HEAD that git's own non-cone sparse checkout of ``patterns`` selects to
    ``work``. ``source`` must be the repository's root, since the patterns are anchored there.

    The patterns are not interpreted here. A throwaway clone that shares the checkout's objects takes
    them through ``git sparse-checkout set --no-cone`` and checks out HEAD, so directory matches,
    ``**`` and attributes mean exactly what they mean in git. Nothing is written to the checkout, and
    nothing is fetched: a blob a partial clone lacks fails the copy rather than being downloaded.
    """
    toplevel = git(source, "rev-parse", "--show-toplevel")
    if toplevel is None or os.path.realpath(toplevel) != os.path.realpath(source):
        raise RuntimeError(
            f"{source}: a sparse trim needs the repository's root, not {toplevel}"
        )
    head = git(source, "rev-parse", "HEAD")
    environment = {**os.environ, "GIT_NO_LAZY_FETCH": "1", "GIT_TERMINAL_PROMPT": "0"}

    def run_git(*arguments, **options):
        result = subprocess.run(
            ["git", *arguments], capture_output=True, env=environment, **options
        )
        if result.returncode != 0:
            message = result.stderr.decode(errors="replace").strip()
            raise RuntimeError(
                f"{source}: git {' '.join(arguments[:3])} failed: {message}"
            )

    with tempfile.TemporaryDirectory() as scratch:
        clone = os.path.join(scratch, "clone")
        run_git("clone", "-q", "--no-checkout", "--shared", source, clone)
        run_git(
            "-C",
            clone,
            "sparse-checkout",
            "set",
            "--no-cone",
            "--stdin",
            input="".join(p + "\n" for p in patterns).encode(),
        )
        run_git("-C", clone, "checkout", "-q", "--detach", head)
        # git reports a selected blob a partial clone lacks but still exits 0, leaving the file out of
        # the checkout; it shows as deleted, so a checkout that is not clean is refused.
        status = subprocess.run(
            ["git", "-C", clone, "status", "--porcelain"],
            capture_output=True,
            env=environment,
        )
        if status.returncode != 0 or status.stdout.strip():
            missing = (
                status.stdout.decode(errors="replace").strip()
                or status.stderr.decode(errors="replace").strip()
            )
            raise RuntimeError(
                f"{source}: git's sparse checkout of HEAD is incomplete, likely blobs a partial "
                f"clone lacks: {missing}"
            )
        selected = any(
            files for d, _, files in os.walk(clone) if ".git" not in d.split(os.sep)
        )
        if not selected:
            raise RuntimeError(
                f"{source}: git's sparse checkout of {patterns} selects no file of HEAD"
            )
        shutil.copytree(
            clone, work, ignore=shutil.ignore_patterns(".git"), symlinks=True
        )


# A manifest entry's trim fields, and the name each has in the recorded trim. The note is
# ``trim_note`` in the manifest, beside the entry's own ``notes``.
TRIM_FIELDS = {
    "sparse": "sparse",
    "caller_scope_complete": "caller_scope_complete",
    "exclude": "exclude",
    "trim_note": "note",
}


def trim_of(subject, local):
    """The subject's trim from its manifest entry, plus its machine-local ``source``, if any.

    A local entry may only add ``source``; a trim field there would let the analyzed tree differ
    from what the committed manifest says, so one is refused.
    """
    misplaced = sorted(set(local) - {"source"})
    if misplaced:
        raise RuntimeError(
            f"{subject['path']}: {', '.join(misplaced)} belongs in the manifest, not --config"
        )
    trim = {
        name: subject[field] for field, name in TRIM_FIELDS.items() if field in subject
    }
    for field in ("sparse", "exclude"):
        value = trim.get(field)
        if value is not None and not (
            isinstance(value, list)
            and value
            and all(isinstance(v, str) and v for v in value)
        ):
            raise RuntimeError(
                f"{subject['path']}: {field} must be a non-empty list of strings, not {value!r}"
            )
    for field, kind in (("caller_scope_complete", bool), ("note", str)):
        if field in trim and not isinstance(trim[field], kind):
            raise RuntimeError(
                f"{subject['path']}: {field} must be a {kind.__name__}, not {trim[field]!r}"
            )
    trim.update(local)
    return trim


def strip_subject(subject, work):
    records = []
    by_file = {}
    for function in subject["functions"]:
        by_file.setdefault(function["file"], []).append(function)
    for file, functions in sorted(by_file.items()):
        for record in strip.strip_file(os.path.join(work, file), functions, root=work):
            record["file"] = file
            records.append(record)
    return records


def project_name(work):
    """The name the evaluator will give the imported project.

    The headless import names a project from a committed ``.project`` when the checkout carries one, and
    from its directory otherwise; asking for the directory's name when the two differ evaluates nothing.
    """
    try:
        with open(os.path.join(work, ".project")) as f:
            match = re.search(r"<name>([^<]*)</name>", f.read())
        if match:
            return match.group(1)
    except OSError:
        pass
    return os.path.basename(work)


def evaluate(subject, work, out, arguments):
    project = project_name(work)
    entry = work + "=" + ":".join(subject["roots"])
    run_dir = os.path.join(out, "run")
    workspace = os.path.join(out, "workspace")
    os.makedirs(workspace)
    environment = dict(os.environ)
    environment.update(EVALUATOR_ENVIRONMENT)
    environment["MAX_HEAP"] = arguments.max_heap
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
    # The launcher's log misses errors the platform records only in the workspace's own log, such as a
    # StackOverflowError that ends the application, so both are read.
    text = ""
    for path in (log_path, os.path.join(workspace, ".metadata", ".log")):
        try:
            with open(path, errors="replace") as log:
                text += log.read()
        except OSError:
            pass
    return run_dir, code, failure_of(text, code)


def failure_of(log, code):
    """Why an evaluation did not complete, or None if it did.

    The product launcher exits nonzero even after a successful evaluation, so the exit code is not a
    witness. The evaluator's own completion line is.
    """
    if SUCCESS_LINE in log:
        return None
    for error in ("OutOfMemoryError", "StackOverflowError"):
        if "java.lang." + error in log:
            return error
    skipped = re.search(
        r"Evaluation completed: 0 of 1 project\(s\) succeeded.*?\((\w+):", log
    )
    if skipped:
        return skipped.group(1)
    return f"did-not-complete (launcher exit {code})"


def harness_provenance(harness):
    """Which harness produced a run, and whether that code is merged.

    The commit alone does not answer the second question after a squash merge, which gives the same
    code a different commit. The git tree of this directory does: a run whose tree equals the tree
    of this directory on main ran merged code, whatever its commit.
    """
    commit = git(harness, "rev-parse", "HEAD")
    tree = git(harness, "rev-parse", "HEAD:./")
    git(harness, "fetch", "-q", "origin", "main")
    main_tree = git(
        harness, "rev-parse", "origin/main:edu.cuny.hunter.hybridize.eval/reinfer"
    )
    on_main = (
        subprocess.run(
            [
                "git",
                "-C",
                harness,
                "merge-base",
                "--is-ancestor",
                commit,
                "origin/main",
            ],
            capture_output=True,
        ).returncode
        == 0
    )
    return {
        "commit": commit,
        "dirty": bool(git(harness, "status", "--porcelain", ".")),
        "tree": tree,
        "commit on main": on_main,
        "main commit": git(harness, "rev-parse", "origin/main"),
        "tree on main": main_tree,
        "tree equals main": tree is not None and tree == main_tree,
    }


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
        "harness": harness_provenance(harness),
        "configuration": {
            **EVALUATOR_ENVIRONMENT,
            "MAX_HEAP": arguments.max_heap,
            "annotations": "none",
        },
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
    parser.add_argument(
        "--subjects-dir",
        help="where each subject is checked out at its manifest path, default the manifest's directory",
    )
    parser.add_argument(
        "--ground-truth",
        help="where <path>/element_spec.json files live, default _ground-truth beside the manifest",
    )
    parser.add_argument(
        "--config", help="machine-local checkout paths (source), keyed by manifest path"
    )
    parser.add_argument(
        "--max-heap",
        default=EVALUATOR_ENVIRONMENT["MAX_HEAP"],
        help="the evaluator's heap, recorded in taken-under.json",
    )
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
    manifest_dir = os.path.dirname(os.path.abspath(arguments.manifest))
    subjects_dir = os.path.abspath(arguments.subjects_dir or manifest_dir)
    ground_truth_dir = os.path.abspath(
        arguments.ground_truth or os.path.join(manifest_dir, "_ground-truth")
    )
    os.makedirs(arguments.out, exist_ok=True)
    combined = {"functions": [], "parameters": [], "axes": []}
    status = 0

    for subject in manifest["subjects"]:
        if arguments.only and subject["path"] not in arguments.only:
            continue
        trim = trim_of(subject, config.get(subject["path"], {}))
        source = os.path.expanduser(
            trim.get("source", os.path.join(subjects_dir, subject["path"]))
        )
        out = os.path.join(arguments.out, subject["path"])
        work = os.path.join(out, subject["path"])
        os.makedirs(out)
        prepare(
            subject,
            source,
            work,
            exclude=trim.get("exclude", ()),
            sparse=trim.get("sparse", ()),
        )
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

        run_dir, code, failure = evaluate(subject, work, out, arguments)
        if failure:
            # Still joined: a failed evaluation is an outcome for every function it covered, not a
            # reason for those functions to leave the output.
            print(
                f"{subject['path']}: EVALUATION FAILED ({failure}); see {out}/run.log",
                flush=True,
            )
            status = 1
        truth_path = os.path.join(
            ground_truth_dir, subject["path"], "element_spec.json"
        )
        truth = None
        if os.path.exists(truth_path):
            with open(truth_path) as f:
                truth = json.load(f)
        rows = join.join_subject(
            subject,
            records,
            run_dir,
            work,
            ground_truth=truth,
            trim=trim,
            failure=failure,
        )
        for name, part in zip(("functions", "parameters", "axes"), rows):
            write_csv(
                os.path.join(out, f"reinfer_{name}.csv"), part, join.COLUMNS[name]
            )
            combined[name].extend(part)
        outcomes = ", ".join(f"{r['function']}={r['outcome']}" for r in rows[0])
        print(f"{subject['path']}: launcher exit {code}; {outcomes}", flush=True)

    if arguments.strip_only:
        # Nothing was joined, so there is no result to write, not an empty one.
        return status
    for name, rows in combined.items():
        write_csv(
            os.path.join(arguments.out, f"reinfer_{name}.csv"), rows, join.COLUMNS[name]
        )
    return status


if __name__ == "__main__":
    sys.exit(main())
