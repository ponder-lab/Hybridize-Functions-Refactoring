# Strip-and-re-infer

How closely does the inferred `input_signature` reproduce the one a developer wrote by hand? For each fix commit in a manifest, this harness:

1. strips only the `input_signature` argument from the fixed functions' `tf.function` decorators, keeping every other argument;
1. re-infers the signatures annotation-free, with the evaluator in its standard configuration and inference on;
1. scores the removed spec against the re-inferred one per function, per parameter and per axis.

Stripping is required, not cosmetic. Ariadne reads a developer's signature as type information, so re-inferring with it in place would score the spec against itself.

## Running

```bash
python3.10 run.py --manifest SUBJECTS/manifest.json --out OUT \
	--eclipse <product>/hybridize-evaluator --runner ../run-headless-evaluator.sh \
	--python /usr/local/bin/python3.10 --consumer <commit the product was built from> \
	[--subjects-dir <dir>] [--ground-truth <dir>] [--config config.json] [--only <manifest path> ...] [--strip-only]
```

The manifest lists each subject's repository, fix SHA, checkout path, true import roots, kind (`signature` or `relax_shapes`) and fixed functions (file, line, qualified name). A non-literal removed spec, such as `dataset.element_spec`, is read from `<path>/element_spec.json` under `--ground-truth` (by default `_ground-truth` beside the manifest), together with how it was obtained. Its entries are keyed by qualified name. An entry may also name its `file` and `definition_ordinal`, and it must name enough of them to pick out one function when the subject strips more than one by that name; otherwise none of them gets it. A subject's trim is part of its manifest entry. `sparse` lists git sparse-checkout patterns, which git itself applies in non-cone mode: a throwaway clone sharing the checkout's objects checks out the manifest's commit with them. A pattern means exactly what it means to git, a directory match included, and how the local checkout happens to be sparse doesn't matter. A partial clone that lacks a selected blob fails the copy rather than fetching it. The tree's own `.gitattributes` and global git configuration apply as in any checkout, but the local checkout's `.git/config` and `.git/info/attributes` do not. `caller_scope_complete` says whether that trim is complete for caller search. `exclude` leaves named top-level paths out of the analyzed copy, for a repository that holds a second, separate program. `trim_note` gives the reason. The trim is recorded with the run. `--subjects-dir` says where the subjects are checked out (by default, beside the manifest), and `--ground-truth` where the `element_spec.json` files live. `--config` holds only what is local to one machine: a `source` checkout per subject, in place of the one under `--subjects-dir`.

The harness never writes to a checkout. It refuses one whose HEAD is not the manifest's SHA, or whose tree is dirty. It then copies the checkout without `.git`, or for a sparse trim the throwaway clone's checkout, and strips the copy.

## Identity

The strip step names a function by relative path, lexical qualified name and definition ordinal. The qualified name is the full lexical chain (`train.training_step`, `C.m.Model.update`), and the ordinal counts `def` statements per qualified name by the line of the `def`, the rule `Function.getDefinitionOrdinal` documents. A line is never the key across trees, because removing a multi-line signature moves every line below it.

The evaluator does not always spell a qualified name that way: a function defined inside a block within a function or class loses its enclosing names (#992). So the join finds the evaluator's row by relative path and `def` line in the stripped tree the evaluator analyzed, where the line is exact, and accepts it only if the bare names agree. The evaluator's own name and ordinal are carried beside the lexical ones.

## The Edit

The keyword is cut from the source by its exact span, along with the comma that separated it, so formatting and comments survive. The edit is then verified on the AST: the original module with that one keyword deleted must equal the edited module, node for node. A decorator whose only argument was the signature becomes `@tf.function()`.

## The Order

`spec.py` restates `InputSignature.relate` (`AGREEMENT`, `SUPPLIED_TIGHTER`, `SUPPLIED_BROADER`, `INCOMPARABLE`) because the harness runs outside the OSGi runtime. It adds the same order restricted to the dtype axis and to the shape axis, so a dtype disagreement is reported apart from a shape one. `relation_cases.txt` is read both by `test_spec.py` and by the Java `InputSignatureTest`. A change to either order that the other does not share fails a build.

## Outcomes

Each function receives exactly one outcome:

| Outcome | Meaning | Scored |
| ------- | ------- | ----------- |
| `scored` | An inferred signature exists and is related to the removed one. | Yes |
| `not-reproduced:mapping` | The removed spec holds a dict, a form the inference cannot express. | Yes |
| `not-reproduced:<reason>` | The function is in scope, but the tool produced no spec; `<reason>` is the evaluator's absence reason, with the failed preconditions beside it. | Yes |
| `no-call-site:in-tree` | No evidence reached the function, and no call to it exists in the checkout. | Left to the consumer |
| `no-call-site:trimmed` | The same, but in a sparse checkout whose caller search is not known to be complete. | Left to the consumer |
| `evaluation-failed:<cause>` | The evaluator did not complete for the subject, for example `OutOfMemoryError`; the function is in scope, but the tool produced nothing. | Left to the consumer |
| `relax-shapes` | A relaxation commit whose function the analysis reached, scored on its axes. | Separately |
| `not-reached:call-graph` | A relaxation commit whose function has in-tree callers the call graph did not connect. | Yes |
| `excluded:<reason>` | Nothing can be scored: an unevaluable spec, a positional signature, or a function the tool never considered. | No |

Reachability takes precedence over spec form: a function with no call site scores `no-call-site`, even if its removed spec holds a dict, since the tool had no evidence to infer from. The form is kept in the `also applies` column. A function counts as having no call site only when all three signals agree: no inferred type reached any of its parameters, no resolved call in `calls.csv` names it, and a text search of the checkout finds no call of its name. All three are columns, so the classification can be audited.

## Output

- `reinfer_functions.csv`, one row per function. It holds the outcome, the removed and re-inferred specs, how the removed spec was obtained, the three relations, the absence reasons and failed preconditions, and the call-site evidence.
- `reinfer_parameters.csv`, one row per parameter, aligned by position with the non-`self` parameters as `input_signature` is. It holds the removed spec's leaves (dtype, rank and axes, with a path into a nested spec), the re-inferred spec and the three relations.
- `reinfer_axes.csv`, one row per parameter axis. It holds the set of Constant extents and the set of dimension classes seen across the parameter's inferred types before consensus, the removed dimension, and a verdict. The verdict is `required` when the axis takes more than one extent or any member is Dynamic, `undetermined` when any member is of another non-Constant class, and `unnecessary` for a single fixed extent.

Each subject directory also holds `strip.json` (what was removed, and from where) and `taken-under.json`. The latter records the subject SHA and roots, the evaluator's bundles and consumer commit, the harness commit, the configuration and the trim.

## What It Does Not Cover

- **Extents per call site.** The extent sets are per distinct inferred type, not per call site. Two call sites passing the same type contribute one row, which is right for a set but gives no call-site counts.
- **Mixed-dtype sequences.** A sequence parameter whose elements have different dtypes cannot be paired back to positions from `tensor_specs.csv`. For those, the dtype relation is `UNDETERMINED`, not guessed.
- **Text search.** The text search is syntactic, and over-approximates. It separates "no caller exists" from "callers exist that the analysis did not connect". It is not a call graph.
- **Dtype aliases.** A removed spec's dtype is read as the last name of its expression (`tf.float32` gives `float32`). Aliases such as `tf.double`, `tf.half` or a builtin `float` therefore relate as a dtype disagreement with the evaluator's canonical names. `InputSignature`'s own parser has the same gap.
- **Launcher exit code.** The product launcher exits nonzero even after a successful evaluation, so the harness takes the evaluator's own completion line as its witness, not the exit code.
