# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Eclipse PDE refactoring plug-ins (Tycho-built) that automate hybridization of Python TensorFlow functions — i.e., deciding when to add/remove `@tf.function` and applying the transformation. Runs inside Eclipse on top of PyDev and uses WALA/Ariadne for static analysis of Python.

The tool is **not compatible with stock PyDev**: it depends on the ponder-lab fork (`pydev_9_3` branch). See `hybridize.target` for the active target platform (Eclipse 4.39, PyDev 9.3.x, WALA 1.7, Ariadne via `maven.pkg.github.com/ponder-lab/ML`).

## Build / test commands

The reactor is a Tycho multi-module Maven build. Requires **Java 25** (`<maven.compiler.source/target>` and every bundle's `Bundle-RequiredExecutionEnvironment` are `JavaSE-25`) and **Maven 3.9.11+** (Tycho 5.0.2 nominally supports 3.9.9 but trips a `TargetPlatformArtifactResolver` binding error on 3.9.9 — see eclipse-tycho/tycho#5384; the project's `maven-enforcer-plugin` rule pins `[3.9.11,)`). `.mvn/extensions.xml` registers `tycho-build` as a Maven core extension (required by Tycho 4+).

The repo ships **Maven Wrapper** (`./mvnw`) pinned to Maven 3.9.11 so contributors don't need to upgrade their system Maven. **Use `./mvnw` rather than system `mvn`** — system Maven is often 3.9.9 and will fail on the Tycho binding error.

```bash
# Full build, no tests
./mvnw -U -s .travis.settings.xml -Dgithub.username=<user> -Dgithub.password=<token> install -DskipTests=true

# Tests + JaCoCo (matches CI)
./mvnw -U -s .travis.settings.xml -Dgithub.username=<user> -Dgithub.password=<token> verify -Pjacoco

# Format check / apply (Spotless: Eclipse formatter for Java, also XML/MD/POM)
./mvnw spotless:check
./mvnw spotless:apply

# Checkstyle (bound to validate phase)
./mvnw validate
```

Ariadne (`com.ibm.wala.cast.python.ml`) is fetched from GitHub Packages, which requires a `~/.m2/settings.xml` (or `-s .travis.settings.xml` with `-Dgithub.username/-Dgithub.password`) carrying a token with `read:packages`. See `CONTRIBUTING.md` and the GitHub Packages docs.

Python formatting in CI uses `black --fast --check --extend-exclude \/out .` (`requirements.txt`).

### Running tests

Tests live in `edu.cuny.hunter.hybridize.tests/` under a non-standard Eclipse-PDE source folder named **`test cases/`** (note the space). The single test class is `HybridizeFunctionRefactoringTest` (~8000 lines, hundreds of `testXxx` methods).

CI requires our PyDev fork to be cloned at `$HOME/git/Pydev` (branch `pydev_9_3`) before running tests — `TestDependent` resolves Python lib paths relative to that checkout. Reproduce locally:

```bash
mkdir -p "$HOME/git" && git clone --depth=50 --branch=pydev_9_3 \
	https://github.com/ponder-lab/Pydev.git "$HOME/git/Pydev"
pip3.10 install -r edu.cuny.hunter.hybridize.tests/requirements.txt   # tensorflow==2.9.3
```

Test resources are under `edu.cuny.hunter.hybridize.tests/resources/HybridizeFunction/<testName>/in/A.py` (plus a per-test `requirements.txt`). Each test method `testFoo` looks up the directory `HybridizeFunction/testFoo/`.

Eclipse PDE `.launch` files are checked in for whole-class and individual-test runs (`HybridizeFunctionRefactoringTest*.launch`). They run as a JUnit Plug-in Test, *not* plain JUnit — these tests need an OSGi/PDE runtime; `mvn verify` drives them via tycho-surefire. There are also launches for the evaluator (`edu.cuny.hunter.hybridize.eval/Evaluate Hybridize Functions.launch`).

Useful system properties (all `Boolean.getBoolean`):
- `edu.cuny.hunter.hybridize.tests.runInput` — actually `python3.10`-execute the input `.py` files (and pip-install their requirements) before analysis.
- `edu.cuny.hunter.hybridize.tests.compareOutput` — diff against expected output files.
- `edu.cuny.hunter.hybridize.dumpCallGraph` — dump WALA call graphs (verbose; off by default in tests, on in some launches).
- Evaluator-only knobs (`edu.cuny.hunter.hybridize.eval.*`): `alwaysCheckPythonSideEffects`, `alwaysCheckRecursion`, `processFunctionsInParallel`, `useTestEntrypoints`, `alwaysFollowTypeHints`, `useSpeculativeAnalysis`, `inferInputSignatures`, `performAnalysis`, `performChange`, `outputCalls`.
- The targeted k-CFA depth the evaluator forwards to the engine is set per-project via a `targetedCfaDepth` entry in an `eval.properties` file (searched from the project directory upward; **not** a system property, mirroring the `nToUseForStreams` knob in `~/Java-8-Stream-Refactoring`), defaulting to `MODEL_FORWARD_CFA_DEPTH` (4).

## Ariadne release verification

Unit pins cover consumer-reachable behavior only; the axes Ariadne releases change are usually whole-project-emergent. For every bump, after the suite is green (failures are usually pins to advance, not bugs) and the PR merges, run the whole-project verification documented privately in `~/Python-Subjects/scripts/RELEASE-VERIFICATION.md` and report measured deltas upstream, reopening issues whose subject-scale case persists.

## Module layout

Maven reactor (see `pom.xml` `<modules>`):

- `edu.cuny.hunter.hybridize.core` — analysis + refactoring engine. The interesting code is here:
	- `core.refactorings.HybridizeFunctionRefactoringProcessor` — LTK `RefactoringProcessor` orchestrating analysis and the AST transformation.
	- `core.analysis.Function` (+ `FunctionDefinition`, `FunctionExtractor`) — the per-function model: decorator parsing, parameter inference, side-effect/recursion classification.
	- `core.analysis.{Refactoring, Transformation, PreconditionFailure, PreconditionSuccess, Information}` — enums forming the precondition/transformation matrix the tests assert against.
	- `core.wala.ml/` — glue around WALA/Ariadne ML extensions.
	- `lib/` ships several WALA jars (`com.ibm.wala.cast.python.{jython3,ml}-*-SNAPSHOT.jar`) referenced by the bundle classpath.
- `edu.cuny.hunter.hybridize.ui` — Eclipse handlers + wizard surfacing the refactoring under PyDev's "Refactor" menu / Quick Access.
- `edu.cuny.hunter.hybridize.eval` — evaluator plug-in that runs the analysis over whole projects and emits CSVs (uses `commons-csv` from `lib/`). Driven by the launch file above.
- `edu.cuny.hunter.hybridize.tests` — see "Running tests" above.
- `edu.cuny.hunter.hybridize.tests.report` — JaCoCo aggregate-report module (only active under `-Pjacoco`).
- `edu.cuny.hunter.hybridize.feature`, `edu.cuny.hunter.hybridize.updatesite` — Eclipse feature + p2 update site (the README's update-site URL points at the `updatesite/` directory on `main`).

## Conventions worth knowing

- **Formatter**: Spotless is wired to the Eclipse formatter at `Formatting/ponder-formatting.xml` (a git submodule; `git submodule update --init` after cloning). Import order from `Formatting/ponder.importorder`. Java is formatted to Eclipse 4.27 conventions and unused imports are removed. Markdown indents with tabs (4 spaces wide).
- **Compiler is strict**: Tycho passes `-err:all -err:+unused -err:+uselessTypeCheck -err:+unnecessaryElse -err:+invalidJavadoc ...` (see `pom.xml`). New unused locals/imports / inadequate Javadoc tags will fail the build.
- **Whitespace**: CI runs `git show HEAD --check` and fails on whitespace errors in the latest commit.
- Checkstyle config is the symlink `checkstyle.xml -> Formatting/ponder-checkstyle.xml`; violations of `warning` severity or higher fail the build (`failsOnError=true`, `violationSeverity=warning`).
- Bundle `PLUGIN_ID` is read at runtime via `FrameworkUtil.getBundle(...)` — don't hard-code `"edu.cuny.hunter.hybridize.core"`.
- `target/` and the test `cachedir/` are generated; don't commit them.
