# Contributing

Please see our [wiki] for more information regarding development.

## Eclipse Environment

The plug-ins are being developed on the following Eclipse versions. Currently, newer versions of Eclipse will not resolve M2E dependencies. The [Eclipse Installer](https://www.eclipse.org/downloads/packages/installer) can be used to install specific versions:

	Eclipse IDE for RCP and RAP Developers (includes Incubating components)

	Version: 2023-03 (4.27.0)
	Build id: 20230309-1520

## Building

The project includes a Maven configuration file using the Tycho plug-in, which is part of the [Maven Eclipse plug-in](http://www.eclipse.org/m2e). Running `mvn install` will install *most* dependencies. Note that if you are not using Maven, this plugin depends on the [Common Eclipse Refactoring Framework], the **Eclipse SDK**, **Eclipse SDK tests**, the **Eclipse testing framework** (may also be called the **Eclipse Test Framework**), [Ariadne], [WALA], and [PyDev]. Some of these can be installed from the "Install New Software..." menu option under "Help" in Eclipse (choose to "work with" "The Eclipse Project Updates"). Others may need to be obtained from their respective update sites (see below).

### JDK and Maven versions

This project requires **Java 25** (`<maven.compiler.source/target>` and every bundle's `Bundle-RequiredExecutionEnvironment` are `JavaSE-25`) and **Maven 3.9.11+** (Tycho 5.0.2 nominally supports Maven 3.9.9 but in practice trips a `TargetPlatformArtifactResolver` binding error on 3.9.9; see [eclipse-tycho/tycho#5384](https://github.com/eclipse-tycho/tycho/issues/5384)). The `maven-enforcer-plugin` rule pins Maven `[3.9.11,)`.

The repo ships **Maven Wrapper** (`./mvnw`) pinned to Maven 3.9.11. Use `./mvnw` rather than system `mvn`; most system Maven installations are 3.9.9 and will fail on the Tycho binding error described above.

If your system default JDK is something other than 25, you can pin Java 25 only for this repository via [direnv](https://direnv.net). `.envrc` is gitignored; create it locally with contents adjusted for your platform.

### Pre-commit Hooks

The repo ships a [pre-commit](https://pre-commit.com) configuration at `.pre-commit-config.yaml` that runs Black on Python files and `./mvnw spotless:apply` on every commit. Install once after cloning:

```sh
pip install pre-commit && pre-commit install
```

After that, every `git commit` auto-formats touched files before the commit lands, so CI's `spotless:check` and `black --check` stay green. To run all hooks ad-hoc against the whole tree: `pre-commit run --all-files`.

## Dependencies

All dependencies are listed in the [target definition file]. Simply set this file as your "active target", refresh and update the items in the list, and you should be good to go. However, if you plan to run the UI plug-in (and not only the tests or evaluation plug-ins), due to https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/264, you should have the following project in your workspace:

1. [Common Eclipse Refactoring Framework](https://github.com/ponder-lab/Common-Eclipse-Refactoring-Framework).

The other projects should be obtained from the target definition. If, for any reason, they aren't, or you also need to modify those projects, you can put the following projects also in your workspace:

1. [PyDev 9.3 branch][PyDev].
1. [Ariadne ponder-lab fork][Ariadne]
1. [WALA v1.7 branch][WALA]

Having PyDev in your workspace in its own "working set" is helpful to visualize the structure of the project. Import it into your Eclipse workspace under a working set named "PyDev." PyDev is already structured as Eclipse projects; you can simply import it as an existing Eclipse project (select the "search for nested projects" option). You'll need to close the "Mylyn" projects that are imported; they won't build since Mylyn has been removed from Eclipse's standard distribution.

To access the [Ariadne packages] for building your project, refer to [GitHub Packages Documentation] for instructions.

You may use the following update sites to install some of the appropriate plugins into your Eclipse installation:

Dependency | Update Site
--- | ---
[Common Eclipse Refactoring Framework] | https://raw.githubusercontent.com/ponder-lab/Common-Eclipse-Java-Refactoring-Framework/master/edu.cuny.citytech.refactoring.common.updatesite
[PyDev] | https://raw.githubusercontent.com/ponder-lab/Pydev/pydev_9_3/org.python.pydev.updatesite

These update sites are also listed in the [target definition file]. Thus, you shouldn't need them unless you plan to make changes to them. WALA is consumed from Maven Central via Tycho's `<location type="Maven">`; no separate update site is required.

### Running the Evaluator

Use the `edu.cuny.hunter.hybridize.evaluator` plug-in project to run the evaluation. The evaluation process will produce several CSVs, as well as perform the transformation if desired (see below for details). For convenience, there is an [Eclipse launch configuration](https://wiki.eclipse.org/FAQ_What_is_a_launch_configuration%3F) that can be used to run the evaluation. The run configuration is named [`edu.cuny.hunter.hybridize.eval/Evaluate Hybridize Functions.launch`](https://github.com/ponder-lab/Hybridize-Functions-Refactoring/blob/691cbeb87be805b8bfc336e799d938a9064a5e0e/edu.cuny.hunter.hybridize.eval/Evaluate%20Hybridize%20Functions.launch). In the run configuration dialog, you can specify several arguments to the evaluator as system properties.

You can run the evaluator in several different ways: as a command or menu item in the IDE, or *headlessly* from the command line (without the IDE), which is convenient for batch or reproducible runs. Either way, you must evaluate *entire* projects, as the evaluator collects project-level data. See [the wiki][evaluator wiki] for the headless launcher, the wrapper script, and configuration.

#### Configuring the Evaluation

Most evaluator settings are system properties passed as VM arguments (the launch configuration above sets them; the full list is on [the wiki][evaluator wiki]). The targeted k-CFA depth, however, is read *per project* from an `eval.properties` file placed at or above the project root, so each subject project can be analyzed at its own depth. For example, an `eval.properties` containing `targetedCfaDepth=2` evaluates that project at depth 2. See [the wiki][evaluator wiki] for the available keys.

[wiki]: https://github.com/ponder-lab/Hybridize-Functions-Refactoring/wiki
[evaluator wiki]: https://github.com/ponder-lab/Hybridize-Functions-Refactoring/wiki/Running-the-Evaluator
[PyDev]: https://github.com/ponder-lab/Pydev/tree/pydev_9_3
[Common Eclipse Refactoring Framework]: https://github.com/ponder-lab/Common-Eclipse-Refactoring-Framework
[Ariadne]: https://github.com/ponder-lab/ML
[Ariadne packages]: https://github.com/orgs/ponder-lab/packages?repo_name=ML
[GitHub Packages Documentation]: https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-apache-maven-registry#authenticating-to-github-packages
[target definition file]: https://github.com/ponder-lab/Hybridize-Functions-Refactoring/blob/02cbd028d09f063f3e4ecd048e2435262abdde64/hybridize.target
