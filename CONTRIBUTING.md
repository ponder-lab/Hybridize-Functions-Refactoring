# Contributing

Please see [the wiki][wiki] for more information regarding development.

## Building

The project includes a maven configuration file using the Tycho plug-in, which is part of the [maven eclipse plugin](http://www.eclipse.org/m2e). Running `mvn install` will install *most* dependencies. Note that if you are not using maven, this plugin depends on https://github.com/ponder-lab/Common-Eclipse-Refactoring-Framework, the **Eclipse SDK**, **Eclipse SDK tests**, the **Eclipse testing framework** (may also be called the **Eclipse Test Framework**), and [PyDev]. Some of these can be installed from the "Install New Software..." menu option under "Help" in Eclipse (choose to "work with" "The Eclipse Project Updates"). Others may need to be obtained from their respective update sites (see below).

## Dependencies

You should have the following projects in your workspace:

1. [Common Eclipse Refactoring Framework](https://github.com/ponder-lab/Common-Eclipse-Refactoring-Framework).
1. [PyDev 9.3 branch](https://github.com/ponder-lab/Pydev/tree/pydev_9_3).

Having PyDev in your workspace in its own "working set" is helpful to visualize the structure of the project.
<!-- Checkout a particular release (e.g., `git checkout pydev_9_3_0`), then-->
Import it into your Eclipse workspace under a working set named "PyDev." PyDev is already structured as Eclipse projects; you can simply import it as an existing Eclipse project (select the "search for nested projects" option). You'll need to close the "Mylyn" projects that are imported; they won't build since Mylyn has been removed from Eclipse's standard distribution.

<!-- It's also possible just to use `mvn install` if you do not intend on changing any of the dependencies. Alternatively, -->
You may use the following update sites to install some of the appropriate plugins into your Eclipse installation:

Dependency | Update Site
--- | ---
[Common Eclipse Java Refactoring Framework](https://github.com/ponder-lab/Common-Eclipse-Java-Refactoring-Framework) | https://raw.githubusercontent.com/ponder-lab/Common-Eclipse-Java-Refactoring-Framework/master/edu.cuny.citytech.refactoring.common.updatesite
[PyDev] | https://raw.githubusercontent.com/ponder-lab/Pydev/pydev_9_3/org.python.pydev.updatesite

[wiki]: https://github.com/ponder-lab/Hybridize-Functions-Refactoring/wiki
[PyDev]: https://github.com/ponder-lab/Pydev
