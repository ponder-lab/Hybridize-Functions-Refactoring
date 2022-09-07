# Contributing

Please see [the wiki][wiki] for more information regarding development.

## Building

The project includes a maven configuration file using the Tycho plug-in, which is part of the [maven eclipse plugin](http://www.eclipse.org/m2e). Running `mvn install` will install *most* dependencies. Note that if you are not using maven, this plugin depends on https://github.com/ponder-lab/Common-Eclipse-Refactoring-Framework, the **Eclipse SDK**, **Eclipse SDK tests**, the **Eclipse testing framework** (may also be called the **Eclipse Test Framework**), and [PyDev](https://github.com/fabioz/Pydev). Some of these can be installed from the "Install New Software..." menu option under "Help" in Eclipse. Others need to be obtained from their respective update sites (see below).

## Dependencies

You should have the following projects in your workspace:

1. [Common Eclipse Refactoring Framework](https://github.com/ponder-lab/Common-Eclipse-Refactoring-Framework).

It's also possible just to use `mvn install` if you do not intend on changing any of the dependencies. Alternatively, you may use the following update sites to install the appropriate plugins into your Eclipse installation:

Dependency | Update Site
--- | ---
[Common Eclipse Java Refactoring Framework](https://github.com/ponder-lab/Common-Eclipse-Java-Refactoring-Framework) | https://raw.githubusercontent.com/ponder-lab/Common-Eclipse-Java-Refactoring-Framework/master/edu.cuny.citytech.refactoring.common.updatesite
[PyDev](https://github.com/fabioz/Pydev) | https://www.pydev.org/updates

[wiki]: https://github.com/ponder-lab/Hybridize-Functions-Refactoring/wiki
