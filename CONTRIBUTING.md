# Contributing

Please see [the wiki][wiki] for more information regarding development.

## Building

The project includes a maven configuration file using the Tycho plug-in, which is part of the [maven eclipse plugin](http://www.eclipse.org/m2e). Running `mvn install` will install *most* dependencies. Note that if you are not using maven, this plugin depends on https://github.com/ponder-lab/Common-Eclipse-Refactoring-Framework, the **Eclipse SDK**, **Eclipse SDK tests**, and the **Eclipse testing framework** (may also be called the **Eclipse Test Framework**). Some of these can be installed from the "Install New Software..." menu option under "Help" in Eclipse.

## Dependencies

You should have the following projects in your workspace:

1. [Common Eclipse Refactoring Framework](https://github.com/ponder-lab/Common-Eclipse-Refactoring-Framework).

It's also possible just to use `mvn install` if you do not intend on changing any of the dependencies.

[wiki]: https://github.com/ponder-lab/Hybridize-Functions-Refactoring/wiki
