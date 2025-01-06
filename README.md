# Hybridize-Functions-Refactoring

[![Build Status](https://github.com/ponder-lab/Hybridize-Functions-Refactoring/actions/workflows/maven.yml/badge.svg)](https://github.com/ponder-lab/Hybridize-Functions-Refactoring/actions/workflows/maven.yml) [![Coverage Status](https://coveralls.io/repos/github/ponder-lab/Hybridize-Functions-Refactoring/badge.svg?branch=main&t=PffqbW)](https://coveralls.io/github/ponder-lab/Hybridize-Functions-Refactoring?branch=main) [![GitHub license](https://img.shields.io/badge/license-Eclipse-blue.svg)]([https://github.com/ponder-lab/Hybridize-Functions-Refactoring/raw/master/LICENSE](https://raw.githubusercontent.com/ponder-lab/Hybridize-Functions-Refactoring/refs/heads/main/LICENSE)) [![Java profiler](https://www.ej-technologies.com/images/product_banners/jprofiler_small.png)](https://www.ej-technologies.com/products/jprofiler/overview.html)

## Introduction

<img src="https://raw.githubusercontent.com/ponder-lab/Hybridize-Functions-Refactoring/master/edu.cuny.hunter.hybridize.ui/icons/icon.drawio.png" alt="Icon" align="left" height=150px /> Imperative Deep Learning programming is a promising paradigm for creating reliable and efficient Deep Learning programs. However, it is [challenging to write correct and efficient imperative Deep Learning programs](https://dl.acm.org/doi/10.1145/3524842.3528455) in TensorFlow (v2), a popular Deep Learning framework. TensorFlow provides a high-level API (`@tf.function`) that allows users to execute computational graphs using nature, imperative programming. However, writing efficient imperative TensorFlow programs requires careful consideration.

This tool consists of automated refactoring research prototype plug-ins for [Eclipse][eclipse] [PyDev][pydev] that assists developers in writing optimal imperative Deep Learning code in a semantics-preserving fashion. Refactoring preconditions and transformations for automatically determining when it is safe and potentially advantageous to migrate an eager function to hybrid and improve upon already hybrid Python functions are included. The approach utilizes the [WALA][wala] [Ariadne][ariadne] static analysis framework that has been modernized to TensorFlow 2 and extended to work with modern Python constructs and whole projects. The tool also features a side-effect analysis that is used to determine if a Python function is safe to hybridize.

## Screenshot

![Screenshot](https://khatchad.commons.gc.cuny.edu/wp-content/blogs.dir/2880/files/2024/10/Screenshot-from-2024-10-01-13-07-03.png)

## Usage

The tool is designed to be used in the Eclipse IDE with the PyDev plug-in. Thus, the tool is designed to operate on Python files contained within PyDev projects, as that is where it obtains metadata from the projects such as `PYTHONPATH`. However, currently, the tool is only compatible only with [*our* PyDev 9.3 development branch][pydev branch]. That means that you will have to have our version of PyDev installed in your Eclipse instance before using this plug-in. Thus, if you have a version of PyDev previously installed, you will need to uninstall it before installing our tool. Integration with the standard PyDev version is being tracked by [#152]. Installation via our [update site](#update-site) should install the necessary PyDev version automatically.

Once the plug-in and dependencies are installed, the refactoring can be run in two different ways:

1. As a command.
	1. Select a Python code entity.
	1. Select "Hybridize function..." from the "Quick Access" dialog (CTRL-3).
1. As a menu item.
	1. Right-click on a Python code entity.
	1. Under "Refactor," choose "Hybridize function..."

Currently, the refactoring works only via the package explorer and the outline views. You can either select a code entity to optimize or select multiple entities. In each case, the tool will find functions in the enclosing entity to refactor.

### Update

Due to https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/370, only the "command" is working.

## Installation

The tool has been tested on Eclipse IDE for RCP and RAP Developers Version: 2023-03 (4.27.0), Build id: 20230309-1520 under Java 17.

### Update Site

An alpha version of our tool is available via an Eclipse update site at:

https://raw.githubusercontent.com/ponder-lab/Hybridize-Functions-Refactoring/main/edu.cuny.hunter.hybridize.updatesite

Please choose the latest version.

### Eclipse Marketplace

Coming soon!

### Dependencies

The refactoring has several dependencies as listed below. If you experience any trouble installing the plug-in using the above update site, you can manually install the dependencies. The latest version of the plug-ins should be installed:

Dependency | Update Site
--- | ---
[Common Eclipse Refactoring Framework] | https://raw.githubusercontent.com/ponder-lab/Common-Eclipse-Java-Refactoring-Framework/master/edu.cuny.citytech.refactoring.common.updatesite
[PyDev] | https://raw.githubusercontent.com/ponder-lab/Pydev/pydev_9_3/org.python.pydev.updatesite
[WALA] | https://raw.githubusercontent.com/ponder-lab/WALA/v1.6/com.ibm.wala-repository

## Contributing

For information on contributing, see [CONTRIBUTING.md][contrib].

## Further Information

See the [wiki][wiki] for further information.

[wiki]: https://github.com/ponder-lab/Hybridize-Functions-Refactoring/wiki
[eclipse]: http://eclipse.org
[contrib]: https://github.com/ponder-lab/Hybridize-Functions-Refactoring/blob/main/CONTRIBUTING.md
[pydev]: http://www.pydev.org/
[wala]: https://github.com/wala/WALA
[ariadne]: https://github.com/wala/ML
[pydev branch]: https://github.com/ponder-lab/Pydev/tree/pydev_9_3
[Common Eclipse Refactoring Framework]: https://github.com/ponder-lab/Common-Eclipse-Refactoring-Framework
[#152]: https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/152
