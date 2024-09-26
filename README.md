# Hybridize-Functions-Refactoring

[![Build Status](https://app.travis-ci.com/ponder-lab/Hybridize-Functions-Refactoring.svg?token=ysqq4ZuxzD688KNytWSA&branch=main)](https://app.travis-ci.com/ponder-lab/Hybridize-Functions-Refactoring) [![Coverage Status](https://coveralls.io/repos/github/ponder-lab/Hybridize-Functions-Refactoring/badge.svg?branch=main&t=PffqbW)](https://coveralls.io/github/ponder-lab/Hybridize-Functions-Refactoring?branch=main) [![GitHub license](https://img.shields.io/badge/license-Eclipse-blue.svg)](https://github.com/ponder-lab/Hybridize-Functions-Refactoring/raw/master/LICENSE) [![Java profiler](https://www.ej-technologies.com/images/product_banners/jprofiler_small.png)](https://www.ej-technologies.com/products/jprofiler/overview.html)

## Introduction

<img src="https://raw.githubusercontent.com/ponder-lab/Hybridize-Functions-Refactoring/master/edu.cuny.hunter.hybridize.ui/icons/icon.drawio.png" alt="Icon" align="left" height=150px /> Imperative Deep Learning programming is a promising paradigm for creating reliable and efficient Deep Learning programs. However, it is [challenging to write correct and efficient imperative Deep Learning programs](https://dl.acm.org/doi/10.1145/3524842.3528455) in TensorFlow (v2), a popular Deep Learning framework. TensorFlow provides a high-level API (`@tf.function`) that allows users to execute computational graphs using nature, imperative programming. However, writing efficient imperative TensorFlow programs requires careful consideration.

This tool consists of automated refactoring research prototype plug-ins for [Eclipse][eclipse] [PyDev][pydev] that assists developers in writing optimal imperative Deep Learning code in a semantics-preserving fashion. Refactoring preconditions and transformations for automatically determining when it is safe and potentially advantageous to migrate an eager function to hybrid and improve upon already hybrid Python functions are included. The approach utilizes the [WALA][wala] [Ariadne][ariadne] static analysis framework that has been modernized to TensorFlow 2 and extended to work with modern Python constructs and whole projects. The tool also features a side-effect analysis that is used to determine if a Python function is safe to hybridize.

## Screenshot

Coming soon!

## Demonstration

Coming soon!

## Usage

Coming soon!

## Installation

Coming soon!

### Update Site

Coming soon!

### Eclipse Marketplace

Coming soon!

## Contributing

For information on contributing, see [CONTRIBUTING.md][contrib].

[wiki]: https://github.com/ponder-lab/Hybridize-Functions-Refactoring/wiki
[eclipse]: http://eclipse.org
[contrib]: https://github.com/ponder-lab/Hybridize-Functions-Refactoring/blob/main/CONTRIBUTING.md
[pydev]: http://www.pydev.org/
[wala]: https://github.com/wala/WALA
[ariadne]: https://github.com/wala/ML
