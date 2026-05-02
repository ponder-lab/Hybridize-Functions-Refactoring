#!/bin/bash
# Copy Ariadne XML summary files into this Eclipse plug-in so they reach the
# OSGi bundle classpath. See wala/ML#419 for why this consumer-side workaround
# is currently necessary; tracked alongside the broader thin-jar packaging
# refactor at wala/ML#418. When adding a new XML to Ariadne, add a copy line
# here AND list the file in `bin.includes` of `build.properties`.
set -ex

ML_PATH=$HOME/ML

cp "$ML_PATH/com.ibm.wala.cast.python/data/flask.xml" .
cp "$ML_PATH/com.ibm.wala.cast.python/data/functools.xml" .
cp "$ML_PATH/com.ibm.wala.cast.python/data/pandas.xml" .
cp "$ML_PATH/com.ibm.wala.cast.python/data/pytest.xml" .
cp "$ML_PATH/com.ibm.wala.cast.python/data/click.xml" .
cp "$ML_PATH/com.ibm.wala.cast.python/data/abseil.xml" .
cp "$ML_PATH/com.ibm.wala.cast.python.ml/data/tensorflow.xml" .
cp "$ML_PATH/com.ibm.wala.cast.python.ml/data/numpy.xml" .
