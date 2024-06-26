#!/bin/bash
set -ex

ML_PATH=$HOME/ML

cp $ML_PATH/com.ibm.wala.cast.python/data/flask.xml .
cp $ML_PATH/com.ibm.wala.cast.python/data/functools.xml .
cp $ML_PATH/com.ibm.wala.cast.python/data/pandas.xml .
cp $ML_PATH/com.ibm.wala.cast.python/data/pytest.xml .
cp $ML_PATH/com.ibm.wala.cast.python/data/click.xml .
cp $ML_PATH/com.ibm.wala.cast.python/data/abseil.xml .
cp $ML_PATH/com.ibm.wala.cast.python.ml/data/tensorflow.xml .
