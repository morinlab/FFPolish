#!/bin/sh

cp -r $SRC_DIR/src/*.py $PREFIX/bin
mkdir $PREFIX/models
cp -r $SRC_DIR/models/* $PREFIX/models
ln -s $PREFIX/bin/cli.py $PREFIX/bin/ffpolish
chmod +x $PREFIX/bin/ffpolish
