#!/bin/bash

for f in *.png; do 
    ff=../PNGFigures/${f}
    basedir=$(basename ${ff})
    mkdir -p ${ff}
    mv $f ${ff}; 
done

for f in */*.png; do 
    ff=../PNGFigures/${f}
    basedir=$(basename ${ff})
    mkdir -p ${ff}
    mv $f ${ff}; 
done

for f in */*/*.png; do 
    ff=../PNGFigures/${f}
    basedir=$(basename ${ff})
    mkdir -p ${ff}
    mv $f ${ff}; 
done

for f in */*/*/*.png; do 
    ff=../PNGFigures/${f}
    basedir=$(basename ${ff})
    mkdir -p ${ff}
    mv $f ${ff}; 
done

for f in */*/*/*/*.png; do 
    ff=../PNGFigures/${f}
    basedir=$(basename ${ff})
    mkdir -p ${ff}
    mv $f ${ff}; 
done

for f in */*/*/*/*/*.png; do 
    ff=../PNGFigures/${f}
    basedir=$(basename ${ff})
    mkdir -p ${ff}
    mv $f ${ff}; 
done
