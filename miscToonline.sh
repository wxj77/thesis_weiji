#!/bin/bash
# how to use: bash miscToonline.sh file

sed -i -e 's/@misc/@online/g' $1
sed -i -e 's/@MISC/@ONLINE/g' $1
sed -i -e 's/{\\_}/\_/g' $1