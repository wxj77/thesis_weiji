#!/bin/bash
COMP="part"
if [ $# \> 0 ]
  then
    COMP=$1    
fi

echo "Compile" ${COMP}

bash ChangeFigName.sh; 
#python pngTojpg.py; 
bash miscToonline.sh thesis.bib; 
rm ${COMP}.bbl; 
rm ${COMP}-blx.bib; 
pdflatex ${COMP}; 
bibtex ${COMP}; 
pdflatex ${COMP};
