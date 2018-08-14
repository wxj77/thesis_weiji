#!/bin/bash

for f in *; do 
    mv $f ${f/\?/} ; 
done
for f in *; do 
    mv $f ${f/\[/} ;  
done
for f in *; do 
    mv $f ${f/\]/} ; 
done

for f in */*; do 
    mv $f ${f/\?/} ; 
done
for f in */*; do 
    mv $f ${f/\[/} ;  
done
for f in */*; do 
    mv $f ${f/\]/} ; 
done

for f in */*/*; do 
    mv $f ${f/\?/} ; 
done
for f in */*/*; do 
    mv $f ${f/\[/} ;  
done
for f in */*/*; do 
    mv $f ${f/\]/} ; 
done

for f in */*/*/*; do 
    mv $f ${f/\?/} ; 
done
for f in */*/*/*; do 
    mv $f ${f/\[/} ;  
done
for f in */*/*/*; do 
    mv $f ${f/\]/} ; 
done

for f in */*/*/*/*; do 
    mv $f ${f/\?/} ; 
done
for f in */*/*/*/*; do 
    mv $f ${f/\[/} ;  
done
for f in */*/*/*/*; do 
    mv $f ${f/\]/} ; 
done

for f in */*/*/*/*/*; do 
    mv $f ${f/\?/} ; 
done
for f in */*/*/*/*/*; do 
    mv $f ${f/\[/} ;  
done
for f in */*/*/*/*/*; do 
    mv $f ${f/\]/} ; 
done




