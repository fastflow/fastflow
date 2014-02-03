#!/bin/bash

# Author: Massimo 
# Date:   February 2014
#
# It just runs executable files (using default params) 
# contained in the current directory and then checks the exit status
# ( if the program terminates ;) )
#
#

execfiles="$(find . -type f -executable| grep -v runtests.sh)"

for file in $execfiles
do    
    echo -n "$file: "
    $($file &> /dev/null)
    if [ $? -eq 0 ]; then
     	echo "OK"
    else
    	echo "FAILED"
    fi
done
echo 
echo "DONE"