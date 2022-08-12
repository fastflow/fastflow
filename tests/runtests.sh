#!/bin/bash

# Author: Massimo 
# Date:   February 2014
#
# It just runs executable files (using default params) 
# contained in the current directory and then checks the exit status
# ( hoping that the program terminates ;) )
#
# 

system=`uname -m`
arch=`uname`
if [ $arch = "Darwin" ]; then
    execfiles="$(find . -maxdepth 1 -type f -perm -a=x| grep -v runtests.sh| grep -v mytime.h)"
else
    execfiles="$(find . -maxdepth 1 -type f -executable| grep -v runtests.sh| grep -v mytime.h)"
fi
echo "*****************************************************************************"
echo "*"
echo "* FastFlow: high-performance parallel patterns and building blocks in C++"
echo "* https://github.com/fastflow/fastflow"
echo "* http://dx.doi.org/10.1002/9781119332015.ch13"
echo "* Running $(wc -w <<< $execfiles) tests on: $system "
test -f /proc/cpuinfo && echo "* CPU cores: $(grep -c processor /proc/cpuinfo)"
test -f /proc/cpuinfo && echo "* $(grep uarch /proc/cpuinfo | head -n1)"
echo "*"
echo "*****************************************************************************"
count=0
success=0
failure=0
for file in $execfiles
do
    ((count=count+1))
    echo -n "[$count] $file: "
    $($file &> /dev/null)
    if [ $? -eq 0 ]; then
     	echo "OK"
	((success=success+1))
    else
    	echo "FAILED"
	((failure=failure+1))
    fi
done
echo "Done. $count tests: $success completed with success, $failure failed."
