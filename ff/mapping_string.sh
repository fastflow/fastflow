#!/bin/bash
#
# Author: Massimo Torquati
# 
# Requires: bash >=4, hwloc
#
# The script builds a string containing the list of logical core of the
# machine that are topologically contiguous, i.e the first context of the
# first core, the first context of the second core, and so on up to the
# last context of the last core.
#
# Example: Given the following OS numbering for the logical cores
#
#   CPU0:                 CPU1:
#    node0   || node1       node2 ||  node3
#   ------------------    ------------------ 
#   | 0 | 4 || 2 | 6 |    | 1 | 5 || 3 | 7 |
#   | 8 |12 ||10 |14 |    | 9 |13 ||11 |15 |
#   ------------------    ------------------
#
# the string produced is:
# "0,4,2,6,1,5,3,7,8,12,10,14,9,13,11,15".
#

topocmd=lstopo # or lstopo-no-graphics or hwloc-ls

# It checks if the topocmd is available
command -v $topocmd >/dev/null ||
    { echo >&2 "This script requires hwloc to be installed."; exit 1; }

physical=$($topocmd --only core | wc -l)  # gets the number of physical cores
logical=$($topocmd --only pu | wc -l)     # gets the number of logical cores
# this is the number of contexts per physical core
nway=$(($logical/$physical))    

# It reads lines from standard input into an indexed array variable.
# The right-hand side script returns the ids of the Processing Unit of
# the machine in the linear order.
# Considering the example above the topocmd command returns something like:
# 0 8 4 12 2 10 6 14 1 9 5 13 3 11 7 15.
mapfile -t array < <( $topocmd --only pu | awk -F'[#)]' '{print $3}' )

for((i=0;i<${#array[@]};i+=nway)); do
    for((j=0;j<$nway;++j)); do    
	V[j]+="${array[i+j]},"    
    done
done
for((j=0;j<$nway;++j)); do
     str+=${V[j]}
done
string=${str::-1} # remove the last comma
echo "The string is:"
echo \"$string\"

read -p "Do you want that I change the ./config.hpp file for you? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    sed -i -e "s/#define FF_MAPPING_STRING \"\"/#define FF_MAPPING_STRING \"$string\"/1" ./config.hpp
    if [ $? -eq 0 ]; then
	echo "This is the new FF_MAPPING_STRING variable in the ./config.hpp file:"
	grep -1 "^#define FF_MAPPING_STRING \"" config.hpp
    else
	echo "something went wrong...."
	exit 1
    fi
else
    echo "Ok, nothing has been changed"
fi

exit 0
