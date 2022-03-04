#!/bin/bash

## Configure the parametric perf
ITEMS=1000
BYTExITEM=1000
EXECTIMESOURCE=10
EXECTIMESINK=10
WORKERSXPROCESS=1



## this scripts allow to run a solution in the cluster automatically assigning addresses and ports ##

if [ "$#" -lt 3 ]; then
    echo "Illegal number of parameters"
    echo "usege: $0 <executable> #processesSx #processesDx #startingnode(Optional)"
    exit -1
fi

start_node=1

if [ -z ${4+x} ]; then start_node=1; else start_node=$4; fi

echo "Running $1 with $2 Sx processes and $3 Dx processes starting from node $start_node"

rm tmpFilePP.json
connString=""

for (( i=0; i<$3-1; i++))
do
    connString+="\"D${i}\", "
done

connString+="\"D$(($3-1))\""

echo "{
    \"groups\" : [" >> tmpFilePP.json

# printing the dx nodes in the configuration file 
for (( i=0; i<$2; i++))
do
    echo "     {   
        \"endpoint\" : \"compute$(($start_node+$i)):8000\",
        \"name\" : \"S${i}\",
        \"OConn\" : [${connString}]
     }
     ," >> tmpFilePP.json
done

# printing the dx nodes in the configuration file 
for (( i=0; i<$3; i++))
do
    echo "     {   
        \"endpoint\" : \"compute$(($start_node+$2+$i)):8000\",
        \"name\" : \"D${i}\"
     }" >> tmpFilePP.json
    if [[ $i -lt ${3}-1 ]]; then 
        echo "     ," >> tmpFilePP.json 
    fi
done

echo "    ]
}" >> tmpFilePP.json


### parametric_perf Usage
##  #items #byteXitem #execTimeSource #execTimeSink #nw_sx #nw_dx
###

DFF_RUN_HOME=../../ff/distributed/loader
$DFF_RUN_HOME/dff_run -V -f tmpFilePP.json $1 $ITEMS $BYTExITEM $EXECTIMESOURCE $EXECTIMESINK $2 $3 $WORKERSXPROCESS

#rm tmpFilePP.json
#exiting just for testin purpose
exit 0
