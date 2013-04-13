gpath=~/scwc/branches/dnode
ex_name=$gpath/build_linux/scwc_distr
#model parameters
input=$gpath/examples/lotka/lotka.cwc
output_prefix=$gpath/examples/tests/lotka
time=30
sampling=0.12

#scalability parameters
ns=96
nw=2

#distr. parameters
ghost=192.168.13.
master=$ghost""111
scatter=$master:5000
fromany=$master:5001
fhp=112
gcmd=$ex_name" -i "$input" -o "$output_prefix" -t "$time" -s "$sampling" -n "$ns" --address1 "$scatter" --address2 "$fromany" -w "$nw

#pseudo-seq. run
echo "* pseudo-seq., n. sims: "$ns", n. stat. workers: "$nw
nh=1
#nw=1
hostname=$ghost$fhp
cmd=$gcmd" --role 0 --n-worker-sites 0 --slices=1"
echo "> launching host #0 on "$hostname
ssh $hostname -f $cmd > $output_prefix"_"cout_slave$s 2> /dev/null
#master
cmd=$gcmd" --role 1 --n-worker-sites "$nh
echo "> launching master ("$nh " hosts)"
$cmd -v | tee tmp
t[$nh]=`tail -n 1 tmp`

#multi-hosts runs
nh=$1
echo "* n. hosts: "$nh", n. sims: "$ns", n. stat. workers: "$nw
fname=TIMES"_"$ns"x_"$nh"h"

#slaves
for s in `seq 0 $(($nh - 1))`
do
    hostname=$ghost""$((fhp + $s))
    cmd=$gcmd" --role 0 --n-worker-sites "$s
    echo "> launching host #"$s" on "$hostname
    ssh $hostname -f $cmd > $output_prefix"_"cout_slave$s 2> /dev/null
done

#master
cmd=$gcmd" --role 1 --n-worker-sites "$nh
echo "> launching master ("$nh " hosts)"
$cmd -v | tee tmp
t[$nh]=`tail -n 1 tmp`
echo -e $nh"\t"${t[1]}"\t"${t[$nh]} | tee $fname

rm tmp