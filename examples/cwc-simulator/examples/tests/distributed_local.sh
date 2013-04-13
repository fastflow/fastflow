exe=scwc_distr
in=~/repos/svn/scwc/branches/dnode/examples/ecoli/low.cwc
out=ecoli_low
sampling=10
time=1000
nsims=10
nhosts="2"
nworkers=2
seed=6378

#slaves
for slave in 0 1
do
    echo "launching slave #"$slave
    $exe -i $in -o $out -f $seed -s $sampling -t $time -n $nsims --n-worker-sites $slave --role 0 -w $nworkers -v -r > log_slave$slave 2> log_cerr_slave$slave &
done

#master
echo "launching master"
$exe -i $in -o $out -f $seed -s $sampling -t $time -n $nsims --n-worker-sites $nh --role 1 -w $nworkers -v -r 2> log_cerr_master