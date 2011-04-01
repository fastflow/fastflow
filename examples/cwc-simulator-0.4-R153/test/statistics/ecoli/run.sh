#simulation parameters
m1=low
m2=high
time=600000
sub_points=250
confidence=$2
n=$1

#get the tool
tool=scwc_parallel_lockfree

#prepare folders
res_dir=`date +%s`_$n
plot_dir=$res_dir/plots
if !(test -d $res_dir)
then mkdir $res_dir
fi
if !(test -d $plot_dir)
then mkdir $plot_dir
fi

#prepare
gnuplot_term=svg
gnuplot_ext=svg
sub=$(echo "scale=10; $time/$sub_points" | bc)
plot_base=$plot_dir/plot
echo 'set grid' > $plot_base
echo 'set xlabel "time"' >> $plot_base
echo 'set ylabel "level"' >> $plot_base
echo "set terminal "$gnuplot_term >> $plot_base

#simulation
$tool -i $m1.cwc -o $res_dir/$m1 -t $time -s $sub -n $n -w 8 -c $confidence
$tool -i $m2.cwc -o $res_dir/$m2 -t $time -s $sub -n $n -w 8 -c $confidence

#reductions
plot_file=$plot_dir/plot_"reductions".gnu
cat $plot_base > $plot_file
echo "set output \"reductions."$gnuplot_ext"\"" >> $plot_file
echo "plot \""$res_dir/$m1"_reductions\" using 1:8:9 with yerrorbars title \"PhoProt: mean and "$confidence"%-confidence  interval (low ext. Pi)\",\\" >> $plot_file
echo "\""$res_dir/$m2"_reductions\" using 1:2:3 with yerrorbars title \"PhoProt: mean and "$confidence"%-confidence  interval (high ext. Pi)\"" >> $plot_file
gnuplot $plot_file
mv $m*.$gnuplot_ext $plot_dir

#clean up
rm $plot_base
rm $plot_dir/*.gnu
