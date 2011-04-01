#simulation parameters
m=$1 #model name (without extension)
time=$2 #time-limit
sub=$3 #sampling period

#simulations and workers
sb=$4 #simulations: start
se=$5 #simulations: end
kb=$6 #workers: start
ke=$7 #workers: end

#scheduling
slices=$8
inflight=$9

#get the tools
tool_sequential=scwc_sequential
tool_parallel=scwc_parallel_lockfree

#prepare folders
res_dir=`date +%s`
#res_dir="1280489328"
plot_dir=$res_dir/plots
if !(test -d $res_dir)
then mkdir $res_dir
fi
if !(test -d $plot_dir)
then mkdir $plot_dir
fi

#prepare
gnuplot_terminal=svg
gnuplot_extension=svg
plot_base=$plot_dir/plot
echo 'set grid' > $plot_base
echo 'set xlabel "N. of workers"' >> $plot_base
echo "set xtics 2,2" >> $plot_base
echo 'set terminal '$gnuplot_terminal >> $plot_base
plot_file=$plot_dir/plot.gnu
cat $plot_base > $plot_file
echo "f(x) = x" >> $plot_file
row="plot "

#main loop
for((n=$sb; n <= se;))
do
    $tool_sequential -i $m.cwc -o $res_dir/$m -t $time -s $sub -n $n
    time_seq=`cat "$res_dir"/"$m"_time_sequential_"$n"x`
    echo "" > "test_"$n"x"
    for((w=$kb; w <= ke;))
    do
	$tool_parallel -i $m.cwc -o $res_dir/$m -t $time -s $sub -n $n -w $w --slices $slices --inflight $inflight
	time_par=`cat "$res_dir"/"$m"_time_parallel_"$n"x_"$w"w`
	echo -e $w"\t"$time_seq"\t"$time_par >> "test_"$n"x"
	w=$(($w + 2))
    done
    row=$row"\"test_"$n"x\" using 1:(\$2/\$3) title \""$n" runs\" with lines, " #seq/par speedup (1)
    n=$((n * 2))
done
row=$row"f(x) title \"ideal\""
echo "set output \"performances_"$m"."$gnuplot_extension"\"" >> $plot_file
echo 'set ylabel "Speedup (parallel vs sequential)"' >> $plot_file
echo $row >> $plot_file

#plotting
gnuplot $plot_file
mv *.$gnuplot_extension $plot_dir

#clean up
rm $plot_base
rm $plot_dir/*.gnu
rm test*
