for i in 2 4 8 16 32
do
	echo "%model \"lotka-volterra\"" > lotka$i.cwc
	echo "%alphabet" >> lotka$i.cwc
	for j in `seq 1 $i`
	do
		echo "p"$j" P"$j" " >> lotka$i.cwc
	done
	
	echo "%rules" >> lotka$i.cwc
	for j in `seq 1 $i`
	do
		echo "{T} p"$j" ~X >>>[10.0]>>> 2*p"$j" ~X %%" >> lotka$i.cwc
		echo "{T} p"$j" P"$j" ~X >>>[0.01]>>> 2*P"$j" ~X %%" >> lotka$i.cwc
		echo "{T} P"$j" ~X >>>[10.0]>>> ~X %%" >> lotka$i.cwc
	done
	
	echo "%term" >> lotka$i.cwc
	for j in `seq 1 $i`
	do
		echo "1000*p"$j" 1000*P"$j >> lotka$i.cwc
	done
	
	echo "%monitors" >> lotka$i.cwc
	echo "\"preys\": {T} p1 %%" >> lotka$i.cwc
	echo "\"predators\": {T} P1 %%" >> lotka$i.cwc
done

#simulation parameters
time=$1 #time-limit
sub=$2 #sampling period

#get the tools
tool_sequential=scwc_sequential
tool_simd=scwc_sequential_simd

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
touch test_seq
touch test_simd

#main loop
for i in 2 4 8 16 32
do
    $tool_sequential -i lotka$i.cwc -o $res_dir/$m -t $time -s $sub
    $tool_simd -i lotka$i.cwc -o $res_dir/$m -t $time -s $sub
    time_seq=`cat "$res_dir"/"$m"_time_sequential_1x`
    time_simd=`cat "$res_dir"/"$m"_time_sequential_simd_1x`
    echo -e $i"\t"$time_seq >> test_seq
    echo -e $i"\t"$time_simd >> test_simd
done
echo "set output \"simd"."$gnuplot_extension"\" >> $plot_file
echo 'set ylabel "Speed (simulation time / execution time)"' >> $plot_file
echo "plot \"test_seq\" using 1:($time_limit/\$2) with lines title \"Sequential\",\\" >> $plot_file
echo "\"test_simd\" using 1:($time_limit/\$2) with lines title \"SIMD\"" >> $plot_file

#plotting
gnuplot $plot_file
