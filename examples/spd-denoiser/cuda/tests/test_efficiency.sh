img_="lena"
df=efficiency
nw_min=2
nw_max=4
nw_int=""
sizes="1024 2048 4096 8192"
variants="spd_ff"

echo "set term postscript" > gnu_base
echo "set xlabel \"n. workers\"" >> gnu_base
echo "set xrange ["$nw_min":"$nw_max"]" >> gnu_base

for v in $variants
  do
  df_=$df"_"$v
  for size in $sizes
    do
      #speedup
      gnu_speedup=speedup_$v"_"$size".gnu"
      cat gnu_base > $gnu_speedup
      echo "set output \"speedup_"$v"_"$size".eps\"" >> $gnu_speedup
      echo "set ylabel \"speedup\"" >> $gnu_speedup
      echo "set title \"variant: "$v", size: "$size"x"$size"\"" >> $gnu_speedup
      echo -n "plot x title \"ideal\"" >> $gnu_speedup
      #efficiency
      gnu_efficiency=efficiency_$v"_"$size".gnu"
      cat gnu_base > $gnu_efficiency
      echo "set output \"efficiency_"$v"_"$size".eps\"" >> $gnu_efficiency
      echo "set ylabel \"efficiency\"" >> $gnu_efficiency
      echo "set xrange ["$nw_min":"$nw_max"]" >> $gnu_efficiency
      echo "set title \"variant: "$v", size: "$size"x"$size"\"" >> $gnu_efficiency
      echo -n "plot 1 title \"ideal\"" >> $gnu_efficiency
      for noise in 10 50 90
      do
	  #img=$img_$size"x"$size"_noise_"$noise".bmp"
	  img=$img_$size"_noise"$noise".bmp"
	  datfile=$df_"_"$img$size"_noise"$noise".dat"
	  echo "#efficiency" > $datfile
	  echo $v"_time" -c 30 -p 1 -o out.bmp $img
	  t1=`$v"_time" -c 30 -p 1 -o out.bmp $img`
	  for w in $nw_min $nw_int $nw_max
	  do
	      echo $v"_time" -c 30 -p $w -o out.bmp $img
	      tn=`$v"_time" -c 30 -p $w -o out.bmp $img`
	      echo -e $size"\t"$w"\t"$t1"\t"$tn >> $datfile
	  done
	  echo -n ", \""$datfile"\" u 2:(\$3/\$4) title \""$noise" % noisy\" with lp" >> $gnu_speedup
	  echo -n ", \""$datfile"\" u 2:(\$3/\$4/\$2) title \""$noise" % noisy\" with lp" >> $gnu_speedup
      done
  done
done
rm out.bmp
rm gnu_base