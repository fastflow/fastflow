#clean: img_2048x2048.bmp
#noisy: img_2048x2048_noise_90.bmp

img=img
img_label=space
w=1
df_=psnr_mae_
gf_=psnr_mae_
cmin=1
cmax=10
#res=2048
#noise=50
#cycles="1 5 10 15 20 30 40 50 60 80 100 120"
ress="256 512"
noises="10 30"
cycles=$cmin" 5 "$cmax

echo "set term postscript" > gnubase
echo "set xlabel \"n. cycles\"" >> gnubase 
echo "set ylabel \"PSNR\"" >> gnubase
echo "set key center right" >> gnubase
echo "set xrange ["$cmin":"$cmax"]" >> gnubase

img_clean=$img"_"$res"x"$res".bmp"

for noise in $noises
do
    for res in $ress
    do
	gnufile=$gf_$img$res"_noise"$noise".gnu"
	cat gnubase > gnufile_tmp
	echo "set output \"psnr_"$res"_noise"$noise".eps\"" >> gnufile_tmp
	echo "set title \"image: "$img_label", size: "$res"x"$res", noise: "$noise"%\"" >> gnufile_tmp
	echo -n "plot " >> gnufile_tmp
	for variant in ff_flat ff_border ff_std ff_cluster cuda
	do
	    datfile=$df_$img$res"_noise"$noise"_"$variant".dat"
	    echo -e "#%noise\tPSNR\tMAE\tcycles" > $datfile
	    echo -n "\""$datfile"\" u 4:2 title \""$variant"\" with lp," >> gnufile_tmp
	    for c in $cycles
	    do
		img_noisy=$img"_"$res"x"$res"_noise_"$noise".bmp"
		img_clean=$img"_"$res"x"$res".bmp"
		echo "spd_"$variant"_passes" -o out.bmp -p $w -c $c $img_noisy
		passes=`"spd_"$variant"_passes" -o out.bmp -p $w -c $c $img_noisy`
		out=`psnr_mae out.bmp $img_clean`
		echo -e $noise"\t"$out"\t"$passes >> $datfile
	    done
	done
	sed '$s/.$//' gnufile_tmp > $gnufile
    done
done
rm gnubase
rm gnufile_tmp