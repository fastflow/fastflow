img=baboon512_noise90.bmp
img_clean=baboon512.bmp
df=psnr_workers

for v in cluster std border flat
do
  datfile=$df"_"$v".dat"
  echo "psnr over workers" > $datfile
  for w in 1 2 4 8 16
    do
    echo "spd_ff_"$v"_time" -p $w -v -o out.bmp $img
    "spd_ff_"$v"_time" -p $w -v -o out.bmp $img
    r=`psnr_mae out.bmp $img_clean`
    echo -e $w"\t"$r >> $datfile
  done
done
rm out.bmp