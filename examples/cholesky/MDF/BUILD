Before compiling please set the following env variable:

export PLASMA_ROOT=<plasma-install-dir>
export MKL_ROOT=/opt/intel/mkl
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/opt/intel/mkl/lib/intel64/:"

make ff_chol_mdf


To run the application:
for((i=1;i<=32;++i)); do ./ff_chol_mdf $i 64 ./chol1024.bin 1024 | grep Time; done

