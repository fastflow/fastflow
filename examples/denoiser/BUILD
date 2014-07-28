Requirements

- g++ 
- FFMPEG
- OpenCV 2.4.x 
- libgtk-dev
- FastFlow

==================
Compiling FFMPEG
==================

 $ ./configure --enable-shared --disable-static --enable-nonfree ; make ; make install
 
==================== 
Compiling OpenCV 
====================

$ export LD_LIBRARY_PATH=/ffmpeg_install_path/lib/
$ export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/ffmpeg_install_path/lib/pkgconfig
$ export PKG_CONFIG_LIBDIR=$PKG_CONFIG_LIBDIR:/ffmpeg_install_path/lib/

.../opencv/build$ cmake -DWITH_CUDA=OFF -DWITH_OPENCL=OFF -DBUILD_EXAMPLES=OFF -DBUILD_PNG=ON ..

==================== 
Compiling Denoiser 
====================

$ export OpenCV_DIR="path containing OpenCVConfig.cmake"

cd to fastflow root directory

fastflow$ mkdir build ; cd build
fastflow/build$ cmake ../src
fastflow/build$ make denoiser denoiser_cuda_auto

=======
Testing
======= 
denoiser/build$ ./denoiser  video.avi  -s -n 10 -v


 