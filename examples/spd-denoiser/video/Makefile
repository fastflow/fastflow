# ---------------------------------------------------------------------------
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License version 2 as 
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
#
#  As a special exception, you may use this file as part of a free software
#  library without restriction.  Specifically, if other files instantiate
#  templates or use macros or inline functions from this file, or you compile
#  this file and link it with other files to produce an executable, this
#  file does not by itself cause the resulting executable to be covered by
#  the GNU General Public License.  This exception does not however
#  invalidate any other reasons why the executable file might be covered by
#  the GNU General Public License.
#
# ---------------------------------------------------------------------------

CC                   = gcc 
#/usr/local/bin/gcc-4.5.1
CXX 		     = g++
#/usr/local/bin/g++-4.5.1
LINK_OPT             = 
VERSION              = 
OPTIMIZE_FLAGS       =  -DTIME -O6 -finline-functions  -march=core2 -msse3 -mfpmath=387 -fno-strict-aliasing #  -fno-rtti -ffast-math -mfpmath=sse -msse2
OMP_FLAGS 	     = -fopenmp
CXXFLAGS             = # -Wall #-DALLOCATOR_STATS #-DTRACE_FASTFLOW
CFLAGS               =
LDFLAGS              = 
INCS                 = -I. -I.. -I../../../.. -I../.. -I../../.. -I../cuda -I../cuda/cuda_utils -I/opt/local/include 
LIBS                 = -lpthread -lopencv_core -lopencv_video -lopencv_highgui -lopencv_imgproc -lopencv_features2d -lopencv_legacy -L/opt/local/lib
BIN 		     = ./bin/

DEFINITIONS := #-DSEQ_DETECTION #-DTIME
NVCC  := nvcc -ccbin /usr/bin
NVCCFLAGS += $(DEFINITIONS) $(INCS) -m64 --use_fast_math -arch=sm_11 -I./cuda_utils -DFF_WITH_CUDA -DSEQ_DETECTION #-DCUDA_PINNED_MEMORY
LIB_CUDA :=  -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib -lcudart

INCLUDES             = -I. $(INCS) 
TARGET               = tctfb_ff tctfb_ff_cuda #seq_video	
OBJ = video.o ../utils.o 
CUDA_OBJ =  ../utils.cuda.o ../cuda/cuda_kernel.cu.o ../cuda/time_interval.cu.o video.cuda.o
DEPS = 

.SUFFIXES: .c .cpp .cu .o

all: $(TARGET)

#sequential
seq_video: seq_video.o
	$(CXX) $(INCLUDES) $(DEFINITIONS) $(CXXFLAGS) $(ALLOC) $(OPTIMIZE_FLAGS) -o $(BIN)$@ $^ $(LDFLAGS) $(LIBS)

#ff
tctfb_ff: $(OBJ)
	$(CXX) $(INCLUDES) $(DEFINITIONS) $(CXXFLAGS) $(ALLOC) $(OPTIMIZE_FLAGS) -o $(BIN)$@ $^ $(LDFLAGS) $(LIBS)

%.o: %.cpp  $(DEPS)
	$(CXX) -c $(INCLUDES) $(DEFINITIONS) $(CXXFLAGS)  $(ALLOC) $(OPTIMIZE_FLAGS) -o $@ $<

#cuda
tctfb_ff_cuda:  $(CUDA_OBJ)
	nvcc --link -o $(BIN)$@ $(NVCCFLAGS) $(ALLOC) $(LDFLAGS) $(LIBS) $(LIB_CUDA)  $(CUDA_OBJ)

%.cu.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

%.cuda.o: %.cpp  $(DEPS)
	$(NVCC) $(NVCCFLAGS) -c $(INCLUDES) $(CXXFLAGS)  $(ALLOC)   -o $@ $<	

clean: 
	-rm -fr *.o *~ ../*.o ../cuda/*.o
cleanall: clean
	cd $(BIN); rm -fr $(TARGET) *.d 

#dependencies
video.o: videoTaskTypes.hpp ff_accel_video.hpp
tctfb_ff_cuda: ../cuda/cuda_definitions.h
*.cu: ../cuda/cuda_definitions.h
