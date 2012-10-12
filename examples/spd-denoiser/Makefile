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
OPTIMIZE_FLAGS       = -g  #-DTIME -O6 -finline-functions -fno-rtti  -march=core2 -msse3 -mfpmath=387 -fno-strict-aliasing # -ffast-math -mfpmath=sse -msse2
OMP_FLAGS 	     = -fopenmp
CXXFLAGS             = -Wall #-DALLOCATOR_STATS #-DTRACE_FASTFLOW
CFLAGS               =
LDFLAGS              = 
INCS                 = -I. -I.. -I../.. -I../../.. -I/opt/local/include 
LIBS                 = -lpthread -L/opt/local/lib
BIN 		     = ./bin/

NVCC_ = /usr/local/cuda/bin/nvcc
NVCC  := $(NVCC_) -ccbin /usr/bin
NVCCFLAGS += $(INCS) -m64 --use_fast_math -I./cuda/cuda_utils -I./cuda
LIB_CUDA := -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib -lcudart


SRCS = driver.cpp parameters.cpp cuda/cuda_kernel.cu
INCLUDES = $(INCS)
TARGET = spd_ff spd_cuda
TARGET_EXTRA = spd_cuda_pinned spd_ff_border spd_ff_nd spd_ff_cascade

COMMON_VARIANT_FLAGS = -DTIME -DAVG_TERMINATION # -DSEQ_DETECTION #-DPASSES

.SUFFIXES: .c .cpp .cu .o #.cudaversion.cpp

all: $(TARGET)

#ff (default: flat)
FF_OBJ = parameters.ff.o driver.ff.o #utils.ff.o
FF_FLAGS = -DFLAT $(COMMON_VARIANT_FLAGS)
spd_ff: $(FF_OBJ)
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(ALLOC) $(OPTIMIZE_FLAGS) -o $(BIN)$@ $^ $(LDFLAGS) $(LIBS)
%.ff.o: %.cpp
	$(CXX) -c $(INCLUDES) $(CXXFLAGS) $(ALLOC) $(OPTIMIZE_FLAGS) $(FF_FLAGS) -o $@ $<

#ff_cascade
FF_CASCADE_OBJ = parameters.ff_cascade.o driver.ff_cascade.o #utils.ff_cascade.o
FF_CASCADE_FLAGS = -DCLUSTER $(COMMON_VARIANT_FLAGS)
spd_ff_cascade: $(FF_CASCADE_OBJ)
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(ALLOC) $(OPTIMIZE_FLAGS) -o $(BIN)$@ $^ $(LDFLAGS) $(LIBS)
%.ff_cascade.o: %.cpp
	$(CXX) -c $(INCLUDES) $(CXXFLAGS) $(ALLOC) $(OPTIMIZE_FLAGS) $(FF_CASCADE_FLAGS) -o $@ $<

#cuda
CUDA_OBJ = parameters.cuda.o driver.cuda.o cuda/cuda_kernel.cu.o # utils.cuda.o
CUDA_VARIANT_FLAGS = $(COMMON_VARIANT_FLAGS) #-DFF_WITH_CUDA
NVCCFLAGS_ = $(NVCCFLAGS)

%.cuda.o: %.cpp
	$(NVCC) $(NVCCFLAGS_) -c $(CUDA_VARIANT_FLAGS) -o $@ $<

%.cu.o: %.cu
	$(NVCC) -c $(NVCCFLAGS_) $(CUDA_VARIANT_FLAGS) $< -o $@

spd_cuda: $(CUDA_OBJ)	
	$(NVCC_) --link -o $(BIN)$@ $(NVCCFLAGS_) $(CUDA_VARIANT_FLAGS) $(LDFLAGS) $(LIBS) $(LIB_CUDA) $(CUDA_OBJ)



#ff_border
FF_BORDER_OBJ = parameters.ff_border.o driver.ff_border.o # utils.ff_border.o
FF_BORDER_FLAGS = -DBORDER $(COMMON_VARIANT_FLAGS)
spd_ff_border: $(FF_BORDER_OBJ)
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(ALLOC) $(OPTIMIZE_FLAGS) -o $(BIN)$@ $^ $(LDFLAGS) $(LIBS)
%.ff_border.o: %.cpp
	$(CXX) -c $(INCLUDES) $(CXXFLAGS) $(ALLOC) $(OPTIMIZE_FLAGS) $(FF_BORDER_FLAGS) -o $@ $<

#ff_nd
FF_STD_OBJ = parameters.ff_nd.o driver.ff_nd.o # utils.ff_nd.o
FF_STD_FLAGS = $(COMMON_VARIANT_FLAGS)
spd_ff_nd: $(FF_STD_OBJ)
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(ALLOC) $(OPTIMIZE_FLAGS) -o $(BIN)$@ $^ $(LDFLAGS) $(LIBS)
%.ff_nd.o: %.cpp
	$(CXX) -c $(INCLUDES) $(CXXFLAGS) $(ALLOC) $(OPTIMIZE_FLAGS) $(FF_STD_FLAGS) -o $@ $<

#cuda_pinned
CUDA_OBJ_ = parameters.pinned.o driver.pinned.o  cuda/cuda_kernel.cu.pinned.o # utils.pinned.o
NVCCFLAGS_PINNED = $(NVCCFLAGS) -DFF_WITH_CUDA_PINNED_MEMORY

%.pinned.o: %.cpp
	$(NVCC) $(NVCCFLAGS_PINNED) -c $(CUDA_VARIANT_FLAGS) -o $@ $<

%.cu.pinned.o: %.cu
	$(NVCC) -c $(NVCCFLAGS_PINNED) $(CUDA_VARIANT_FLAGS) $< -o $@

spd_cuda_pinned: $(CUDA_OBJ_)	
	$(NVCC_) --link -o $(BIN)$@ $(NVCCFLAGS_PINNED) $(CUDA_VARIANT_FLAGS) $(LDFLAGS) $(LIBS) $(LIB_CUDA) $(CUDA_OBJ_)

#clean
clean: 
	-rm -fr *.o *~ cuda/*.o cuda/*~

cleanall: clean
	cd $(BIN); rm -fr $(TARGET) $(TARGET_EXTRA) *.d 

depend:
	makedepend -- $(NVCCFLAGS) -- $(SRCS)

# DO NOT DELETE

