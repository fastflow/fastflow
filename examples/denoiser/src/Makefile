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


#gcc
CC                   = gcc
CXX 		     = g++
LINK_OPT             =
VERSION              =
OPTIMIZE_FLAGS       = -O3 -finline-functions
CXXFLAGS             = -DNO_CMAKE_CONFIG -std=c++0x #-Wall
CFLAGS               =
LDFLAGS              = 
INCS                 = -I. -I../../.. `pkg-config --cflags opencv`
LIBS                 = -lpthread `pkg-config --libs opencv`
OBJ 		     = parameters.gcc.o mainFF.gcc.o
DEPS                 =

#CUDA
CXX_CUDA_STENCIL 	    = nvcc
OPTIMIZE_FLAGS_CUDA_STENCIL = 
CXXFLAGS_CUDA_STENCIL       = -DNO_CMAKE_CONFIG
INCS_CUDA_STENCIL           = $(INCS) -I/usr/local/cuda-6.0/include 
LIBS_CUDA_STENCIL    	    = $(LIBS) -L/usr/local/cuda-6.0/lib64 -lcuda -lcudart
OBJ_CUDA_STENCIL            = parameters.cuda.o mainCUDAAUTO.cuda.o
DEPS_CUDA_STENCIL    	    =

TARGET               = denoiser denoiser_cuda_stencil




all : $(TARGET)

#gcc
%.gcc.o: %.cpp $(DEPS)
	$(CXX) -c -o $@ $< $(CXXFLAGS) $(OPTIMIZE_FLAGS) $(INCS)

denoiser: $(OBJ)
	$(CXX) -o $@ $^ $(LIBS) $(OPTIMIZE_FLAGS)

#CUDA
%.cuda.o: %.cu $(DEPS)
	$(CXX_CUDA_STENCIL) -c -o $@ $< $(CXXFLAGS_CUDA_STENCIL) $(OPTIMIZE_FLAGS_CUDA_STENCIL) $(INCS_CUDA_STENCIL)	

%.cuda.o: %.cpp $(DEPS)
	$(CXX_CUDA_STENCIL) -c -o $@ $< $(CXXFLAGS_CUDA_STENCIL) $(OPTIMIZE_FLAGS_CUDA_STENCIL) $(INCS_CUDA_STENCIL)

denoiser_cuda_stencil: $(OBJ_CUDA_STENCIL)
	$(CXX_CUDA_STENCIL) -o $@ $^ $(LIBS_CUDA_STENCIL) $(OPTIMIZE_FLAGS_CUDA_STENCIL)



#clean
clean: 
	-rm -fr *.o *~
cleanall: clean
	-rm -fr $(TARGET) *.d
