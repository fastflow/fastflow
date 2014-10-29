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

CC		     = gcc
CXX		     = g++
#CC                  = icc
#CXX 		     = icpc
LINK_OPT             = 
VERSION              = 
OPTIMIZE_FLAGS       = -DNDEBUG -O3 -finline-functions -fargument-noalias-global
CUDA_OPTIMIZE_FLAGS  = -O3
OMP_FLAGS 	     = -fopenmp -lrt -ldl #-openmp
CXXFLAGS             = -DNO_CMAKE_CONFIG -Wall #-DALLOCATOR_STATS #-DTRACE_FASTFLOW
CUDA_CXXFLAGS	     = 
CFLAGS               =
LDFLAGS              = 
CUDA_LDFLAGS         = -L ${CUDA_ROOT}/lib64  
INCS                 = -I../../
LIBS                 = -lpthread
CUDA_LIBS	     = -lcudart -lpthread #-lcuda
CUDA_INCS	    = 
INTEL_LIBS           = -L /opt/intel/tbb/lib/intel64/gcc4.4 -ltbb -lrt -ldl
INTEL_LIBS_MIC       = -L /opt/intel/tbb/lib/mic -ltbb  

INCLUDES             = -I. $(INCS) $(CUDA_INCS)
INTEL_INCLUDES       = -I /opt/intel/tbb/include
TBB_INCLUDES         = -I /opt/local/include/tbb/

TARGET               = jacobi_omp jacobi_ff jacobi_tbb jacobi_stencil jacobi_CUDA

all: $(TARGET)

jacobi_ff:jacobi.cpp
	$(CXX) -std=c++0x -DNO_PARFOR_SCHEDULER_THREAD $(INCLUDES) $(CXXFLAGS) $(ALLOC) $(OPTIMIZE_FLAGS) -o $@ $< $(LDFLAGS) $(LIBS)
jacobi_ff_mic:jacobi.cpp
	$(CXX) -mmic -std=c++0x -DNO_PARFOR_SCHEDULER_THREAD  $(INCLUDES) $(CXXFLAGS) $(ALLOC) $(OPTIMIZE_FLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

jacobi_tbb:jacobi.cpp
	$(CXX) -std=c++0x -DUSE_TBB -DUSE_LAMBDA -DHAS_CXX11_AUTO -DHAS_CXX11_LAMBDA $(INCLUDES) $(INTEL_INCLUDES) $(CXXFLAGS) $(ALLOC) $(OPTIMIZE_FLAGS) -o $@ $< $(LDFLAGS) $(LIBS) $(INTEL_LIBS)
jacobi_tbb_mic:jacobi.cpp
	$(CXX) -mmic -std=c++0x -DUSE_TBB -DUSE_LAMBDA -DHAS_CXX11_AUTO -DHAS_CXX11_LAMBDA $(INCLUDES) $(INTEL_INCLUDES) $(CXXFLAGS) $(ALLOC) $(OPTIMIZE_FLAGS) -o $@ $< $(LDFLAGS) $(LIBS) $(INTEL_LIBS_MIC)

jacobi_stencil:jacobi_stencil.cpp
	$(CXX) -std=c++0x $(INCLUDES) $(CXXFLAGS) $(ALLOC) $(OPTIMIZE_FLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

jacobi_omp:jacobi.cpp
	$(CXX) -std=c++0x -DUSE_OPENMP $(INCLUDES) $(CXXFLAGS) $(ALLOC) $(OPTIMIZE_FLAGS) -o $@ $< $(LDFLAGS) $(LIBS) $(OMP_FLAGS)
jacobi_omp_mic:jacobi.cpp
	$(CXX) -mmic -std=c++0x -DUSE_OPENMP $(INCLUDES) $(CXXFLAGS) $(ALLOC) $(OPTIMIZE_FLAGS) -o $@ $< $(LDFLAGS) $(LIBS) $(OMP_FLAGS)

jacobi_CUDA:jacobi_CUDA.cu
	${CUDA_ROOT}/bin/nvcc $(INCLUDES) $(CUDA_CXXFLAGS) $(CUDA_OPTIMIZE_FLAGS) -o $@ $< $(CUDA_LDFLAGS) $(CUDA_LIBS) 
clean: 
	-rm -fr *.o *~
cleanall: clean
	-rm -fr $(TARGET) *.d 

