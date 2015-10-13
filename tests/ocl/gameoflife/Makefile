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

#find OpenCL
OS = $(shell uname)
ifeq ($(strip $(OS)),Darwin)
  OPENCL =  -framework OpenCL
else
  OPENCL = -lOpenCL
ifndef OCL_ROOT
  $(error The OCL_ROOT variable is not defined in your environment)
endif
  INCS   = -I$(OCL_ROOT)/include
endif

FF_ROOT  = ../../../

CXX      = g++ 
CXXFLAGS = -Wall -std=c++11 -DFF_OPENCL -DNO_CMAKE_CONFIG -DVERIFY #-DPRINT
OPTFLAGS = -O3

LDFLAGS  = -L$(OCL_ROOT)/lib/x86_64   #-L$(OCL_ROOT)/lib64
LIBS     = $(OPENCL) -lpthread 
INCS     += -I$(FF_ROOT) 

OBJ      = gameoflife.o
DEPS     = gameoflifeSRL.hpp $(FF_ROOT)/ff/stencilReduceOCL.hpp
TARGET   = gameoflife



all : $(TARGET)

#gcc
%.o: %.cpp $(DEPS)
	$(CXX) -c -o $@ $< $(CXXFLAGS) $(OPTFLAGS) $(INCS)

gameoflife: $(OBJ)
	$(CXX) $(OPTFLAGS) -o $@ $^ $(LDFLAGS) $(LIBS) 

#clean
clean: 
	-rm -fr *.o *~ ./code_cl/*~
cleanall: clean
	-rm -fr $(TARGET) *.d ./code_cl/*.bin
