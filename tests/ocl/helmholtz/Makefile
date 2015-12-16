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

# 
# Author: Massimo Torquati <torquati@di.unipi.it> 
# Date:   December 2015
#
#
# You have to set at least the following 2 variables:
# FF_ROOT     pointing to the FastFlow root directory (i.e.
#             the one containing the ff directory).
# OCL_ROOT    pointing to the OpenCL root directory
#
FF_ROOT		= ../../..
OCL_ROOT    = /usr/local/cuda

CXX		= g++ -std=c++11 #-DNO_DEFAULT_MAPPING
INCLUDES	= -I $(FF_ROOT)  -I $(OCL_ROOT)/include
CXXFLAGS  	= -Wall -DFF_OPENCL

LDFLAGS 	= -L$(OCL_ROOT)/lib/x86_64 -lOpenCL -pthread
OPTFLAGS	=  -O3 -DNDEBUG

TARGETS		= helmholtz

.PHONY: all clean cleanall
.SUFFIXES: .cpp 


%: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OPTFLAGS) -o $@ $< $(LDFLAGS)

all		: $(TARGETS)
clean		: 
	rm -f $(TARGETS)
cleanall	: clean
	\rm -f *.o *~ out/*



