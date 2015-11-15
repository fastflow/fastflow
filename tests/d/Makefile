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

ifndef ZMQ_PATH
$(error The ZMQ_PATH variable is not defined in your environment)
endif

CC		     = gcc 
CXX 		     = g++ 
#CC                  = icc
#CXX 		     = icpc 
LINK_OPT             = 
VERSION              = 
OPTIMIZE_FLAGS       = -O3 -finline-functions -DEBUG #-DBLOCKING_MODE
CXXFLAGS             = -DNO_CMAKE_CONFIG -Wall -std=c++11 -DEBUG
CFLAGS               =
LDFLAGS              = -L ${ZMQ_PATH}/lib
INCS                 = -I../.. -I${ZMQ_PATH}/include
LIBS                 = -lzmq -lpthread #-lrt
ARCH                 = -march=$(shell uname -m)
OS   		     = $(shell uname)
ifeq ($(ARCH),-march=x86_64)
  ARCH = -march=core2
endif

ifeq ($(strip $(OS)),Darwin)
  ifeq  ($(strip $(ARCH)),x86_64 )
  ARCH = -march=core2
  else
  ARCH = -arch ppc
  endif
endif

INCLUDES             = -I. $(INCS)
TARGET               = unicast broadcast allgather fromany scatter ondemand test11_pipe test11_torus test_marshal test_dinout pipe_farm pipe_farm2 farm farm_farm dmap lat11 lat11_v2 bw11


.PHONY: all clean cleanall distclean install uninstall
.SUFFIXES: .c .cpp .o

%.d: %.cpp
	set -e; $(CXX) -MM $(INCLUDES) $(CXXFLAGS) $< \
		| sed 's/\($*\)\.o[ :]*/\1.o $@ : /g' > $@; \
		[ -s $@ ] || rm -f $@
%.d: %.c
	set -e; $(CC) -MM $(INCLUDES) $(CFLAGS)  $< \
		| sed 's/\($*\)\.o[ :]*/\1.o $@ : /g' > $@; \
		[ -s $@ ] || rm -f $@
%.o: %.cpp
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(OPTIMIZE_FLAGS) -c -o $@ $<
%.o: %.c
	$(CC) $(INCLUDES) $(CFLAGS) -c -o $@ $<
%: %.o
	$(CXX) $< -o $@ $(LDFLAGS) $(LIBS)


all: $(TARGET)

test: all
	@echo "*****************************************************"
	@echo "*                                                   *"
	@echo "* TODO: 'make test' is not being implemented. Sorry!*"
	@echo "*                                                   *"
	@echo "*****************************************************"

unicast:unicast.o
broadcast:broadcast.o
allgather:allgather.o
fromany:fromany.o
scatter:scatter.o
ondemand:ondemand.o
test11_pipe:test11_pipe.o
test11_torus:test11_torus.o
test_marshal:test_marshal.o
test_dinout:test_dinout.o
pipe_farm:pipe_farm.o
pipe_farm2:pipe_farm2.o
farm:farm.o
farm_farm:farm_farm.o
dmap:dmap.o
lat11:lat11.o
lat11_v2:lat11_v2.o
bw11:bw11.o

clean: 
	-rm -fr *.o *~
cleanall: clean
	-rm -fr $(TARGET) *.d 

include $(OBJS:.o=.d)
