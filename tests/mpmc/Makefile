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

ifdef ALLOCATOR
ALLOC=-D$(shell echo ${ALLOCATOR})
else
ALLOC=
endif

ifeq ($(ALLOC),-DUSE_TBB)
INCS    += -I. -I${TBB_HOME}/include
LDFLAGS  = -L ${TBB_HOME}/build/release/
LIBS    +=  -ltbbmalloc -ltbb 
endif

#########################################################################
# some possible options to exercise:
#  -DFF_ESAVER             only for Linux, the run-time will use nanosleep
#                          in busy-waiting loops
#  -DNO_DEFAULT_MAPPING    no default thread pinning 
#
#
###############################��#########################################
CC                   = gcc
CXX 		     = g++ -std=c++11 # -std=c++0x for most of the code
#CC		     = icc  -mmic
#CXX 		     = icpc -mmic 
#CXX		     = clang++
LINK_OPT             = 
VERSION              = 
OPTIMIZE_FLAGS       = -O3 #-DBLOCKING_MODE -DTRACE_FASTFLOW 
CXXFLAGS             = -DNO_CMAKE_CONFIG -Wall 
CFLAGS               =
LDFLAGS              = 
INCS                 = -I../..
LIBS                 = -pthread
ARCH                 = -march=$(shell uname -m)

# FIXME: quick and dirty fix for ARM platform
ifneq ($(findstring $(shell uname -m),armv7l),)
        ARCH = -mcpu=cortex-a9 -march=armv7-a
        LIBS += -lrt
endif

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
TARGET               = test_mpmc


#test_taskf2 test_taskf3
#test_mpmc2 test_bmpmc latency_MPMC 


.PHONY: all clean cleanall distclean install uninstall runtest.sh
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
	@echo "* make test is rather naive at moment, it just      *"
	@echo "* checks compilation and default running            *"
	@echo "*                                                   *"
	@echo "*****************************************************"
	@./runtests.sh

#latency_MPMC:latency_MPMC.o
test_mpmc:test_mpmc.cpp
	$(CXX) -DUNBOUNDED_MPMC -Wno-strict-aliasing $(INCLUDES) $(CXXFLAGS) $(OPTIMIZE_FLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

clean: 
	-rm -fr *.o *~
cleanall: clean
	-rm -fr $(TARGET) *.d 

include $(OBJS:.o=.d)
