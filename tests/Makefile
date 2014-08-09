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
###############################à#########################################
CC                   = gcc
CXX 		     = g++
#CC		     = icc  -mmic
#CXX 		     = icpc -mmic 
LINK_OPT             = 
VERSION              = 
OPTIMIZE_FLAGS       = -O3 -finline-functions #-DTRACE_FASTFLOW
CXXFLAGS             = -DNO_CMAKE_CONFIG -Wall
CFLAGS               =
LDFLAGS              = 
INCS                 = -I../ 
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
TARGET               = simplest test1 test2 test3 test3b test3_farm test4 test5 test6 test7 perf_test1 test_accelerator test_accelerator2 test_accelerator3 test_accelerator_farm+pipe test_accelerator_pipe test_ofarm test_accelerator_ofarm test_accelerator_ofarm_multiple_freezing test_accelerator_pipe+farm test_farm+pipe test_freeze test_masterworker test_multi_masterworker test_pipe+masterworker test_scheduling test_dt test_torus perf_test_alloc1 perf_test_alloc2 perf_test_alloc3 perf_test_noalloc test_uBuffer test_sendq test_spinBarrier test_multi_input test_multi_input2 test_multi_input3 test_multi_input4 test_multi_input5 test_multi_input6 test_multi_input7 test_multi_input8 test_accelerator+pinning test_mpmc test_dataflow test_dataflow2 test_noinput_pipe test_stopstartthreads test_stopstartthreads2 test_stopstartall test_MISD test_parfor test_parfor2 test_parforpipereduce test_dotprod_parfor test_parfor_unbalanced test_parfor_multireduce test_lb_affinity test_farm test_pipe farm_basic pipe_basic perf_parfor perf_parfor2 test_graphsearch test_multi_output test_multi_output2 test_pool1 test_pool2 test_pool3 test_devicequery test_map test_mdf parfor_basic
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

simplest:simplest.o
test1:test1.o
test2:test2.o
test3:test3.o
test3b:test3b.o
test3_farm:test3_farm.o
test4:test4.o
test5:test5.o
test6:test6.o
test7:test7.o
perf_test1:perf_test1.o
test_accelerator:test_accelerator.o

test_accelerator2:test_accelerator2.o
test_accelerator3:test_accelerator3.o
test_accelerator_farm+pipe:test_accelerator_farm+pipe.o
test_accelerator_pipe+farm:test_accelerator_pipe+farm.o
test_accelerator_pipe:test_accelerator_pipe.o
test_accelerator+pinning:test_accelerator+pinning.o
test_ofarm:test_ofarm.o
test_accelerator_ofarm:test_accelerator_ofarm.o
test_accelerator_ofarm_multiple_freezing:test_accelerator_ofarm_multiple_freezing.o
test_accelerator_masterworker:test_accelerator_masterworker.o
test_farm+pipe:test_farm+pipe.o
test_freeze:test_freeze.o
test_masterworker:test_masterworker.o
test_multi_masterworker:test_multi_masterworker.o
test_pipe+masterworker:test_pipe+masterworker.o
test_dt:test_dt.o
test_scheduling:test_scheduling.o
test_sendq:test_sendq.o
test_parfor_unbalanced:test_parfor_unbalanced.cpp
	$(CXX) -std=c++0x $(INCLUDES) $(CXXFLAGS) $(OPTIMIZE_FLAGS) -o $@ $< $(LDFLAGS) $(LIBS)
test_parfor_multireduce:test_parfor_multireduce.cpp
	$(CXX) -std=c++0x $(INCLUDES) $(CXXFLAGS) $(OPTIMIZE_FLAGS) -o $@ $< $(LDFLAGS) $(LIBS)
latency_MPMC:latency_MPMC.o
test_spinBarrier:test_spinBarrier.o
test_multi_input:test_multi_input.o
test_multi_input2:test_multi_input2.o
test_multi_input3:test_multi_input3.o
test_multi_input4:test_multi_input4.o
test_multi_input5:test_multi_input5.o
test_multi_input6:test_multi_input6.o
test_multi_input7:test_multi_input7.o
test_multi_input8:test_multi_input8.o
test_multi_output:test_multi_output.o
test_multi_output2:test_multi_output2.cpp
	$(CXX) -std=c++0x -DHAS_CXX11_VARIADIC_TEMPLATES -DHAS_CXX11_LAMBDA $(INCLUDES) $(CXXFLAGS) $(OPTIMIZE_FLAGS) -o $@ $< $(LDFLAGS) $(LIBS)
test_dataflow:test_dataflow.o
test_dataflow2:test_dataflow2.o
test_noinput_pipe:test_noinput_pipe.o
test_stopstartthreads:test_stopstartthreads.o
test_stopstartthreads2:test_stopstartthreads2.o
test_stopstartall:test_stopstartall.o
test_MISD:test_MISD.o
test_lb_affinity:test_lb_affinity.o
test_farm:test_farm.o
parfor_basic: parfor_basic.cpp
	$(CXX) -std=c++0x -DFF -DHAS_CXX11_VARIADIC_TEMPLATES -DHAS_CXX11_LAMBDA $(INCLUDES) $(CXXFLAGS) $(OPTIMIZE_FLAGS) -o $@ $< $(LDFLAGS) $(LIBS)	
farm_basic:farm_basic.cpp
	$(CXX) -std=c++0x -DHAS_CXX11_VARIADIC_TEMPLATES -DHAS_CXX11_LAMBDA $(INCLUDES) $(CXXFLAGS) $(OPTIMIZE_FLAGS) -o $@ $< $(LDFLAGS) $(LIBS)
pipe_basic:pipe_basic.cpp
	$(CXX) -std=c++0x -DHAS_CXX11_VARIADIC_TEMPLATES $(INCLUDES) $(CXXFLAGS) $(OPTIMIZE_FLAGS) -o $@ $< $(LDFLAGS) $(LIBS)
test_pipe:test_pipe.o
test_parfor:test_parfor.cpp
	$(CXX) -std=c++0x $(INCLUDES) $(CXXFLAGS) $(OPTIMIZE_FLAGS) -o $@ $< $(LDFLAGS) $(LIBS)
test_parfor2:test_parfor2.cpp
	$(CXX) -std=c++0x $(INCLUDES) $(CXXFLAGS) $(OPTIMIZE_FLAGS) -o $@ $< $(LDFLAGS) $(LIBS)
test_parforpipereduce:test_parforpipereduce.cpp
	$(CXX) -std=c++0x -DHAS_CXX11_VARIADIC_TEMPLATES $(INCLUDES) $(CXXFLAGS) $(OPTIMIZE_FLAGS) -o $@ $< $(LDFLAGS) $(LIBS)
test_graphsearch:test_graphsearch.cpp
	$(CXX) -std=c++0x $(INCLUDES) $(CXXFLAGS) $(OPTIMIZE_FLAGS) -o $@ $< $(LDFLAGS) $(LIBS)
test_dotprod_parfor:test_dotprod_parfor.cpp
	$(CXX) -std=c++0x $(INCLUDES) $(CXXFLAGS) $(OPTIMIZE_FLAGS) -o $@ $< $(LDFLAGS) $(LIBS)
test_uBuffer:test_uBuffer.cpp
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(OPTIMIZE_FLAGS) -o $@ $< $(LDFLAGS) $(LIBS)
test_torus:test_torus.cpp
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(OPTIMIZE_FLAGS) -o $@ $< $(LDFLAGS) $(LIBS)
test_mpmc:test_mpmc.cpp
	$(CXX) -DUNBOUNDED_MPMC -Wno-strict-aliasing $(INCLUDES) $(CXXFLAGS) $(OPTIMIZE_FLAGS) -o $@ $< $(LDFLAGS) $(LIBS)
test_mpmc2:test_mpmc2.cpp
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(OPTIMIZE_FLAGS) -o $@ $< $(LDFLAGS) $(LIBS)
test_bmpmc:test_bmpmc.cpp
	$(CXX) -Wno-strict-aliasing $(INCLUDES) $(CXXFLAGS) $(OPTIMIZE_FLAGS) -o $@ $< $(LDFLAGS) $(LIBS)
perf_test_alloc1:perf_test_alloc1.cpp
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(ALLOC) $(OPTIMIZE_FLAGS) -o $@ $< $(LDFLAGS) $(LIBS)
perf_test_alloc2:perf_test_alloc2.cpp
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(ALLOC) $(OPTIMIZE_FLAGS) -o $@ $< $(LDFLAGS) $(LIBS)
perf_test_alloc3:perf_test_alloc3.cpp
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(ALLOC) $(OPTIMIZE_FLAGS) -o $@ $< $(LDFLAGS) $(LIBS)
perf_test_noalloc:perf_test_noalloc.cpp
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(ALLOC) $(OPTIMIZE_FLAGS) -o $@ $< $(LDFLAGS) $(LIBS)
perf_parfor:perf_parfor.cpp
	$(CXX) -std=c++0x $(INCLUDES) $(CXXFLAGS) $(ALLOC) $(OPTIMIZE_FLAGS) -o $@ $< $(LDFLAGS) $(LIBS)
perf_parfor2:perf_parfor2.cpp
	$(CXX) -std=c++0x $(INCLUDES) $(CXXFLAGS) $(ALLOC) $(OPTIMIZE_FLAGS) -o $@ $< $(LDFLAGS) $(LIBS)
test_map:test_map.cpp
	$(CXX) -std=c++0x $(INCLUDES) $(CXXFLAGS) $(ALLOC) $(OPTIMIZE_FLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

test_pool1:test_pool1.cpp
	$(CXX) -std=c++0x $(INCLUDES) $(CXXFLAGS) $(ALLOC) $(OPTIMIZE_FLAGS) -o $@ $< $(LDFLAGS) $(LIBS)
test_pool2:test_pool2.cpp
	$(CXX) -std=c++0x $(INCLUDES) $(CXXFLAGS) $(ALLOC) $(OPTIMIZE_FLAGS) -o $@ $< $(LDFLAGS) $(LIBS)
test_pool3:test_pool3.cpp
	$(CXX) -std=c++0x $(INCLUDES) $(CXXFLAGS) $(ALLOC) $(OPTIMIZE_FLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

test_mdf:test_mdf.cpp
	$(CXX) -std=c++0x $(INCLUDES) $(CXXFLAGS) $(ALLOC) $(OPTIMIZE_FLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

test_devicequery: test_devicequery.o

clean: 
	-rm -fr *.o *~
cleanall: clean
	-rm -fr $(TARGET) *.d 

include $(OBJS:.o=.d)
