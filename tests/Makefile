
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

#############################################################################
# Some possible options:
# ^^^^^^^^^^^^^^^^^^^^^^
#
# -DFF_ESAVER           Only for Linux, the run-time will use nanosleep
#                       in busy-waiting loops
# -DNO_DEFAULT_MAPPING  It disables the default thread pinning 
#
# -DBLOCKING_MODE       It enables passive waiting communication protocol,
#                       This option makes the nodes less reactive, but
#                       more energy friendly.
#
# -DTRACE_FASTFLOW      It enable statistics. Typically this has an inpact
#                       on the application performance of about 2-4%
#
# -DMAMMUT              It requires to define MAMMUT_HOME and for collecting
#                       energy consumption metrics -DTRACE_FASTFLOW has to be
#                       defined
#                       if MAMMUT is defined, it will be also used for thread
#                       pinning
#
# -DMAPPING_STRING      It allows to manually set the mapping string used
#                       to pin threads to core contexts. To obtain the
#                       string, please use the 'mapping_string.sh' script
#                       contained in the ff directory.
#
###############################à#############################################
#CC                 ?= gcc
#CXX                ?= g++ -std=c++17 #-DBLOCKING_MODE -DDEFAULT_BUFFER_CAPACITY=32768 -DFF_BOUNDED_BUFFER
#CC                 ?= icc  -mmic
#CXX                ?= icpc -mmic 
#CXX                ?= clang++
#LINK_OPT            = 
#VERSION             = 
#CFLAGS              =
#LDFLAGS             =
CXXFLAGS            += -std=c++20
ifdef DEBUG
    OPTIMIZE_FLAGS  += -g -fno-inline-functions
else
    OPTIMIZE_FLAGS  += -O3 -finline-functions -DNDEBUG
endif

ifdef FF_ESAVER
    CXXFLAGS        += -DFF_ESAVER
endif
ifdef NO_DEFAULT_MAPPING
    CXXFLAGS        += -DNO_DEFAULT_MAPPING
endif
ifdef BLOCKING_MODE
    CXXFLAGS        += -DBLOCKING_MODE
endif
ifdef TRACE_FASTFLOW
    CXXFLAGS        += -DTRACE_FASTFLOW
endif
ifdef DEFAULT_BUFFER_CAPACITY
    CXXFLAGS        += -DDEFAULT_BUFFER_CAPACITY=${DEFAULT_BUFFER_CAPACITY}
endif
CXXFLAGS            += -DNO_CMAKE_CONFIG -Wall
INCS                 = -I.. 
LIBS                 = -pthread
#MAMMUT_HOME         = ../mammut
ifdef MAMMUT_HOME
    CXX             += -DMAMMUT
    INCS            += -I $(MAMMUT_HOME)
    LIBS            +=  $(MAMMUT_HOME)/mammut/libmammut.a
endif
ARCH                 = -march=$(shell uname -m)


# FIXME: quick and dirty fix for ARM platform
ifneq ($(findstring $(shell uname -m),armv7l),)
        ARCH = -mcpu=cortex-a9 -march=armv7-a
        LIBS += -lrt -latomic
endif

OS                   = $(shell uname)
ifeq ($(ARCH),-march=x86_64)
  ARCH = -march=core2
endif

ifeq ($(strip $(OS)),Darwin)
  CXXFLAGS        += -DNO_DEFAULT_MAPPING -DBLOCKING_MODE
  ifeq  ($(strip $(ARCH)),x86_64 )
  ARCH = -march=core2
  else ifeq ($(strip $(ARCH)),arm64 )
  ARCH = -arch arm64
  else
  ARCH = -arch ppc
  endif
endif

#INCLUDES            = -I. $(INCS)
INCLUDES             = $(INCS)
TARGET               = simplest test1 test1b test2 test3 test3b test3_farm test4 test5 test6 test7 test8 perf_test1 test_accelerator test_accelerator2 test_accelerator3 test_accelerator_farm+pipe test_accelerator_pipe test_ofarm test_ofarm2 test_accelerator_ofarm test_accelerator_ofarm_multiple_freezing test_accelerator_pipe+farm test_farm+pipe test_farm+pipe2 test_freeze test_masterworker bench_masterworker test_multi_masterworker test_pipe+masterworker test_scheduling test_dt test_torus test_torus2 perf_test_alloc1 perf_test_alloc2 perf_test_alloc3 perf_test_noalloc test_uBuffer test_sendq test_spinBarrier test_multi_input test_multi_input2 test_multi_input3 test_multi_input4 test_multi_input5 test_multi_input6 test_multi_input7 test_multi_input8 test_multi_input9 test_multi_input10 test_multi_input11 test_accelerator+pinning test_dataflow test_dataflow2 test_noinput_pipe test_stopstartthreads test_stopstartthreads2 test_stopstartthreads3 test_stopstartall test_MISD test_parfor test_parfor2 test_parforpipereduce test_dotprod_parfor test_parfor_unbalanced test_parfor_multireduce test_parfor_multireduce2 test_lb_affinity test_farm test_farm2 test_pipe test_pipe2 perf_parfor perf_parfor2 test_graphsearch test_multi_output test_multi_output2 test_multi_output3 test_multi_output4 test_multi_output5 test_multi_output6 test_pool1 test_pool2 test_pool3 test_devicequery test_map test_mdf test_taskf latptr11 test_taskcallbacks test_eosw test_nodeselector test_stats test_dc test_combine test_combine1 test_combine2 test_combine3 test_combine4 test_combine5 test_combine6 test_combine7 test_combine8 test_combine9 test_combine10 test_combine11 test_combine12 test_combine13 test_combine14 test_all-to-all test_all-to-all2 test_all-to-all3 test_all-to-all4 test_all-to-all5 test_all-to-all6 test_all-to-all7 test_all-to-all8 test_all-to-all9 test_all-to-all10 test_all-to-all11 test_all-to-all12 test_all-to-all13 test_all-to-all14 test_all-to-all15 test_all-to-all16 test_all-to-all17 test_all-to-all18 test_all-to-all19 test_optimize test_optimize2 test_optimize3 test_optimize4 test_optimize5 test_all-or-none test_farm+farm test_farm+farm2 test_farm+A2A test_farm+A2A2 test_farm+A2A3 test_farm+A2A4 test_staticallocator test_staticallocator2 test_staticallocator3 test_changenode test_changesize test_changesize2


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
%.o: %.c
	$(CC) $(INCLUDES) $(CFLAGS) -c -o $@ $<
%: %.cpp
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(OPTIMIZE_FLAGS) -o $@ $< $(LDFLAGS) $(LIBS)



all: $(TARGET)

tests: test
test: all
	@echo "*****************************************************"
	@echo "*                                                   *"
#	@echo "* make test(s) is rather naive at moment, it just   *"
#	@echo "* checks compilation and execution with default     *"
#	@echo "* argument parameters                               *"	
	@echo "* FastFlow - efficient parallel streaming runtime   *"
	@echo "* https://github.com/fastflow                       *"
	@echo "* uname -m: riscv, nproc: 4                         *"
	@echo "* running test suite with 2-8 threads               *"
	@echo "*                                                   *"
	@echo "*****************************************************"
	@./runtests.sh

#test_scheduling2:test_scheduling2.cpp
#	$(CXX) -DLB_CALLBACK $(INCLUDES) $(CXXFLAGS) $(OPTIMIZE_FLAGS) -o $@ $< $(LDFLAGS) $(LIBS)
test_mpmc:test_mpmc.cpp
	$(CXX) -DUNBOUNDED_MPMC -Wno-strict-aliasing $(INCLUDES) $(CXXFLAGS) $(OPTIMIZE_FLAGS) -o $@ $< $(LDFLAGS) $(LIBS)
test_mpmc2:test_mpmc2.o
test_bmpmc:test_bmpmc.cpp
	$(CXX) -Wno-strict-aliasing $(INCLUDES) $(CXXFLAGS) $(OPTIMIZE_FLAGS) -o $@ $< $(LDFLAGS) $(LIBS)
test_taskcallbacks:test_taskcallbacks.cpp
	$(CXX) -DFF_TASK_CALLBACK $(INCLUDES) $(CXXFLAGS) $(ALLOC) $(OPTIMIZE_FLAGS) -o $@ $< $(LDFLAGS) $(LIBS)
test_stats:test_stats.cpp
	$(CXX) -DTRACE_FASTFLOW $(INCLUDES) $(CXXFLAGS) $(ALLOC) $(OPTIMIZE_FLAGS) -o $@ $< $(LDFLAGS) $(LIBS)	

# test_taskf2:test_taskf2.cpp
# 	$(CXX) $(INCLUDES) $(CXXFLAGS) $(ALLOC) $(OPTIMIZE_FLAGS) -o $@ $< $(LDFLAGS) $(LIBS)
# test_taskf3:test_taskf3.cpp
# 	$(CXX) $(INCLUDES) $(CXXFLAGS) $(ALLOC) $(OPTIMIZE_FLAGS) -o $@ $< $(LDFLAGS) $(LIBS)


test_dc: test_dc.cpp
	$(CXX) -DDONT_USE_FFALLOC $(INCLUDES) $(CXXFLAGS) $(OPTIMIZE_FLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

ifdef MAMMUT_HOME
test_mammut: test_mammut.cpp
	$(CXX) -DTRACE_FASTFLOW $(INCLUDES) $(CXXFLAGS) $(OPTIMIZE_FLAGS) -o $@ $< $(LDFLAGS) $(LIBS)
else
test_mammut:
	@echo "The test cannot be compiled, please enable MAMMUT_HOME first."
endif

clean: 
	-rm -fr *.o *~
cleanall: clean
	-rm -fr $(TARGET) *.d 

include $(OBJS:.o=.d)
