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


CC                   = gcc 
CXX 		     = g++ 
LINK_OPT             = 
VERSION              = 
OPTIMIZE_FLAGS       = -O6 -finline-functions
CXXFLAGS             = -Wall  -DTRACE_FASTFLOW
CFLAGS               =
LDFLAGS              = 
INCS                 = -I../ 
LIBS                 = -lpthread
ARCH                 = -march=$(shell uname -m)
OS					 = $(shell uname)
ifeq ($(ARCH),-march=x86_64)
  ARCH = -march=core2
endif

ifeq ($(strip $(OS)),Darwin)
  ifeq  ($(strip $(ARCH)),i386)
  ARCH = -march=core2
  else
  ARCH = -arch ppc
  endif
endif

INCLUDES             = -I. $(INCS)
TARGET               = test1 test2 test3 test3b test4 test5 test6 test7 perf_test1 perf_test2 test_accelerator test_accelerator2 test_accelerator_farm+pipe test_freeze test_masterworker test_scheduling test_dt test_torus perf_test_alloc1 perf_test_alloc2 perf_test_alloc3 perf_test_noalloc test_uBuffer test_mpmc


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
	$(CXX) $< -o $@ $(LIBS)


all: $(TARGET)

test: all
	@echo "*****************************************************"
	@echo "*                                                   *"
	@echo "* TODO: 'make test' is not being implemented. Sorry!*"
	@echo "*                                                   *"
	@echo "*****************************************************"

test1:test1.o
test2:test2.o
test3:test3.o
test3b:test3b.o
test4:test4.o
test5:test5.o
test6:test6.o
test7:test7.o
perf_test1:perf_test1.o
perf_test2:perf_test2.o
perf_test3:perf_test3.o
test_accelerator:test_accelerator.o
test_accelerator2:test_accelerator2.o
test_accelerator_farm+pipe:test_accelerator_farm+pipe.o
test_freeze:test_freeze.o
test_masterworker:test_masterworker.o
test_dt:test_dt.o
test_scheduling:test_scheduling.o
test_uBuffer:test_uBuffer.cpp
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(OPTIMIZE_FLAGS) -o $@ $< $(LDFLAGS) $(LIBS)
test_torus:test_torus.cpp
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(OPTIMIZE_FLAGS) -o $@ $< $(LDFLAGS) $(LIBS)
test_mpmc:test_mpmc.cpp
	$(CXX) $(ARCH) -Wno-strict-aliasing $(INCLUDES) $(CXXFLAGS) $(OPTIMIZE_FLAGS) -o $@ $< $(LDFLAGS) $(LIBS)
perf_test_alloc1:perf_test_alloc1.cpp
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(ALLOC) $(OPTIMIZE_FLAGS) -o $@ $< $(LDFLAGS) $(LIBS)
perf_test_alloc2:perf_test_alloc2.cpp
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(ALLOC) $(OPTIMIZE_FLAGS) -o $@ $< $(LDFLAGS) $(LIBS)
perf_test_alloc3:perf_test_alloc3.cpp
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(ALLOC) $(OPTIMIZE_FLAGS) -o $@ $< $(LDFLAGS) $(LIBS)
perf_test_noalloc:perf_test_noalloc.cpp
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(ALLOC) $(OPTIMIZE_FLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

clean: 
	-rm -fr *.o *~
cleanall: clean
	-rm -fr $(TARGET) *.d 

include $(OBJS:.o=.d)
