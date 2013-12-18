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
OPTIMIZE_FLAGS       = -O3 -finline-functions #-DTRACE_FASTFLOW
CXXFLAGS             = -DNO_CMAKE_CONFIG -Wall
CFLAGS               =
LDFLAGS              = 
INCS                 = -I../../ 
LIBS                 = -lpthread
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

INCLUDES             = -I./. $(INCS)
TARGET               = l1_ff_nodes_graph l2_generic_dag l2_cyclic_graph l2_cyclic_graph_trace l2_mapping_utils l2_generic_dag_pinning funny-pipeline


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

farm150:farm150.cpp
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(OPTIMIZE_FLAGS) -o $@ $< $(LDFLAGS) $(LIBS) -DFF_BOUNDED_BUFFER

l1_ff_nodes_graph: l1_ff_nodes_graph.o
l2_generic_dag: l2_generic_dag.o
l2_cyclic_graph: l2_cyclic_graph.o
l2_cyclic_graph_trace: l2_cyclic_graph_trace.o
l2_mapping_utils: l2_mapping_utils.o
l2_generic_dag_pinning: l2_generic_dag_pinning.o
funny-pipeline: funny-pipeline.o

clean: 
	-rm -fr *.o *~
cleanall: clean
	-rm -fr $(TARGET) *.d 

include $(OBJS:.o=.d)
