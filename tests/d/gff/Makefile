# ---------------------------------------------------------------------------
#  
#  This file is part of gam.
#
#  gam is free software; you can redistribute it and/or modify it
#  under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  gam is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with gam. If not, see <http://www.gnu.org/licenses/>.
#
# ---------------------------------------------------------------------------

#########################################################################
#
# the following paths are used only for building
#
#########################################################################
FF_HOME              = ../../..
FF_INCS              = $(FF_HOME)
GAM_HOME             ?= $(FF_HOME)/gam
GAM_INCS             = $(GAM_HOME)/include
GAM_RUN              = $(GAM_HOME)/bin/gamrun
GAM_RUN_LOCAL        = $(GAM_HOME)/bin/gamrun-local
GAM_CONF             ?= $(GAM_HOME)/conf/local.conf
GAM_LOCALHOST        ?= localhost

ifdef GAM_TIMEOUT
GAM_CMD              = timeout -s HUP $(GAM_TIMEOUT)s $(GAM_RUN)
GAM_CMD_LOCAL        = timeout -s HUP $(GAM_TIMEOUT)s $(GAM_RUN_LOCAL)
else
GAM_CMD              = $(GAM_RUN)
GAM_CMD_LOCAL        = $(GAM_RUN_LOCAL)

endif

VERBOSE              = -v

#########################################################################
#
# some possible options to exercise:
#  - DGAM_LOG               enable GAM-level logging
#  - DGFF_LOG               enable GFF-level logging
#  - DGAM_DBG               enable internal debugging
#
#########################################################################
CXX 		             ?= g++
LINK_OPT             = 
VERSION              = 
OPTIMIZE_FLAGS       = -O0 -g3
CXXFLAGS             =  -std=c++11 -Wall -DGAM_LOG -DGAM_DBG -DGFF_LOG
LDFLAGS              = 
INCS                 = -I$(GAM_INCS) -I$(FF_INCS) `pkg-config --cflags libfabric`
LIBS                 = -lpthread `pkg-config --libs libfabric`
ARCH                 = -march=$(shell uname -m)

INCLUDES             = -I. $(INCS)
TARGET               = gff_bb_pipeline gff_bb_farm gff_bb_ofarm

.PHONY: all clean distclean
.SUFFIXES: .cpp .o

%.o: %.cpp
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(OPTIMIZE_FLAGS) -c -o $@ $<
%: %.o
	$(CXX) $< -o $@ $(LDFLAGS) $(LIBS)

all: $(TARGET)

gff_bb_pipeline: gff_bb_pipeline.o
gff_bb_farm: gff_bb_farm.o
gff_bb_ofarm: gff_bb_ofarm.o

test: all
	$(GAM_CMD) $(VERBOSE) -n 4 -f $(GAM_CONF) $(PWD)/gff_bb_pipeline
	$(GAM_CMD) $(VERBOSE) -n 6 -f $(GAM_CONF) $(PWD)/gff_bb_farm
	$(GAM_CMD) $(VERBOSE) -n 6 -f $(GAM_CONF) $(PWD)/gff_bb_ofarm

test-local: all
	$(GAM_CMD_LOCAL) $(VERBOSE) -n 4 -l $(GAM_LOCALHOST) $(PWD)/gff_bb_pipeline
	$(GAM_CMD_LOCAL) $(VERBOSE) -n 6 -l $(GAM_LOCALHOST) $(PWD)/gff_bb_farm
	$(GAM_CMD_LOCAL) $(VERBOSE) -n 6 -l $(GAM_LOCALHOST) $(PWD)/gff_bb_ofarm
	
kill:
	killall $(TARGET)

clean: 
	-rm -fr *.o *~

distclean: clean
	-rm -fr $(TARGET)
	-rm -fr *.out *.err *.log
	-rm -fr *.dSYM *.btr
