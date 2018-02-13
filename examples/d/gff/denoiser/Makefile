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
# the following paths are used only for building and launching:
# in a cluster environment, these paths are relative to frontend
#
#########################################################################
FF_HOME              = ../../../..
FF_INCS              = $(FF_HOME)
GAM_HOME             ?= $(FF_HOME)/gam
GAM_INCS             = $(GAM_HOME)/include
GAM_RUN              = $(GAM_HOME)/bin/gamrun
GAM_CONF             ?= $(GAM_HOME)/conf/local.conf

ifdef GAM_TIMEOUT
TIMEOUT_PREFIX       = timeout -s HUP $(GAM_TIMEOUT)s
endif
GAM_CMD              = $(TIMEOUT_PREFIX) $(GAM_RUN)

VERBOSE              = #-v

#########################################################################
#
# application parameters
#
#########################################################################
ifdef FRAME_WIDTH
APPFLAGS += -DFRAME_WIDTH=$(FRAME_WIDTH)
endif
ifdef FRAME_HEIGHT
APPFLAGS += -DFRAME_HEIGHT=$(FRAME_HEIGHT)
endif
ifdef NOISE
APPFLAGS += -DNOISE=$(NOISE)
endif
ifdef CYCLES
APPFLAGS += -DCYCLES=$(CYCLES)
endif
ifdef NFRAMES
APPFLAGS += -DNFRAMES=$(NFRAMES)
endif
ifdef NWORKERS
APPFLAGS += -DNWORKERS=$(NWORKERS)
endif
ifdef MWORKERS
APPFLAGS += -DMWORKERS=$(MWORKERS)
endif

GAMCARD = $(shell echo $$(( $(NWORKERS) + 2 )))

#########################################################################
#
# some possible options to exercise:
#  - DGAM_LOG               enable GAM-level logging
#  - DGFF_LOG               enable GFF-level logging
#  - DGAM_DBG               enable internal debugging
#  - DGFF_PROFILE           enable GFF-level profiling
#
#########################################################################
CXX 		             = g++ -std=c++11
NVCC                 = nvcc -ccbin=g++ -std=c++11
LINK_OPT             = 
VERSION              = 
OPTIMIZE_FLAGS       = -O3 -g0
CXXFLAGS             = -DGFF_PROFILE #-Wall -DGAM_LOG -DGAM_DBG -DGFF_LOG
LDFLAGS              = 
INCS                 = -I$(FF_INCS) -I$(GAM_INCS) `pkg-config --cflags libfabric`
LIBS                 = -lpthread `pkg-config --libs libfabric`
ARCH                 = -march=$(shell uname -m)

INCLUDES             = -I. $(INCS)
TARGET               = cuda_denoiser baseline_denoiser seq_denoiser

.PHONY: all clean distclean
.SUFFIXES: .cpp .o

%.o: %.cpp
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(OPTIMIZE_FLAGS) $(APPFLAGS) -c -o $@ $<
%: %.o
	$(CXX) $< -o $@ $(LDFLAGS) $(LIBS)
%: %.cu
	$(NVCC) -w $(INCLUDES) -Xcompiler "-DFF_CUDA $(CXXFLAGS) $(OPTIMIZE_FLAGS) $(APPFLAGS)" $(LDFLAGS) $(LIBS) -o $@ $<

all: $(TARGET)

seq_denoiser: seq_denoiser.o
baseline_denoiser: baseline_denoiser.cu
cuda_denoiser: cuda_denoiser.cu

#########################################################################
#
# test targets
#
#########################################################################

# assume all nodes can reach executables from $(PWD)
cuda-test: cuda_denoiser
	$(GAM_CMD) $(VERBOSE) -n $(GAMCARD) -f $(GAM_CONF) $(PWD)/cuda_denoiser
	
seq-test: seq_denoiser
	$(GAM_CMD) $(VERBOSE) -n $(GAMCARD) -f $(GAM_CONF) $(PWD)/seq_denoiser

BASELINE_PREFIX=logs/baseline_denoiser
BASELINE_LOG=$(BASELINE_PREFIX)/$(shell date +%s)

baseline-test: baseline_denoiser
	mkdir -p $(BASELINE_LOG)
	./baseline_denoiser > $(BASELINE_LOG)/usr.out 2> $(BASELINE_LOG)/usr.err

#########################################################################
#
# cleanup
#
#########################################################################
clean: 
	-rm -fr *.o *~

distclean: clean
	-rm -fr $(TARGET)
	-rm -fr *.out *.err *.log
	-rm -fr *.dSYM *.btr

