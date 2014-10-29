FF_ROOT			= ../../..
CC			= gcc
CXX			= g++
LINK_OPT		= 
OPTIMIZE_FLAGS		= -O3 -finline-functions
CXXFLAGS	     	= -Wall
CFLAGS		     	=
LDFLAGS		     	= 
INCS 		     	= -I. -I../../include -I$(FF_ROOT) 

FASTFLOW_INCLUDES	= -I$(FF_ROOT)

PLASMA_INCLUDES		= `PKG_CONFIG_PATH=$(PLASMA_ROOT)/lib/pkgconfig pkg-config plasma --cflags`
PLASMA_LIBS		= `PKG_CONFIG_PATH=$(PLASMA_ROOT)/lib/pkgconfig pkg-config plasma --libs` -lplasma

MKL_INCLUDES		= -I$(MKL_ROOT)/include
MKL_LIBS		= -L$(MKL_ROOT)/lib/intel64 -lmkl_gf_lp64 -lmkl_sequential -lmkl_core

TARGET			= ff_chol_mdf

.PHONY: all clean cleanall distclean install uninstall

all: $(TARGET)

ff_chol_mdf:ff_chol_mdf.cpp 
	$(CXX) -std=c++0x $(CXXFLAGS)  $(FASTFLOW_INCLUDES) $(MKL_INCLUDES) $(OPTIMIZE_FLAGS) -o $@ $< $(LDFLAGS) $(MKL_LIBS) -lm -lpthread 

testPLASMA_static:testPLASMA_stable.c
	$(CXX) -std=c++0x $(CXXFLAGS)  $(FASTFLOW_INCLUDES) $(PLASMA_INCLUDES) $(MKL_INCLUDES) $(OPTIMIZE_FLAGS) -o $@ $< $(LDFLAGS) $(MKL_LIBS) $(PLASMA_LIBS) -lm -lpthread 

testPLASMA_dynamic:testPLASMA_stable.c
	$(CXX) -std=c++0x $(CXXFLAGS)  $(FASTFLOW_INCLUDES) $(PLASMA_INCLUDES) $(MKL_INCLUDES) $(OPTIMIZE_FLAGS) -o $@ $< $(LDFLAGS) $(MKL_LIBS) $(PLASMA_LIBS) -lm -lpthread 

example_dpotrf:example_dpotrf.c
	$(CXX) -std=c++0x $(CXXFLAGS)  $(FASTFLOW_INCLUDES) $(PLASMA_INCLUDES) $(MKL_INCLUDES) $(OPTIMIZE_FLAGS) -o $@ $< $(LDFLAGS) $(MKL_LIBS) $(PLASMA_LIBS) -lm -lpthread 


clean: 
	-rm -fr *.o *~
cleanall: clean
	-rm -fr $(TARGET)
