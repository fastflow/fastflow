CXX      = g++ 
CXXFLAGS =

FF_ROOT  = ../../../..
#Non-standard OpenCL Location
#OCL_ROOT = /opt/AMDAPP
OCLFLAGS =
OPTFLAGS = -O3
LDFLAGS  = -L$(OCL_ROOT)/lib/x86_64
LIBS     = -lOpenCL -lpthread 
INCLUDES = -I$(FF_ROOT) -I$(OCL_ROOT)/include


TARGETS = ffocl_nested_farm

all: $(TARGETS)

ffocl_nested_farm: farm_mix_cpu_gpu.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OCLFLAGS) $(OPTFLAGS) -o $@ $< $(LFLAGS) $(LIBS)

clean:
	-rm -f $(TARGETS)

cleanall: clean
	rm -f *~
