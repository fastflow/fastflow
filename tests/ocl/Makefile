CXX      = g++ 
CXXFLAGS =

OS = $(shell uname)
ifeq ($(strip $(OS)),Darwin)
  OPENCL =  -framework OpenCL
else
  OPENCL = -lOpenCL
endif

FF_ROOT  = ../../
#AMD OpenCL location
#OCL_ROOT = /opt/AMDAPP
# 
OCLFLAGS =
OPTFLAGS = -O3
LDFLAGS  = -L$(OCL_ROOT)/lib/x86_64
LIBS     =  $(OPENCL) -lpthread 
INCLUDES = -I$(FF_ROOT) -I$(OCL_ROOT)/include


TARGETS = oclMap farm_oclMap oclReduce farm_oclReduce pipeFarmMap dotProduct

all: $(TARGETS)

oclMap: oclMap.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OCLFLAGS) $(OPTFLAGS) -o $@ $< $(LFLAGS) $(LIBS)

oclReduce: oclReduce.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OCLFLAGS) $(OPTFLAGS) -o $@ $< $(LFLAGS) $(LIBS)

farm_oclMap: farm_oclMap.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OCLFLAGS) $(OPTFLAGS) -o $@ $< $(LFLAGS) $(LIBS)

farm_oclReduce: farm_oclReduce.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OCLFLAGS) $(OPTFLAGS) -o $@ $< $(LFLAGS) $(LIBS)

pipeFarmMap: pipeFarmMap.cpp
	$(CXX) -DCHECK $(CXXFLAGS) $(INCLUDES) $(OCLFLAGS) $(OPTFLAGS) -o $@ $< $(LFLAGS) $(LIBS)

dotProduct: dotProduct.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OCLFLAGS) $(OPTFLAGS) -o $@ $< $(LFLAGS) $(LIBS)

clean:
	-rm -f $(TARGETS)

cleanall: clean
	rm -f *~