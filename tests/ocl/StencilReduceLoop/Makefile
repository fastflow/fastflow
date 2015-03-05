CXX      = g++ 
CXXFLAGS = -Wall -std=c++11 -DCHECK

OS = $(shell uname)
ifeq ($(strip $(OS)),Darwin)
  OPENCL =  -framework OpenCL
else
  OPENCL = -lOpenCL
endif

FF_ROOT  = ../../../
#AMD OpenCL location
OCL_ROOT = /usr/local/cuda

# 
OCLFLAGS =
OPTFLAGS = -O0 -DCHECK -g
LDFLAGS  = -L$(OCL_ROOT)/lib64
LIBS     =  $(OPENCL) -lpthread 
INCLUDES = -I$(FF_ROOT) -I$(OCL_ROOT)/include


TARGETS = oclStencilReduce oclMap oclReduce oclMapReduce farm_oclMap farm_oclReduce pipeFarmMap #dotProduct

all: $(TARGETS)

oclStencilReduce: oclStencilReduce.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OCLFLAGS) $(OPTFLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

oclMap: oclMap.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OCLFLAGS) $(OPTFLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

oclReduce: oclReduce.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OCLFLAGS) $(OPTFLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

oclMapReduce: oclMapReduce.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OCLFLAGS) $(OPTFLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

farm_oclMap: farm_oclMap.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OCLFLAGS) $(OPTFLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

farm_oclReduce: farm_oclReduce.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OCLFLAGS) $(OPTFLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

pipeFarmMap: pipeFarmMap.cpp
	$(CXX) -DCHECK $(CXXFLAGS) $(INCLUDES) $(OCLFLAGS) $(OPTFLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

dotProduct: dotProduct.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OCLFLAGS) $(OPTFLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

clean:
	-rm -f $(TARGETS)

cleanall: clean
	rm -f *~
