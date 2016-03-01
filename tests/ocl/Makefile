CXX      = g++ 
CXXFLAGS = -Wall -std=c++11 -DCHECK -DFF_OPENCL

OS = $(shell uname)
ifeq ($(strip $(OS)),Darwin)
  CXX = clang++	
  OPENCL =  -framework OpenCL
  EXTRA_FLAGS =
else
ifndef OCL_ROOT
$(error The OCL_ROOT variable is not defined in your environment)
endif
  OPENCL = -lOpenCL
  LDFLAGS  = -L$(OCL_ROOT)/lib/x86_64   #-L$(OCL_ROOT)/lib64	
  EXTRA_FLAGS = -finline-functions
endif

FF_ROOT  = ../../

# 
OCLFLAGS =
OPTFLAGS =  -g $(EXTRA_FLAGS) 
LIBS =  $(OPENCL) -lpthread  	
INCLUDES = -I$(FF_ROOT) -I$(OCL_ROOT)/include


TARGETS = helloKernel helloKernel2 testRelease oclStencilReduceLoop oclMap oclMap2 oclReduce oclMapReduce farm_oclMap farm_oclReduce pipeFarmMap dotProduct dotProduct2 device_selection dotProd_cpugpu device_selection2 mixedpipe mixedpipe2 convolution2D


all: $(TARGETS)

helloKernel: helloKernel.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OCLFLAGS) $(OPTFLAGS) -o $@ $< $(LDFLAGS) $(LIBS)
helloKernel2: helloKernel2.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OCLFLAGS) $(OPTFLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

testRelease: testRelease.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OCLFLAGS) $(OPTFLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

oclStencilReduceLoop: oclStencilReduceLoop.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OCLFLAGS) $(OPTFLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

oclMap: oclMap.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OCLFLAGS) $(OPTFLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

oclMap2: oclMap2.cpp
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
dotProduct2: dotProduct2.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OCLFLAGS) $(OPTFLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

dotProd_cpugpu: dotProd_cpugpu.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OCLFLAGS) $(OPTFLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

mixedpipe: mixedpipe.cpp
	$(CXX) -DBUILD_WITH_SOURCE $(CXXFLAGS) $(INCLUDES) $(OCLFLAGS) $(OPTFLAGS) -o $@ $< $(LDFLAGS) $(LIBS)
mixedpipe2: mixedpipe2.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OCLFLAGS) $(OPTFLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

device_selection: device_selection.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OCLFLAGS) $(OPTFLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

device_selection2: device_selection2.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OCLFLAGS) $(OPTFLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

convolution2D: convolution2D.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OCLFLAGS) $(OPTFLAGS) -o $@ $< $(LDFLAGS) $(LIBS)


clean:
	-rm -f $(TARGETS)

cleanall: clean
	rm -f *~ cl_code/*.cl.bin
