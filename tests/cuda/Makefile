CXX      = nvcc
CXXFLAGS = -w  # disable warnings

FF_ROOT  = ../../
CUDAFLAGS= # -arch=compute_30 -code=sm_30 
LDFLAGS  = #-L /usr/local/cuda/lib64/  
OPTFLAGS = -O3
LIBS     = -lcuda -lcudart -lpthread 
INCLUDES = -I$(FF_ROOT)

TARGETS  = cudaMap farm_cudaMap

.PHONY: all clean cleanall 
.SUFFIXES: .cu 

all: 	$(TARGETS)

cudaMap: cudaMap.cu
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(CUDAFLAGS) $(OPTFLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

farm_cudaMap: farm_cudaMap.cu
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(CUDAFLAGS) $(OPTFLAGS) -o $@ $< $(LDFLAGS) $(LIBS)


clean: 
	-rm -f $(TARGETS) *.o
cleanall: clean
	-rm -f *~

