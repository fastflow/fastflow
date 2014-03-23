CXX      = ${CUDA_ROOT}/bin/nvcc
CXXFLAGS = -w  # disable warnings

FF_ROOT  = ../../
CUDAFLAGS= #-arch=compute_30 -code=sm_30  -ccbin /usr/bin/clang++ -Xcompiler -stdlib=libstdc++
LDFLAGS  = -L ${CUDA_ROOT}/lib64  
OPTFLAGS = -DCHECK -g -G #-O3
LIBS     = -lcudart -lpthread #-lcuda
INCLUDES = -I$(FF_ROOT)

TARGETS  = cudaMap farm_cudaMap farm_cudaStencil cudaMapManaged farm_cudaReduce

.PHONY: all clean cleanall 
.SUFFIXES: .cu 

all: 	$(TARGETS)

cudaMap: cudaMap.cu
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(CUDAFLAGS) $(OPTFLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

cudaMapManaged: cudaMapManaged.cu
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(CUDAFLAGS) $(OPTFLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

farm_cudaMap: farm_cudaMap.cu
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(CUDAFLAGS) $(OPTFLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

farm_cudaStencil: farm_cudaStencil.cu
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(CUDAFLAGS) $(OPTFLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

farm_cudaReduce: farm_cudaReduce.cu
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(CUDAFLAGS) $(OPTFLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

clean: 
	-rm -f $(TARGETS) *.o
cleanall: clean
	-rm -f *~

