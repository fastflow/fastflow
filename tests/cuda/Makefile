CXX      = ${CUDA_ROOT}/bin/nvcc
CXXFLAGS = -w  # disable warnings

FF_ROOT  = ../../
CUDAFLAGS= #-arch=compute_30 -code=sm_30  -ccbin /usr/bin/clang++ -Xcompiler -stdlib=libstdc++
LDFLAGS  = -L ${CUDA_ROOT}/lib64  
OPTFLAGS = -DCHECK -g -std=c++11#-O3
LIBS     = -lcudart -lpthread #-lcuda
INCLUDES = -I$(FF_ROOT)

TARGETS  = cudaMap cudaMapAccelerator farm_cudaMap farm_cudaStencil farm_cudaReduce dotProductCUDA test_pool1CUDA test_pool2CUDA cudaMapManaged
 

.PHONY: all clean cleanall 
.SUFFIXES: .cu 

all: 	$(TARGETS)

cudaMap: cudaMap.cu
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(CUDAFLAGS) $(OPTFLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

cudaMapAccelerator: cudaMapAccelerator.cu
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(CUDAFLAGS) $(OPTFLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

cudaMapManaged: cudaMapManaged.cu
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(CUDAFLAGS) $(OPTFLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

farm_cudaMap: farm_cudaMap.cu
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(CUDAFLAGS) $(OPTFLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

farm_cudaStencil: farm_cudaStencil.cu
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(CUDAFLAGS) $(OPTFLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

farm_cudaReduce: farm_cudaReduce.cu
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(CUDAFLAGS) $(OPTFLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

dotProductCUDA: dotProductCUDA.cu
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(CUDAFLAGS) $(OPTFLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

test_pool1CUDA: test_pool1CUDA.cu
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(CUDAFLAGS) $(OPTFLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

test_pool2CUDA: test_pool2CUDA.cu
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(CUDAFLAGS) $(OPTFLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

clean: 
	-rm -f $(TARGETS) *.o
cleanall: clean
	-rm -f *~

