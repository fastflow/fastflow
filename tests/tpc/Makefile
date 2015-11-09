CXX      = arm-xilinx-linux-gnueabi-g++
CXXFLAGS = -std=c++11 -DCHECK -DFF_TPC -Wall
FF_ROOT  = ../..

TPC_LIB_DIR = $(HOME)/Zedboard/libs

# 
OCLFLAGS =
OPTFLAGS =  -g $(EXTRA_FLAGS) 
LIBS =  $(TPC_LIB_DIR)/libtpc.so $(TPC_LIB_DIR)/libplatform.so -lpthread  	
INCLUDES = -I$(FF_ROOT) 

TARGETS = test_upi test_upi_farm test_upi_farm2 test_upi_async test_demo1

all: $(TARGETS)

test_upi:test_upi.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OCLFLAGS) $(OPTFLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

test_upi_farm:test_upi_farm.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OCLFLAGS) $(OPTFLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

test_upi_farm2:test_upi_farm2.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OCLFLAGS) $(OPTFLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

test_upi_async:test_upi_async.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OCLFLAGS) $(OPTFLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

test_demo1:test_demo1.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OCLFLAGS) $(OPTFLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

clean:
	rm -f $(TARGETS)

cleanall: clean
	rm -f *~ 
