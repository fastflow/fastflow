CXX      = arm-xilinx-linux-gnueabi-g++
CXXFLAGS = -std=c++11 -DCHECK -DFF_TPC
FF_ROOT  = ../..

TPC_ROOT = $(HOME)/Zedboard/UPI

# 
OCLFLAGS =
OPTFLAGS =  -g $(EXTRA_FLAGS) 
LIBS =  $(TPC_ROOT)/libtpc.so $(TPC_ROOT)/libplatform.so -lpthread  	
INCLUDES = -I$(FF_ROOT) 

TARGETS = test_upi test_upi_farm test_upi_farm2

all: $(TARGETS)

test_upi:test_upi.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OCLFLAGS) $(OPTFLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

test_upi_farm:test_upi_farm.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OCLFLAGS) $(OPTFLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

test_upi_farm2:test_upi_farm2.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OCLFLAGS) $(OPTFLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

clean:
	rm -f $(TARGETS)

cleanall: clean
	rm -f *~ 
