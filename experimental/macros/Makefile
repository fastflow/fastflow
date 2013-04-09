FF_HOME              = $(HOME)/FastFlow/fastflow
#ZMQ_HOME             = /usr/local/zmq
ZMQ_HOME             = /usr/local
CXX 		     = g++ 
OPTIMIZE_FLAGS       = -O3 -finline-functions #-DTRACE_FASTFLOW
CXXFLAGS             = -DNO_CMAKE_CONFIG -Wall
CFLAGS               =
LDFLAGS              = -L $(ZMQ_HOME)/lib
INCLUDES             = -I . -I$(ZMQ_HOME)/include -I $(FF_HOME)
LIBS                 = -lpthread -lzmq

.PHONY: all clean cleanall distclean install uninstall
.SUFFIXES: .c .cpp .o

TARGET= bench map map1 farm farm_matmul map_matmul

all: $(TARGET)

bench: bench.cpp
	$(CXX) -DDNODE_FARM $(INCLUDES) $(CXXFLAGS) $(OPTIMIZE_FLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

map: map.cpp
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(OPTIMIZE_FLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

map1: map1.cpp
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(OPTIMIZE_FLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

farm: farm.cpp
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(OPTIMIZE_FLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

farm_matmul: farm_matmul.cpp
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(OPTIMIZE_FLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

map_matmul: map_matmul.cpp
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(OPTIMIZE_FLAGS) -o $@ $< $(LDFLAGS) $(LIBS)


clean: 
	-rm -fr *.o *~
cleanall: clean
	-rm -fr $(TARGET)
