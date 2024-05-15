EXEC ?= vadd_final
SRC := $(EXEC).cpp
OBJ := $(SRC:.cpp=.o)


FASTFLOW = $(HOME)/fastflow
FFPGA = $(HOME)/FFpga/ff_fpga


CC = g++
CFLAGS = -std=c++17 -Wall -Wextra -O0
LDFLAGS = -L$(XILINX_XRT)/lib -L$(XILINX_XRT)/liblnx64.o

INCS = -I$(XILINX_VIVADO)/include -I$(XILINX_XRT)/include/
INCS += -I$(FASTFLOW)/include -I$(FFPGA)
INCS += -I .
INCS += -I ../ff_fpga

LIBS = -lOpenCL -pthread -lrt -lstdc++


all: $(EXEC)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@ $(INCS)

$(EXEC): $(OBJ)
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@ $(LIBS) $(INCS)


.PHONY: clean
clean:
	rm -f $(OBJ) $(EXEC)


.PHONY: debug
debug: CFLAGS += -DDEBUG -g3
debug: clean all

.PHONY: release
release: CFLAGS = -std=c++17 -O3
release: clean all

.PHONY: all debug clean
