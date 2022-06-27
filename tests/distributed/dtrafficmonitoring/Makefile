
CXXFLAGS            += -std=c++20
ifdef DEBUG
    OPTIMIZE_FLAGS  += -g -fno-inline-functions
else
    OPTIMIZE_FLAGS  += -O3 -finline-functions -DNDEBUG
endif
ifdef DISABLE_FF_DISTRIBUTED
    CXXFLAGS += -DDISABLE_FF_DISTRIBUTED
endif
ifdef EXCLUDE_BLOCKING
    CXXFLAGS        += -DDFF_EXCLUDE_BLOCKING
endif
ifdef PRINT_IR
    CXXFLAGS += -DPRINT_IR
endif
ifdef EXCLUDE_MPI
    CXXFLAGS	    += -DDFF_EXCLUDE_MPI
else
ifdef MPI_HOME
    INCS            += `pkg-config --cflags-only-I $(MPI_HOME)/lib/pkgconfig/ompi-cxx.pc`
    LIBS            += `pkg-config --libs $(MPI_HOME)/lib/pkgconfig/ompi-cxx.pc`
else
    CXXFLAGS	    += -DDFF_EXCLUDE_MPI
endif
endif
ifdef FF_HOME
    INCS            += -I$(FF_HOME)
else
    INCS            += -I ~/fastflow
endif
ifdef CEREAL_HOME
    INCS            += -I$(CEREAL_HOME)
else
    INCS            += -I ~/cereal
endif

#gdal include
INCS += -I /usr/local/include

CXXFLAGS            += -Wall
LIBS                += -pthread -L /usr/local/Cellar/gdal/3.5.0_1/lib/ -lgdal
INCLUDES             = $(INCS)

SOURCES              = $(wildcard *.cpp)
TARGET               = $(SOURCES:.cpp=)

.PHONY: all clean cleanall 
.SUFFIXES: .c .cpp .o

%.d: %.cpp
	set -e; $(CXX) -MM $(INCLUDES) $(CXXFLAGS) $< \
		| sed 's/\($*\)\.o[ :]*/\1.o $@ : /g' > $@; \
		[ -s $@ ] || rm -f $@
%.d: %.c
	set -e; $(CC) -MM $(INCLUDES) $(CFLAGS)  $< \
		| sed 's/\($*\)\.o[ :]*/\1.o $@ : /g' > $@; \
		[ -s $@ ] || rm -f $@
%.o: %.c
	$(CC) $(INCLUDES) $(CFLAGS) -c -o $@ $<
%: %.cpp
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(OPTIMIZE_FLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

all: $(TARGET)

clean: 
	-rm -fr *.o *~
cleanall: clean
	-rm -fr $(TARGET) *.d ./socket*

include $(OBJS:.o=.d)
