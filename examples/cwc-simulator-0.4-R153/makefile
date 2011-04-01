SOURCES:=$(filter-out ode.cpp, $(wildcard *.cpp))

#Hybrid semantics
ifeq ($(HYBRID), yes)
HYBRIDFLAG=-DHYBRID
HLDFLAGS=-lgsl -lgslcblas
SOURCES:=$(wildcard *.cpp)
endif

#parallelism
FFFLAGS_BASE=-DUSE_FF_ACCEL -DFF_ALLOCATOR #-DTRACE_FASTFLOW
#ifeq ($(PARALLEL), lockfree)
FFFLAGS = $(FFFLAGS_BASE) -DLOCKFREE
EXECUTABLE_DEPENDENCIES_FF=ff_accel.hpp statistics.h
#endif
ifeq ($(PARALLEL), locking)
FFFLAGS=$(FFFLAGS_BASE)
endif
ifeq ($(PARALLEL), no)
FFLAGS=
EXECUTABLE_DEPENDENCIES_FF=
endif

#SIMD
ifeq ($(SIMD), enabled)
SIMDFLAG = -D SIMD -msse #-msse4
endif

#debug
ifeq ($(DEBUG), yes)
GFLAG=-g
endif
ifeq ($(DEBUG_SIM), yes)
DSFLAG=-DDEBUG_SIMULATION
endif
DBGFLAG=$(GFLAG) $(DSFLAG)

#dependencies for the executable
EXECUTABLE_DEPENDENCIES = scwc.cpp Driver.h Gillespie.h MatchSet.h ode.h progress.h output.h Model.h $(EXECUTABLE_DEPENDENCIES_FF)

CFLAGS= -Wall -O3 $(DBGFLAG) $(FFFLAGS) $(SIMDFLAG) $(HYBRIDFLAG)

LDFLAGS=-lboost_program_options-mt -L/opt/local/lib/ -lpthread $(HLDFLAGS)
CC=/usr/bin/g++
BOOST=/opt/local/include/
OBJECTS:=$(SOURCES:.cpp=.o)
EXECUTABLE=scwc

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) -I$(BOOST) -I../.. -I./include -c $< -o $@

clean:
	-rm -f $(OBJECTS) $(EXECUTABLE)

############################################################## Dependencies ##############################################################

Species.o: Species.cpp Species.h definitions.h
PSpecies.o: PSpecies.cpp PSpecies.h Species.h
Compartment.o: Compartment.cpp Compartment.h Species.cpp Species.h
Driver.o: Driver.cpp Driver.h Model.h
Gillespie.o: Gillespie.cpp Gillespie.h Compartment.h Instantiation.h Match.h MatchSet.h OCompartment.h PCompartment.h Rule.h SubMatch.h SubMatchSet.h
Instantiation.h: Species.h Compartment.h
Match.h: SubMatchSet.h
MatchSet.o: MatchSet.cpp MatchSet.h Match.h Model.h Rule.h ode.h
Model.o: Model.cpp Model.h Compartment.h Monitor.h Rule.h
Monitor.o: Monitor.cpp Monitor.h Compartment.h
OCompartment.o: OCompartment.cpp OCompartment.h Compartment.h Instantiation.h
parser.o: Species.h Compartment.h Driver.h Model.h Monitor.h OCompartment.h PCompartment.h Rule.h scanner.h
PCompartment.o: PCompartment.cpp PCompartment.h Compartment.h PSpecies.h SubMatchSet.h
Rule.o: Rule.cpp Rule.h Compartment.h OCompartment.h PCompartment.h
scanner.o: scanner.h
scwc.o: $(EXECUTABLE_DEPENDENCIES)
statistics.o: statistics.cpp statistics.h
SubMatch.h: Compartment.h
SubMatchSet.o: SubMatchSet.cpp SubMatchSet.h Compartment.h MatchSet.h Rule.h SubMatch.h

##########################################################################################################################################
