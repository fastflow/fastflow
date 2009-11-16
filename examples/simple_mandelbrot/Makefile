# Makefile
#

# Define MANDEL_NO_DISPLAY if you do not want to display image
# ex: export MANDEL_NO_DISPLAY=1  or
#     setenv MANDEL_NO_DISPLAY 1
#
ifdef MANDEL_NO_DISPLAY
DIS=-DNO_DISPLAY
else
DIS=
endif

CC                   = gcc 
CXX                  = g++ 
LINK_OPT             = 
VERSION              = 
OPTIMIZE_FLAGS       = -O3 -finline-functions
CXXFLAGS             = -Wall
CFLAGS               = $(DIS) 
LDFLAGS              = $(DIS) -L/usr/X11R6/lib -L.
INCS                 = -I../../ -I../../atomic 
LIBS                 = -lpthread -lmarX2 -lX11 -lm
INCLUDES             = -I. $(INCS) -I/usr/X11R6/include
TARGET               = mandel_ff mandel_seq mandel_ff_mem_all mandel_pt

SRC = .
INC = .

.PHONY: all clean cleanall

all:$(TARGET)

marX.a: $(SRC)/marX2.c $(INC)/marX2.h
	gcc -c $(OPTIMIZE_FLAGS) $(CFLAGS) -I/usr/X11R6/include $(INCLUDES) $(SRC)/marX2.c 
	ar -rv libmarX2.a marX2.o
	ranlib libmarX2.a

mandel_seq: marX.a
	$(CC)  $(CFLAGS) $(OPTIMIZE_FLAGS) $(CFLAGS)  $(INCLUDES) $(LDFLAGS)  -o mandel_seq mandel_seq.c $(LIBS)

mandel_ff: marX.a
	$(CXX)  $(CXXFLAGS) $(OPTIMIZE_FLAGS)  $(CFLAGS) $(INCLUDES) $(LDFLAGS)  -o mandel_ff mandel_ff.cpp $(LIBS)

mandel_ff_mem_all: marX.a
	$(CXX)  $(CXXFLAGS) $(OPTIMIZE_FLAGS)  $(CFLAGS) $(INCLUDES) $(LDFLAGS)  -o mandel_ff_mem_all mandel_ff_mem_all.cpp $(LIBS)

mandel_pt: marX.a
	$(CXX)  $(CXXFLAGS) $(OPTIMIZE_FLAGS)  $(CFLAGS) $(INCLUDES) $(LDFLAGS)  -o mandel_pt mandel_pt.cpp $(LIBS)

clean:
	rm -f libmarX2.a *.o

cleanall: clean
	rm -fr $(TARGET) *~ *.dSYM
