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
CXXFLAGS             = -DNO_CMAKE_CONFIG -Wall
CFLAGS               = $(DIS) 
LDFLAGS              = $(DIS) -L/usr/X11R6/lib -L.
INCS                 = -I../../
LIBS                 = -lpthread -lmarX2 -lX11 -lm
INCLUDES             = -I. $(INCS) -I/usr/X11R6/include
TARGET               = mandel_ff mandel_seq mandel_ff_mem_all mandel_pt
SOURCES		     = mandel_ff.cpp mandel_seq.c mandel_ff_mem_all.cpp mandel_pt.cpp marX2.c
SRC = .
INC = .

.PHONY: all clean cleanall

all:$(TARGET)

libmarX2.a: $(SRC)/marX2.c $(INC)/marX2.h
	gcc -c $(OPTIMIZE_FLAGS) $(CFLAGS) -I/usr/X11R6/include $(INCLUDES) $(SRC)/marX2.c 
	ar -rv libmarX2.a marX2.o
	ranlib libmarX2.a

mandel_seq: mandel_seq.c libmarX2.a
	$(CC)  $(CFLAGS) $(OPTIMIZE_FLAGS) $(CFLAGS)  $(INCLUDES) $(LDFLAGS)  -o mandel_seq mandel_seq.c $(LIBS)

mandel_ff: mandel_ff.cpp  libmarX2.a
	$(CXX)  $(CXXFLAGS) $(OPTIMIZE_FLAGS)  $(CFLAGS) $(INCLUDES) $(LDFLAGS)  -o mandel_ff mandel_ff.cpp $(LIBS)

mandel_ff_mem_all: 	mandel_ff_mem_all.cpp libmarX2.a
	$(CXX)  $(CXXFLAGS) $(OPTIMIZE_FLAGS)  $(CFLAGS) $(INCLUDES) $(LDFLAGS)  -o mandel_ff_mem_all mandel_ff_mem_all.cpp $(LIBS)

mandel_pt: mandel_pt.cpp libmarX2.a
	$(CXX)  $(CXXFLAGS) $(OPTIMIZE_FLAGS)  $(CFLAGS) $(INCLUDES) $(LDFLAGS)  -o mandel_pt mandel_pt.cpp $(LIBS)

clean:
	rm -f libmarX2.a *.o

cleanall: clean
	rm -fr $(TARGET) *~ *.dSYM

depend:
	makedepend $(SOURCES)


# DO NOT DELETE THIS LINE -- make depend depends on it.

mandel_seq.o: /usr/include/stdio.h /usr/include/_types.h
mandel_seq.o: /usr/include/sys/_types.h /usr/include/sys/cdefs.h
mandel_seq.o: /usr/include/machine/_types.h /usr/include/i386/_types.h
mandel_seq.o: marX2.h /usr/include/X11/Xlib.h /usr/include/sys/types.h
mandel_seq.o: /usr/include/sys/appleapiopts.h /usr/include/machine/types.h
mandel_seq.o: /usr/include/i386/types.h /usr/include/machine/endian.h
mandel_seq.o: /usr/include/i386/endian.h /usr/include/sys/_endian.h
mandel_seq.o: /usr/include/libkern/_OSByteOrder.h
mandel_seq.o: /usr/include/libkern/i386/_OSByteOrder.h
mandel_seq.o: /usr/include/sys/_structs.h /usr/include/X11/X.h
mandel_seq.o: /usr/include/X11/Xfuncproto.h /usr/include/X11/Xosdefs.h
mandel_seq.o: /usr/include/stddef.h /usr/include/X11/Xutil.h
mandel_seq.o: /usr/include/X11/keysym.h /usr/include/X11/keysymdef.h
mandel_seq.o: /usr/include/stdlib.h /usr/include/available.h
mandel_seq.o: /usr/include/sys/wait.h /usr/include/sys/signal.h
mandel_seq.o: /usr/include/machine/signal.h /usr/include/i386/signal.h
mandel_seq.o: /usr/include/i386/_structs.h /usr/include/sys/resource.h
mandel_seq.o: /usr/include/alloca.h /usr/include/string.h /usr/include/math.h
mandel_seq.o: /usr/include/architecture/i386/math.h /usr/include/sys/time.h
mandel_seq.o: /usr/include/time.h /usr/include/_structs.h
mandel_seq.o: /usr/include/sys/_select.h
marX2.o: marX2.h /usr/include/X11/Xlib.h /usr/include/sys/types.h
marX2.o: /usr/include/sys/appleapiopts.h /usr/include/sys/cdefs.h
marX2.o: /usr/include/machine/types.h /usr/include/i386/types.h
marX2.o: /usr/include/i386/_types.h /usr/include/sys/_types.h
marX2.o: /usr/include/machine/_types.h /usr/include/machine/endian.h
marX2.o: /usr/include/i386/endian.h /usr/include/sys/_endian.h
marX2.o: /usr/include/libkern/_OSByteOrder.h
marX2.o: /usr/include/libkern/i386/_OSByteOrder.h /usr/include/sys/_structs.h
marX2.o: /usr/include/X11/X.h /usr/include/X11/Xfuncproto.h
marX2.o: /usr/include/X11/Xosdefs.h /usr/include/stddef.h
marX2.o: /usr/include/_types.h /usr/include/X11/Xutil.h
marX2.o: /usr/include/X11/keysym.h /usr/include/X11/keysymdef.h
marX2.o: /usr/include/stdio.h /usr/include/stdlib.h /usr/include/available.h
marX2.o: /usr/include/sys/wait.h /usr/include/sys/signal.h
marX2.o: /usr/include/machine/signal.h /usr/include/i386/signal.h
marX2.o: /usr/include/i386/_structs.h /usr/include/sys/resource.h
marX2.o: /usr/include/alloca.h /usr/include/string.h /usr/include/math.h
marX2.o: /usr/include/architecture/i386/math.h
