# Make file for parallel BZIP2
SHELL = /bin/sh

FF_HOME = ../..

# Compiler to use
CC = g++
CFLAGS = -DNO_CMAKE_CONFIG -O2 -finline-functions

# Comment out CFLAGS line below for compatability mode for 32bit file sizes
# (less than 2GB) and systems that have compilers that treat int as 64bit
# natively (ie: modern AIX)
CFLAGS += -D_LARGEFILE64_SOURCE -D_FILE_OFFSET_BITS=64 

# Uncomment CFLAGS line below if you want to compile pbzip2 without load
# average support for systems that do not support it
#CFLAGS += -DPBZIP_NO_LOADAVG

# Where you want pbzip2 installed when you do 'make install'
PREFIX = /usr

all: pbzip2 pbzip2_ff

# FastFlow version
pbzip2_ff: pbzip2_ff.cpp
	$(CC) $(CFLAGS) -I$(FF_HOME) $< -o $@ -pthread -lpthread -lbz2

# FastFlow version
streamzip: streamzip.cpp
	$(CC) $(CFLAGS) -I$(FF_HOME) $< -o $@ -pthread -lpthread -lbz2

# Standard pbzip2 compile
pbzip2: pbzip2.cpp
	$(CC) $(CFLAGS) $^ -o pbzip2 -pthread -lpthread -lbz2

# Choose this if you want to compile in a static version of the libbz2 library
pbzip2-static: pbzip2.cpp libbz2.a
	$(CC) $(CFLAGS) $^ -o pbzip2 -pthread -lpthread -I. -L. -lbz2

# Install the binary pbzip2 program and man page
install: pbzip2
	if ( test ! -d $(PREFIX)/bin ) ; then mkdir -p $(PREFIX)/bin ; fi
	if ( test ! -d $(PREFIX)/man ) ; then mkdir -p $(PREFIX)/man ; fi
	if ( test ! -d $(PREFIX)/man/man1 ) ; then mkdir -p $(PREFIX)/man/man1 ; fi
	cp -f pbzip2 $(PREFIX)/bin/pbzip2
	chmod a+x $(PREFIX)/bin/pbzip2
	ln -s -f $(PREFIX)/bin/pbzip2 $(PREFIX)/bin/pbunzip2
	ln -s -f $(PREFIX)/bin/pbzip2 $(PREFIX)/bin/pbzcat
	cp -f pbzip2.1 $(PREFIX)/man/man1
	chmod a+r $(PREFIX)/man/man1/pbzip2.1

clean:
	rm -f *.o pbzip2 pbzip2_ff streamzip

cleanall: clean