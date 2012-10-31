FF_ROOT              = ../..
CC                   = gcc 
CXX                  = g++ 
LINK_OPT             = 
VERSION              = 
OPTIMIZE_FLAGS       = -O3 -finline-functions
CXXFLAGS             = -DNO_CMAKE_CONFIG -Wall
CFLAGS               =
LDFLAGS              = 
INCS                 = -I../../
LIBS                 = -lpthread

INCLUDES             = -I. -I./include $(INCS)
TARGET               = cholesky blockCholesky


.PHONY: all clean cleanall distclean install uninstall
.SUFFIXES: .c .cpp .o

%.d: %.cpp
	set -e; $(CXX) -MM $(INCLUDES) $(CXXFLAGS) $< \
		| sed 's/\($*\)\.o[ :]*/\1.o $@ : /g' > $@; \
		[ -s $@ ] || rm -f $@
%.d: %.c
	set -e; $(CC) -MM $(INCLUDES) $(CFLAGS)  $< \
		| sed 's/\($*\)\.o[ :]*/\1.o $@ : /g' > $@; \
		[ -s $@ ] || rm -f $@
%.o: %.cpp
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(OPTIMIZE_FLAGS) -c -o $@ $<
%.o: %.c
	$(CC) $(INCLUDES) $(CFLAGS) -c -o $@ $<
%: %.o
	$(CXX) $< -o $@ $(LIBS)


all: $(TARGET)

cholesky:cholesky.o
blockCholesky:blockCholesky.o

clean: 
	-rm -fr *.o *~ cholesky blockCholesky
cleanall: clean
	-rm -fr $(TARGET) *.d 

include $(OBJS:.o=.d)
