# ---------------------------------------------------------------------------
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License version 2 as 
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
#
#  As a special exception, you may use this file as part of a free software
#  library without restriction.  Specifically, if other files instantiate
#  templates or use macros or inline functions from this file, or you compile
#  this file and link it with other files to produce an executable, this
#  file does not by itself cause the resulting executable to be covered by
#  the GNU General Public License.  This exception does not however
#  invalidate any other reasons why the executable file might be covered by
#  the GNU General Public License.
#
# ---------------------------------------------------------------------------


CXX	= g++ -std=c++11 
FFHOME 	= ../..
CFLAGS 	= -Wall `pkg-config --cflags opencv` #-DTRACE_FASTFLOW  -pg
OPT     = -O3 -DNDEBUG  # -g
LIBS 	= `pkg-config --libs opencv` -pthread

TARGETS	= video ffvideo ffvideo_nf ffvideo_nf2 ffvideo_nf3

all: $(TARGETS)

video:	video.cpp
	$(CXX) $(OPT) -I$(FFHOME) $(CFLAGS) -o $@ $< $(LIBS)

ffvideo: ffvideo.cpp
	$(CXX) -DNO_DEFAULT_MAPPING $(OPT) -I$(FFHOME) $(CFLAGS) -o $@ $< $(LIBS)  

ffvideo_nf: ffvideo_nf.cpp
	$(CXX) -DNO_DEFAULT_MAPPING $(OPT) -I$(FFHOME) $(CFLAGS) -o $@ $< $(LIBS)  

ffvideo_nf2: ffvideo_nf2.cpp
	$(CXX) -DNO_DEFAULT_MAPPING $(OPT) -I$(FFHOME) $(CFLAGS) -o $@ $< $(LIBS)  
ffvideo_nf3: ffvideo_nf3.cpp
	$(CXX) -DNO_DEFAULT_MAPPING $(OPT) -I$(FFHOME) $(CFLAGS) -o $@ $< $(LIBS)  

ffvideo_nf3: ffvideo_nf3.cpp
	$(CXX) -DNO_DEFAULT_MAPPING $(OPT) -I$(FFHOME) $(CFLAGS) -o $@ $< $(LIBS)  

clean: 
	rm -f $(TARGETS)

cleanall: clean
	rm -f *.o *.a *~ 

