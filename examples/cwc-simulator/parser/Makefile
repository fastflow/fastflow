LEX=/usr/bin/flex
YACC=bison #/usr/bin/bison
#SED=/usr/bin/sed

all: parser.cpp scanner.cpp

# sed: avoid warning on position.hh@line 136
parser.cpp: parser.yy
	$(YACC) -o parser.cpp --defines=parser.h parser.yy

# pedantic works on GNU if you uncomment the isatty (int ) throw(); patch below
scanner.cpp: scanner.ll
	$(LEX) -o scanner.cpp scanner.ll

clean:
	-rm -f parser.cpp parser.h position.hh location.hh stack.hh scanner.cpp

#@$(SED) -i~ 's,pos1.filename && pos2.filename && \*pos1.filename == \*pos2.filename,(pos1.filename \&\& pos2.filename \&\& *pos1.filename == *pos2.filename),' position.hh
#@$(SED) -i~ 's,extern "C" int isatty (int );,extern "C" int isatty (int ) throw();,' $@
