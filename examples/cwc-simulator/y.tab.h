/** \file y.tab.h Forwarding include file to parser.h (actually by including scanner.h) */

/* When using automake the bison parser file "xyz.yy" is processed by the
 * ylwrap script. It calls bison in a separate directory, which outputs source
 * to the default names "y.tab.c" and "y.tab.h". The ylwrap script then renames
 * these files into "xyz.cc" and "xyz.h" and tries to update include references
 * using sed. However this does not work for the C++ parser skeleton, so the
 * source file "xyz.cc" still refers to the default "y.tab.h". The easiest
 * work-around is to use this forwarding include file. */

#include "scanner.h"

