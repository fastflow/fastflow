/** \file debug.c
 *
 * Routines for error reporting.
 */
/*
 * Copyright (c) 2007-2008 ETH Zürich, Institute of Computational Science
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */


#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>

/** <code>printf</code>-like function for critical error reporting.
 *
 * Reports a message to <code>stderr</code> and terminates the program.
 * \param str A format string.
 */
void error(const char* str, ...) {
	va_list ap;
	fprintf(stderr,"error: ");
	va_start(ap, str);
	vfprintf(stderr, str, ap);
	va_end(ap);
	fprintf(stderr,"\n");
	fflush(stderr);
	exit(1);
}

/** <code>printf</code>-like function for noncritical error reporting.
 *
 * Reports a message to <code>stderr</code> without terminating the program.
 * \param str A format string.
 */
void warning(const char* str, ...) {
	va_list ap;
	fprintf(stderr,"warning: ");
	va_start(ap, str);
	vfprintf(stderr, str, ap);
	va_end(ap);
	fprintf(stderr,"\n");
	fflush(stderr);
}
