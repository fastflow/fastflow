%{ /*** C/C++ Declarations ***/

#include <string>
#include <vector>
#include <cstdio>

#include "scanner.h"

/* replacement of strtoll for windows platforms */
#if (defined(_MSC_VER) || defined(__INTEL_COMPILER)) && defined(_WIN32)
#define strtoll _strtoi64
#endif

/* import the parser's token type into a local typedef */
typedef scwc::Parser::token token;
typedef scwc::Parser::token_type token_type;

/* By default yylex returns int, we use token_type. Unfortunately yyterminate
 * by default returns 0, which is not of token_type. */
#define yyterminate() return token::END

/* This disables inclusion of unistd.h, which is not available under Visual C++
 * on Win32. The C++ scanner uses STL streams instead. */
#define YY_NO_UNISTD_H

%}

/*** Flex Declarations and Options ***/

/* enable c++ scanner class generation */
%option c++

/* change the name of the scanner class. results in "SCWCFlexLexer" */
%option prefix="SCWC"

/* the manual says "somewhat more optimized" */
%option batch

/* enable scanner to generate debug output. disable this for release
 * versions. */
%option debug

/* no support for include files is planned */
%option yywrap nounput

/* enables the use of start condition stacks */
%option stack

/* The following paragraph suffices to track locations accurately. Each time
 * yylex is invoked, the begin position is moved onto the end position. */
%{
#define YY_USER_ACTION  yylloc->columns(yyleng);
%}

%% /*** Regular Expressions Part ***/

 /* code to place at the beginning of yylex() */
%{
    // reset location
    yylloc->step();
%}

 /*** Change the lexer rules below ***/

[0-9]+ {
    yylval->multiplicity = (unsigned long long)strtoll(yytext, NULL, 0);
    return token::INTEGER;
}

[0-9]+"."[0-9]* {
    yylval->constantRate = atof(yytext);
    return token::DOUBLE;
}

[A-Za-z][A-Za-z0-9_.]* {
    yylval->atomLabel = new std::string(yytext, yyleng);
    return token::ATOM_LABEL;
}

~[A-Z][A-Za-z0-9_]* {
    yylval->termVariable = new std::string(yytext, yyleng);
    return token::TVAR;
}

~[a-z][A-Za-z0-9_]* {
    yylval->wrapVariable = new std::string(yytext, yyleng);
    return token::WVAR;
}

">>>" return token::RSEP;

"%%" return token::LSEP;

"%model" return token::MODEL_;

"%alphabet" return token::ALPHABET_;

"%rules" return token::RULES_;

"%term" return token::TERM_;

"%monitors" return token::MONITORS_;

"%F" return token::FUNCTION_;



\"[^\"\n]*\" {
    yylval->qstring = new std::string(yytext, yyleng);
    return token::QSTRING;
}

 /* gobble up white-spaces */
[ \t\r]+ {
    yylloc->step();
}

 /* gobble up end-of-lines */
\n {
    yylloc->lines(yyleng); yylloc->step();
    /* return token::EOL; */
}

 /* gobble up single-line comments */
\/\/[^\n]* {}

 /* pass all other characters up to bison */
. {
    return static_cast<token_type>(*yytext);
}

 /*** Change the lexer rules above ***/

%% /*** Additional Code ***/

namespace scwc {

Scanner::Scanner(std::istream* in,
                 std::ostream* out)
    : SCWCFlexLexer(in, out)
{
}

Scanner::~Scanner()
{
}

void Scanner::set_debug(bool b)
{
    yy_flex_debug = b;
}

}

/* This implementation of SCWCFlexLexer::yylex() is required to fill the
 * vtable of the class SCWCFlexLexer. We define the scanner's main yylex
 * function via YY_DECL to reside in the Scanner class instead. */

#ifdef yylex
#undef yylex
#endif

int SCWCFlexLexer::yylex()
{
    std::cerr << "in SCWCFlexLexer::yylex() !" << std::endl;
    return 0;
}

/* When the scanner receives an end-of-file indication from YY_INPUT, it then
 * checks the yywrap() function. If yywrap() returns false (zero), then it is
 * assumed that the function has gone ahead and set up `yyin' to point to
 * another input file, and scanning continues. If it returns true (non-zero),
 * then the scanner terminates, returning 0 to its caller. */

int SCWCFlexLexer::yywrap()
{
    return 1;
}
