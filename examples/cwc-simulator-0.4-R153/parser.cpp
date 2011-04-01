/* A Bison parser, made by GNU Bison 2.4.3.  */

/* Skeleton implementation for Bison LALR(1) parsers in C++
   
      Copyright (C) 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010 Free
   Software Foundation, Inc.
   
   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
   
   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.
   
   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.
   
   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

// Take the name prefix into account.
#define yylex   scwclex

/* First part of user declarations.  */

/* Line 311 of lalr1.cc  */
#line 1 "parser.yy"
/*** C/C++ Declarations ***/

#include <stdio.h>
#include <string>
#include <vector>
#include <sstream>
#include <iostream>
//#include <stdint.h>

#include "definitions.h"
#include "Species.h"
#include "Compartment.h"
#include "OCompartment.h"
#include "PCompartment.h"
#include "Model.h"
#include "Monitor.h"
#include "Rule.h"
  using namespace std;

  typedef map<string, symbol_adress> symbol_table;

  symbol_table atom_table;
  symbol_table type_table;
  symbol_adress last_type_free;

  void print_symbol_table() {
    for(symbol_table::iterator it = atom_table.begin(); it != atom_table.end(); it++) {
      cout << it->first << ": " << it->second << endl;
    }
  }

  unsigned int ode_offset = 0;

  

/* Line 311 of lalr1.cc  */
#line 78 "parser.cpp"


#include "parser.h"

/* User implementation prologue.  */

/* Line 317 of lalr1.cc  */
#line 149 "parser.yy"


#include "Driver.h"
#include "scanner.h"

  /* this "connects" the bison parser in the driver to the flex scanner class
   * object. it defines the yylex() function call to pull the next token from the
   * current lexer object of the driver context. */
#undef yylex
#define yylex driver.lexer->lex

  

/* Line 317 of lalr1.cc  */
#line 101 "parser.cpp"

#ifndef YY_
# if defined YYENABLE_NLS && YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* FIXME: INFRINGES ON USER NAME SPACE */
#   define YY_(msgid) dgettext ("bison-runtime", msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(msgid) msgid
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#define YYUSE(e) ((void) (e))

/* Enable debugging if requested.  */
#if YYDEBUG

/* A pseudo ostream that takes yydebug_ into account.  */
# define YYCDEBUG if (yydebug_) (*yycdebug_)

# define YY_SYMBOL_PRINT(Title, Type, Value, Location)	\
do {							\
  if (yydebug_)						\
    {							\
      *yycdebug_ << Title << ' ';			\
      yy_symbol_print_ ((Type), (Value), (Location));	\
      *yycdebug_ << std::endl;				\
    }							\
} while (false)

# define YY_REDUCE_PRINT(Rule)		\
do {					\
  if (yydebug_)				\
    yy_reduce_print_ (Rule);		\
} while (false)

# define YY_STACK_PRINT()		\
do {					\
  if (yydebug_)				\
    yystack_print_ ();			\
} while (false)

#else /* !YYDEBUG */

# define YYCDEBUG if (false) std::cerr
# define YY_SYMBOL_PRINT(Title, Type, Value, Location)
# define YY_REDUCE_PRINT(Rule)
# define YY_STACK_PRINT()

#endif /* !YYDEBUG */

#define yyerrok		(yyerrstatus_ = 0)
#define yyclearin	(yychar = yyempty_)

#define YYACCEPT	goto yyacceptlab
#define YYABORT		goto yyabortlab
#define YYERROR		goto yyerrorlab
#define YYRECOVERING()  (!!yyerrstatus_)


namespace scwc {

/* Line 380 of lalr1.cc  */
#line 167 "parser.cpp"
#if YYERROR_VERBOSE

  /* Return YYSTR after stripping away unnecessary quotes and
     backslashes, so that it's suitable for yyerror.  The heuristic is
     that double-quoting is unnecessary unless the string contains an
     apostrophe, a comma, or backslash (other than backslash-backslash).
     YYSTR is taken from yytname.  */
  std::string
  Parser::yytnamerr_ (const char *yystr)
  {
    if (*yystr == '"')
      {
        std::string yyr = "";
        char const *yyp = yystr;

        for (;;)
          switch (*++yyp)
            {
            case '\'':
            case ',':
              goto do_not_strip_quotes;

            case '\\':
              if (*++yyp != '\\')
                goto do_not_strip_quotes;
              /* Fall through.  */
            default:
              yyr += *yyp;
              break;

            case '"':
              return yyr;
            }
      do_not_strip_quotes: ;
      }

    return yystr;
  }

#endif

  /// Build a parser object.
  Parser::Parser (class Driver& driver_yyarg)
    :
#if YYDEBUG
      yydebug_ (false),
      yycdebug_ (&std::cerr),
#endif
      driver (driver_yyarg)
  {
  }

  Parser::~Parser ()
  {
  }

#if YYDEBUG
  /*--------------------------------.
  | Print this symbol on YYOUTPUT.  |
  `--------------------------------*/

  inline void
  Parser::yy_symbol_value_print_ (int yytype,
			   const semantic_type* yyvaluep, const location_type* yylocationp)
  {
    YYUSE (yylocationp);
    YYUSE (yyvaluep);
    switch (yytype)
      {
         default:
	  break;
      }
  }


  void
  Parser::yy_symbol_print_ (int yytype,
			   const semantic_type* yyvaluep, const location_type* yylocationp)
  {
    *yycdebug_ << (yytype < yyntokens_ ? "token" : "nterm")
	       << ' ' << yytname_[yytype] << " ("
	       << *yylocationp << ": ";
    yy_symbol_value_print_ (yytype, yyvaluep, yylocationp);
    *yycdebug_ << ')';
  }
#endif

  void
  Parser::yydestruct_ (const char* yymsg,
			   int yytype, semantic_type* yyvaluep, location_type* yylocationp)
  {
    YYUSE (yylocationp);
    YYUSE (yymsg);
    YYUSE (yyvaluep);

    YY_SYMBOL_PRINT (yymsg, yytype, yyvaluep, yylocationp);

    switch (yytype)
      {
  
	default:
	  break;
      }
  }

  void
  Parser::yypop_ (unsigned int n)
  {
    yystate_stack_.pop (n);
    yysemantic_stack_.pop (n);
    yylocation_stack_.pop (n);
  }

#if YYDEBUG
  std::ostream&
  Parser::debug_stream () const
  {
    return *yycdebug_;
  }

  void
  Parser::set_debug_stream (std::ostream& o)
  {
    yycdebug_ = &o;
  }


  Parser::debug_level_type
  Parser::debug_level () const
  {
    return yydebug_;
  }

  void
  Parser::set_debug_level (debug_level_type l)
  {
    yydebug_ = l;
  }
#endif

  int
  Parser::parse ()
  {
    /// Lookahead and lookahead in internal form.
    int yychar = yyempty_;
    int yytoken = 0;

    /* State.  */
    int yyn;
    int yylen = 0;
    int yystate = 0;

    /* Error handling.  */
    int yynerrs_ = 0;
    int yyerrstatus_ = 0;

    /// Semantic value of the lookahead.
    semantic_type yylval;
    /// Location of the lookahead.
    location_type yylloc;
    /// The locations where the error started and ended.
    location_type yyerror_range[3];

    /// $$.
    semantic_type yyval;
    /// @$.
    location_type yyloc;

    int yyresult;

    YYCDEBUG << "Starting parse" << std::endl;


    /* User initialization code.  */
    
/* Line 553 of lalr1.cc  */
#line 63 "parser.yy"
{
    // initialize the initial location object
    yylloc.begin.filename = yylloc.end.filename = &driver.streamname;
  }

/* Line 553 of lalr1.cc  */
#line 351 "parser.cpp"

    /* Initialize the stacks.  The initial state will be pushed in
       yynewstate, since the latter expects the semantical and the
       location values to have been already stored, initialize these
       stacks with a primary value.  */
    yystate_stack_ = state_stack_type (0);
    yysemantic_stack_ = semantic_stack_type (0);
    yylocation_stack_ = location_stack_type (0);
    yysemantic_stack_.push (yylval);
    yylocation_stack_.push (yylloc);

    /* New state.  */
  yynewstate:
    yystate_stack_.push (yystate);
    YYCDEBUG << "Entering state " << yystate << std::endl;

    /* Accept?  */
    if (yystate == yyfinal_)
      goto yyacceptlab;

    goto yybackup;

    /* Backup.  */
  yybackup:

    /* Try to take a decision without lookahead.  */
    yyn = yypact_[yystate];
    if (yyn == yypact_ninf_)
      goto yydefault;

    /* Read a lookahead token.  */
    if (yychar == yyempty_)
      {
	YYCDEBUG << "Reading a token: ";
	yychar = yylex (&yylval, &yylloc);
      }


    /* Convert token to internal form.  */
    if (yychar <= yyeof_)
      {
	yychar = yytoken = yyeof_;
	YYCDEBUG << "Now at end of input." << std::endl;
      }
    else
      {
	yytoken = yytranslate_ (yychar);
	YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
      }

    /* If the proper action on seeing token YYTOKEN is to reduce or to
       detect an error, take that action.  */
    yyn += yytoken;
    if (yyn < 0 || yylast_ < yyn || yycheck_[yyn] != yytoken)
      goto yydefault;

    /* Reduce or error.  */
    yyn = yytable_[yyn];
    if (yyn <= 0)
      {
	if (yyn == 0 || yyn == yytable_ninf_)
	goto yyerrlab;
	yyn = -yyn;
	goto yyreduce;
      }

    /* Shift the lookahead token.  */
    YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);

    /* Discard the token being shifted.  */
    yychar = yyempty_;

    yysemantic_stack_.push (yylval);
    yylocation_stack_.push (yylloc);

    /* Count tokens shifted since error; after three, turn off error
       status.  */
    if (yyerrstatus_)
      --yyerrstatus_;

    yystate = yyn;
    goto yynewstate;

  /*-----------------------------------------------------------.
  | yydefault -- do the default action for the current state.  |
  `-----------------------------------------------------------*/
  yydefault:
    yyn = yydefact_[yystate];
    if (yyn == 0)
      goto yyerrlab;
    goto yyreduce;

  /*-----------------------------.
  | yyreduce -- Do a reduction.  |
  `-----------------------------*/
  yyreduce:
    yylen = yyr2_[yyn];
    /* If YYLEN is nonzero, implement the default value of the action:
       `$$ = $1'.  Otherwise, use the top of the stack.

       Otherwise, the following line sets YYVAL to garbage.
       This behavior is undocumented and Bison
       users should not rely upon it.  */
    if (yylen)
      yyval = yysemantic_stack_[yylen - 1];
    else
      yyval = yysemantic_stack_[0];

    {
      slice<location_type, location_stack_type> slice (yylocation_stack_, yylen);
      YYLLOC_DEFAULT (yyloc, slice, yylen);
    }
    YY_REDUCE_PRINT (yyn);
    switch (yyn)
      {
	  case 2:

/* Line 678 of lalr1.cc  */
#line 166 "parser.yy"
    {
  (yysemantic_stack_[(2) - (1)].alphabet)->push_back(*(yysemantic_stack_[(2) - (2)].atomLabel));
  delete (yysemantic_stack_[(2) - (2)].atomLabel);
 }
    break;

  case 3:

/* Line 678 of lalr1.cc  */
#line 170 "parser.yy"
    {
  //empty
  (yyval.alphabet) = new vector<string>;
 }
    break;

  case 4:

/* Line 678 of lalr1.cc  */
#line 175 "parser.yy"
    {
  reverse_symbol_table &rst = *(yysemantic_stack_[(1) - (1)].alphabet);
  sort(rst.begin(), rst.end());
  for(symbol_adress i=0; i<rst.size(); i++) {
    atom_table.insert(pair<string, symbol_adress>(rst[i], i));
  }
  type_table.insert(pair<string, symbol_adress>(string("T"), TOP_LEVEL));
  last_type_free = TOP_LEVEL + 1;
 }
    break;

  case 5:

/* Line 678 of lalr1.cc  */
#line 185 "parser.yy"
    {
  //singleton
  (yyval.atomNode) = new pair<symbol_adress, multiplicityType>(atom_table[*(yysemantic_stack_[(1) - (1)].atomLabel)], 1);
  delete (yysemantic_stack_[(1) - (1)].atomLabel);
 }
    break;

  case 6:

/* Line 678 of lalr1.cc  */
#line 190 "parser.yy"
    {
  //bag
  if((yysemantic_stack_[(3) - (1)].multiplicity) <= 0) {
    error(yyloc, string("Negative multiplicity"));
    YYERROR;
  } else {
    (yyval.atomNode) = new pair<symbol_adress, multiplicityType>(atom_table[*(yysemantic_stack_[(3) - (3)].atomLabel)], (yysemantic_stack_[(3) - (1)].multiplicity));
    delete (yysemantic_stack_[(3) - (3)].atomLabel);
  }
  }
    break;

  case 7:

/* Line 678 of lalr1.cc  */
#line 201 "parser.yy"
    {
  (yysemantic_stack_[(2) - (1)].atomsMultiset)->add((yysemantic_stack_[(2) - (2)].atomNode)->first, (yysemantic_stack_[(2) - (2)].atomNode)->second);
  delete (yysemantic_stack_[(2) - (2)].atomNode);
 }
    break;

  case 8:

/* Line 678 of lalr1.cc  */
#line 205 "parser.yy"
    {
  //empty
  (yyval.atomsMultiset) = new Species(atom_table.size());
 }
    break;

  case 9:

/* Line 678 of lalr1.cc  */
#line 210 "parser.yy"
    {
  //compartment type-label
  map<string, symbol_adress>::iterator it;
  it = type_table.find(*(yysemantic_stack_[(3) - (2)].atomLabel));
  if(it != type_table.end()) {
    (yyval.compartmentType) = it->second;
  }
  else {
    (yyval.compartmentType) = (type_table[*(yysemantic_stack_[(3) - (2)].atomLabel)] = last_type_free++);
  }
  delete (yysemantic_stack_[(3) - (2)].atomLabel);
 }
    break;

  case 10:

/* Line 678 of lalr1.cc  */
#line 227 "parser.yy"
    {
  //ground compartment
  (yyval.compartmentNode) = new Compartment((yysemantic_stack_[(6) - (3)].atomsMultiset), (yysemantic_stack_[(6) - (5)].termsMultiset)->first, (yysemantic_stack_[(6) - (5)].termsMultiset)->second, (yysemantic_stack_[(6) - (2)].compartmentType));
  delete (yysemantic_stack_[(6) - (5)].termsMultiset);
 }
    break;

  case 11:

/* Line 678 of lalr1.cc  */
#line 238 "parser.yy"
    { 
  //species
  (yysemantic_stack_[(2) - (1)].termsMultiset)->first->add((yysemantic_stack_[(2) - (2)].atomNode)->first, (yysemantic_stack_[(2) - (2)].atomNode)->second);
  delete (yysemantic_stack_[(2) - (2)].atomNode);
 }
    break;

  case 12:

/* Line 678 of lalr1.cc  */
#line 243 "parser.yy"
    {
  //compartment
  (yysemantic_stack_[(2) - (1)].termsMultiset)->second->push_back((yysemantic_stack_[(2) - (2)].compartmentNode));
 }
    break;

  case 13:

/* Line 678 of lalr1.cc  */
#line 247 "parser.yy"
    {
  //empty
  (yyval.termsMultiset) = new pair<Species *, vector<Compartment *> *>(new Species(atom_table.size()), new vector<Compartment *>);
  }
    break;

  case 14:

/* Line 678 of lalr1.cc  */
#line 256 "parser.yy"
    {
  //non-empty-content compartment
  (yyval.pcompartmentNode) = new PCompartment((yysemantic_stack_[(8) - (3)].atomsMultiset), (yysemantic_stack_[(8) - (6)].patternsMultiset)->first, (yysemantic_stack_[(8) - (6)].patternsMultiset)->second, *(yysemantic_stack_[(8) - (4)].wrapVariable), *(yysemantic_stack_[(8) - (7)].termVariable), (yysemantic_stack_[(8) - (2)].compartmentType));
  delete (yysemantic_stack_[(8) - (6)].patternsMultiset);
 }
    break;

  case 15:

/* Line 678 of lalr1.cc  */
#line 261 "parser.yy"
    {
  //empty-content compartment
  (yyval.pcompartmentNode) = new PCompartment((yysemantic_stack_[(7) - (3)].atomsMultiset), new Species(atom_table.size()), new vector<PCompartment *>, *(yysemantic_stack_[(7) - (4)].wrapVariable), *(yysemantic_stack_[(7) - (6)].termVariable), (yysemantic_stack_[(7) - (2)].compartmentType));
  }
    break;

  case 16:

/* Line 678 of lalr1.cc  */
#line 266 "parser.yy"
    {
  //species
  (yysemantic_stack_[(2) - (1)].patternsMultiset)->first->add((yysemantic_stack_[(2) - (2)].atomNode)->first, (yysemantic_stack_[(2) - (2)].atomNode)->second);
  delete (yysemantic_stack_[(2) - (2)].atomNode);
 }
    break;

  case 17:

/* Line 678 of lalr1.cc  */
#line 271 "parser.yy"
    {
  //compartment
  (yysemantic_stack_[(2) - (1)].patternsMultiset)->second->push_back((yysemantic_stack_[(2) - (2)].pcompartmentNode));
 }
    break;

  case 18:

/* Line 678 of lalr1.cc  */
#line 275 "parser.yy"
    {
  Species *species = new Species(atom_table.size());
  species->add((yysemantic_stack_[(1) - (1)].atomNode)->first, (yysemantic_stack_[(1) - (1)].atomNode)->second);
  delete (yysemantic_stack_[(1) - (1)].atomNode);
  (yyval.patternsMultiset) = new pair<Species *, vector<PCompartment *> *>(species, new vector<PCompartment *>);
  }
    break;

  case 19:

/* Line 678 of lalr1.cc  */
#line 281 "parser.yy"
    {
  vector<PCompartment *> *cc = new vector<PCompartment *>;
  cc->push_back((yysemantic_stack_[(1) - (1)].pcompartmentNode));
  (yyval.patternsMultiset) = new pair<Species *, vector<PCompartment *> *>(new Species(atom_table.size()), cc);
  }
    break;

  case 20:

/* Line 678 of lalr1.cc  */
#line 291 "parser.yy"
    { (yysemantic_stack_[(2) - (1)].termVariablesMultiset)->push_back((yysemantic_stack_[(2) - (2)].termVariable));}
    break;

  case 21:

/* Line 678 of lalr1.cc  */
#line 292 "parser.yy"
    { (yyval.termVariablesMultiset) = new vector<string *>; }
    break;

  case 22:

/* Line 678 of lalr1.cc  */
#line 294 "parser.yy"
    { (yysemantic_stack_[(2) - (1)].wrapVariablesMultiset)->push_back((yysemantic_stack_[(2) - (2)].wrapVariable));}
    break;

  case 23:

/* Line 678 of lalr1.cc  */
#line 295 "parser.yy"
    { (yyval.wrapVariablesMultiset) = new vector<string *>; }
    break;

  case 24:

/* Line 678 of lalr1.cc  */
#line 297 "parser.yy"
    {
  (yyval.ocompartmentNode) = new OCompartment((yysemantic_stack_[(8) - (3)].atomsMultiset), (yysemantic_stack_[(8) - (6)].opentermMultiset)->first, (yysemantic_stack_[(8) - (6)].opentermMultiset)->second, *(yysemantic_stack_[(8) - (4)].wrapVariablesMultiset), *(yysemantic_stack_[(8) - (7)].termVariablesMultiset), (yysemantic_stack_[(8) - (2)].compartmentType));
  delete (yysemantic_stack_[(8) - (6)].opentermMultiset);
 }
    break;

  case 25:

/* Line 678 of lalr1.cc  */
#line 302 "parser.yy"
    {
  (yysemantic_stack_[(2) - (1)].opentermMultiset)->first->add((yysemantic_stack_[(2) - (2)].atomNode)->first, (yysemantic_stack_[(2) - (2)].atomNode)->second);
  delete (yysemantic_stack_[(2) - (2)].atomNode);
 }
    break;

  case 26:

/* Line 678 of lalr1.cc  */
#line 306 "parser.yy"
    {
  (yysemantic_stack_[(2) - (1)].opentermMultiset)->second->push_back((yysemantic_stack_[(2) - (2)].ocompartmentNode));
 }
    break;

  case 27:

/* Line 678 of lalr1.cc  */
#line 309 "parser.yy"
    {
  (yyval.opentermMultiset) = new pair<Species *, vector<OCompartment *> *>(new Species(atom_table.size()), new vector<OCompartment *>);
  }
    break;

  case 28:

/* Line 678 of lalr1.cc  */
#line 316 "parser.yy"
    {
  (yyval.parametersList) = new vector<double>(1, (yysemantic_stack_[(1) - (1)].constantRate));
}
    break;

  case 29:

/* Line 678 of lalr1.cc  */
#line 319 "parser.yy"
    {
  (yysemantic_stack_[(2) - (1)].parametersList)->push_back((yysemantic_stack_[(2) - (2)].constantRate));
  (yyval.parametersList) = (yysemantic_stack_[(2) - (1)].parametersList);
}
    break;

  case 30:

/* Line 678 of lalr1.cc  */
#line 325 "parser.yy"
    {
  int p_lhs = (yysemantic_stack_[(11) - (2)].patternsMultiset)->second->size();
  if(p_lhs <= 1) {
    (yyval.ruleNode) = new Rule(*(yysemantic_stack_[(11) - (2)].patternsMultiset)->first, *(yysemantic_stack_[(11) - (2)].patternsMultiset)->second, *(yysemantic_stack_[(11) - (10)].opentermMultiset)->first, *(yysemantic_stack_[(11) - (10)].opentermMultiset)->second, (yysemantic_stack_[(11) - (6)].multiplicity), *(yysemantic_stack_[(11) - (7)].parametersList), *(yysemantic_stack_[(11) - (3)].termVariable), *(yysemantic_stack_[(11) - (11)].termVariablesMultiset), (yysemantic_stack_[(11) - (1)].compartmentType));
    delete (yysemantic_stack_[(11) - (2)].patternsMultiset);
    delete (yysemantic_stack_[(11) - (10)].opentermMultiset);
    delete (yysemantic_stack_[(11) - (7)].parametersList);
  }
  else {
    if(p_lhs > 2) {
      error(yyloc, std::string("No more than two compartment for rule are allowed"));
    }
    else {
      error(yyloc, std::string("Not Implemented Yet: two compartments in the left hand side of a rule"));
    }
    delete (yysemantic_stack_[(11) - (2)].patternsMultiset)->first;
    for(unsigned int i = 0; i < (yysemantic_stack_[(11) - (2)].patternsMultiset)->second->size(); ++i) delete ((yysemantic_stack_[(11) - (2)].patternsMultiset)->second->at(i));
    delete (yysemantic_stack_[(11) - (2)].patternsMultiset)->second;
    delete (yysemantic_stack_[(11) - (2)].patternsMultiset);
    delete (yysemantic_stack_[(11) - (10)].opentermMultiset)->first;
    for(unsigned int i = 0; i < (yysemantic_stack_[(11) - (10)].opentermMultiset)->second->size(); ++i) delete ((yysemantic_stack_[(11) - (10)].opentermMultiset)->second->at(i));
    delete (yysemantic_stack_[(11) - (10)].opentermMultiset)->second;
    delete (yysemantic_stack_[(11) - (10)].opentermMultiset);
    delete (yysemantic_stack_[(11) - (7)].parametersList);
    YYERROR;
  }
 }
    break;

  case 31:

/* Line 678 of lalr1.cc  */
#line 352 "parser.yy"
    {
  int p_lhs = (yysemantic_stack_[(10) - (2)].patternsMultiset)->second->size();
  if(p_lhs <= 1) {
    (yyval.ruleNode) = new Rule(*(yysemantic_stack_[(10) - (2)].patternsMultiset)->first, *(yysemantic_stack_[(10) - (2)].patternsMultiset)->second, *(yysemantic_stack_[(10) - (9)].opentermMultiset)->first, *(yysemantic_stack_[(10) - (9)].opentermMultiset)->second, 0, *(yysemantic_stack_[(10) - (6)].parametersList), *(yysemantic_stack_[(10) - (3)].termVariable), *(yysemantic_stack_[(10) - (10)].termVariablesMultiset), (yysemantic_stack_[(10) - (1)].compartmentType));
    delete (yysemantic_stack_[(10) - (2)].patternsMultiset);
    delete (yysemantic_stack_[(10) - (9)].opentermMultiset);
  }
  else {
    if(p_lhs > 2) {
      error(yyloc, std::string("No more than two compartment for rule are allowed"));
    }
    else {
      error(yyloc, std::string("Not Implemented Yet: two compartments in the left hand side of a rule"));
    }
    delete (yysemantic_stack_[(10) - (2)].patternsMultiset)->first;
    for(unsigned int i = 0; i < (yysemantic_stack_[(10) - (2)].patternsMultiset)->second->size(); ++i) delete ((yysemantic_stack_[(10) - (2)].patternsMultiset)->second->at(i));
    delete (yysemantic_stack_[(10) - (2)].patternsMultiset)->second;
    delete (yysemantic_stack_[(10) - (2)].patternsMultiset);
    delete (yysemantic_stack_[(10) - (9)].opentermMultiset)->first;
    for(unsigned int i = 0; i < (yysemantic_stack_[(10) - (9)].opentermMultiset)->second->size(); ++i) delete ((yysemantic_stack_[(10) - (9)].opentermMultiset)->second->at(i));
    delete (yysemantic_stack_[(10) - (9)].opentermMultiset)->second;
    delete (yysemantic_stack_[(10) - (9)].opentermMultiset);
    delete (yysemantic_stack_[(10) - (6)].parametersList);
    YYERROR;
  }
 }
    break;

  case 32:

/* Line 678 of lalr1.cc  */
#line 380 "parser.yy"
    {
  if((yysemantic_stack_[(3) - (2)].ruleNode)->is_biochemical) {
    (yysemantic_stack_[(3) - (2)].ruleNode)->ode_index = ode_offset;
    ode_offset += (yysemantic_stack_[(3) - (2)].ruleNode)->parameters.size();
  }
  (yysemantic_stack_[(3) - (1)].rulesList)->push_back((yysemantic_stack_[(3) - (2)].ruleNode));
 }
    break;

  case 33:

/* Line 678 of lalr1.cc  */
#line 387 "parser.yy"
    {
  vector<Rule *> *v = new vector<Rule *>;
  if((yysemantic_stack_[(2) - (1)].ruleNode)->is_biochemical) {
    (yysemantic_stack_[(2) - (1)].ruleNode)->ode_index = ode_offset;
    ode_offset += (yysemantic_stack_[(2) - (1)].ruleNode)->parameters.size();
  }
  v->push_back((yysemantic_stack_[(2) - (1)].ruleNode));
  (yyval.rulesList) = v;
 }
    break;

  case 34:

/* Line 678 of lalr1.cc  */
#line 397 "parser.yy"
    {
  string *t = new string((yysemantic_stack_[(4) - (1)].qstring)->substr(1, (yysemantic_stack_[(4) - (1)].qstring)->size() - 2)); //cut quotes
  (yyval.monitorNode) = new Monitor(*t, (yysemantic_stack_[(4) - (4)].patternsMultiset)->first, (yysemantic_stack_[(4) - (4)].patternsMultiset)->second, (yysemantic_stack_[(4) - (3)].compartmentType));
  delete (yysemantic_stack_[(4) - (4)].patternsMultiset);
  delete (yysemantic_stack_[(4) - (1)].qstring);
 }
    break;

  case 35:

/* Line 678 of lalr1.cc  */
#line 404 "parser.yy"
    {
  (yysemantic_stack_[(3) - (1)].monitorsList)->push_back((yysemantic_stack_[(3) - (2)].monitorNode));
 }
    break;

  case 36:

/* Line 678 of lalr1.cc  */
#line 407 "parser.yy"
    {
  vector<Monitor *> *v = new vector<Monitor *>;
  v->push_back((yysemantic_stack_[(2) - (1)].monitorNode));
  (yyval.monitorsList) = v;
 }
    break;

  case 37:

/* Line 678 of lalr1.cc  */
#line 418 "parser.yy"
    {
  //print_symbol_table();
  string *title = new string((yysemantic_stack_[(10) - (2)].qstring)->substr(1, (yysemantic_stack_[(10) - (2)].qstring)->size() - 2)); //cut quotes
  (yyval.modelNode) = new Model(*title, *(yysemantic_stack_[(10) - (4)].alphabet), *(yysemantic_stack_[(10) - (6)].rulesList), ode_offset, *(yysemantic_stack_[(10) - (8)].termsMultiset)->first, *(yysemantic_stack_[(10) - (8)].termsMultiset)->second, *(yysemantic_stack_[(10) - (10)].monitorsList));
  delete (yysemantic_stack_[(10) - (2)].qstring);
  delete (yysemantic_stack_[(10) - (8)].termsMultiset);
}
    break;

  case 39:

/* Line 678 of lalr1.cc  */
#line 427 "parser.yy"
    { driver.model = (yysemantic_stack_[(3) - (2)].modelNode); }
    break;



/* Line 678 of lalr1.cc  */
#line 907 "parser.cpp"
	default:
          break;
      }
    YY_SYMBOL_PRINT ("-> $$ =", yyr1_[yyn], &yyval, &yyloc);

    yypop_ (yylen);
    yylen = 0;
    YY_STACK_PRINT ();

    yysemantic_stack_.push (yyval);
    yylocation_stack_.push (yyloc);

    /* Shift the result of the reduction.  */
    yyn = yyr1_[yyn];
    yystate = yypgoto_[yyn - yyntokens_] + yystate_stack_[0];
    if (0 <= yystate && yystate <= yylast_
	&& yycheck_[yystate] == yystate_stack_[0])
      yystate = yytable_[yystate];
    else
      yystate = yydefgoto_[yyn - yyntokens_];
    goto yynewstate;

  /*------------------------------------.
  | yyerrlab -- here on detecting error |
  `------------------------------------*/
  yyerrlab:
    /* If not already recovering from an error, report this error.  */
    if (!yyerrstatus_)
      {
	++yynerrs_;
	error (yylloc, yysyntax_error_ (yystate, yytoken));
      }

    yyerror_range[1] = yylloc;
    if (yyerrstatus_ == 3)
      {
	/* If just tried and failed to reuse lookahead token after an
	 error, discard it.  */

	if (yychar <= yyeof_)
	  {
	  /* Return failure if at end of input.  */
	  if (yychar == yyeof_)
	    YYABORT;
	  }
	else
	  {
	    yydestruct_ ("Error: discarding", yytoken, &yylval, &yylloc);
	    yychar = yyempty_;
	  }
      }

    /* Else will try to reuse lookahead token after shifting the error
       token.  */
    goto yyerrlab1;


  /*---------------------------------------------------.
  | yyerrorlab -- error raised explicitly by YYERROR.  |
  `---------------------------------------------------*/
  yyerrorlab:

    /* Pacify compilers like GCC when the user code never invokes
       YYERROR and the label yyerrorlab therefore never appears in user
       code.  */
    if (false)
      goto yyerrorlab;

    yyerror_range[1] = yylocation_stack_[yylen - 1];
    /* Do not reclaim the symbols of the rule which action triggered
       this YYERROR.  */
    yypop_ (yylen);
    yylen = 0;
    yystate = yystate_stack_[0];
    goto yyerrlab1;

  /*-------------------------------------------------------------.
  | yyerrlab1 -- common code for both syntax error and YYERROR.  |
  `-------------------------------------------------------------*/
  yyerrlab1:
    yyerrstatus_ = 3;	/* Each real token shifted decrements this.  */

    for (;;)
      {
	yyn = yypact_[yystate];
	if (yyn != yypact_ninf_)
	{
	  yyn += yyterror_;
	  if (0 <= yyn && yyn <= yylast_ && yycheck_[yyn] == yyterror_)
	    {
	      yyn = yytable_[yyn];
	      if (0 < yyn)
		break;
	    }
	}

	/* Pop the current state because it cannot handle the error token.  */
	if (yystate_stack_.height () == 1)
	YYABORT;

	yyerror_range[1] = yylocation_stack_[0];
	yydestruct_ ("Error: popping",
		     yystos_[yystate],
		     &yysemantic_stack_[0], &yylocation_stack_[0]);
	yypop_ ();
	yystate = yystate_stack_[0];
	YY_STACK_PRINT ();
      }

    yyerror_range[2] = yylloc;
    // Using YYLLOC is tempting, but would change the location of
    // the lookahead.  YYLOC is available though.
    YYLLOC_DEFAULT (yyloc, yyerror_range, 2);
    yysemantic_stack_.push (yylval);
    yylocation_stack_.push (yyloc);

    /* Shift the error token.  */
    YY_SYMBOL_PRINT ("Shifting", yystos_[yyn],
		     &yysemantic_stack_[0], &yylocation_stack_[0]);

    yystate = yyn;
    goto yynewstate;

    /* Accept.  */
  yyacceptlab:
    yyresult = 0;
    goto yyreturn;

    /* Abort.  */
  yyabortlab:
    yyresult = 1;
    goto yyreturn;

  yyreturn:
    if (yychar != yyempty_)
      yydestruct_ ("Cleanup: discarding lookahead", yytoken, &yylval, &yylloc);

    /* Do not reclaim the symbols of the rule which action triggered
       this YYABORT or YYACCEPT.  */
    yypop_ (yylen);
    while (yystate_stack_.height () != 1)
      {
	yydestruct_ ("Cleanup: popping",
		   yystos_[yystate_stack_[0]],
		   &yysemantic_stack_[0],
		   &yylocation_stack_[0]);
	yypop_ ();
      }

    return yyresult;
  }

  // Generate an error message.
  std::string
  Parser::yysyntax_error_ (int yystate, int tok)
  {
    std::string res;
    YYUSE (yystate);
#if YYERROR_VERBOSE
    int yyn = yypact_[yystate];
    if (yypact_ninf_ < yyn && yyn <= yylast_)
      {
	/* Start YYX at -YYN if negative to avoid negative indexes in
	   YYCHECK.  */
	int yyxbegin = yyn < 0 ? -yyn : 0;

	/* Stay within bounds of both yycheck and yytname.  */
	int yychecklim = yylast_ - yyn + 1;
	int yyxend = yychecklim < yyntokens_ ? yychecklim : yyntokens_;
	int count = 0;
	for (int x = yyxbegin; x < yyxend; ++x)
	  if (yycheck_[x + yyn] == x && x != yyterror_)
	    ++count;

	// FIXME: This method of building the message is not compatible
	// with internationalization.  It should work like yacc.c does it.
	// That is, first build a string that looks like this:
	// "syntax error, unexpected %s or %s or %s"
	// Then, invoke YY_ on this string.
	// Finally, use the string as a format to output
	// yytname_[tok], etc.
	// Until this gets fixed, this message appears in English only.
	res = "syntax error, unexpected ";
	res += yytnamerr_ (yytname_[tok]);
	if (count < 5)
	  {
	    count = 0;
	    for (int x = yyxbegin; x < yyxend; ++x)
	      if (yycheck_[x + yyn] == x && x != yyterror_)
		{
		  res += (!count++) ? ", expecting " : " or ";
		  res += yytnamerr_ (yytname_[x]);
		}
	  }
      }
    else
#endif
      res = YY_("syntax error");
    return res;
  }


  /* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
     STATE-NUM.  */
  const signed char Parser::yypact_ninf_ = -68;
  const signed char
  Parser::yypact_[] =
  {
       -68,    71,   -68,     2,    28,    37,   -68,   -68,    32,    44,
     -68,    60,    66,    42,    76,     7,    62,    67,   -68,    60,
     -68,   -68,    29,   -68,   -68,    79,   -68,    72,   -68,    83,
     -68,   -68,    18,   -68,   -68,    59,    64,    74,    60,   -68,
     -68,    68,   -68,    -4,    65,    87,    74,   -68,    43,    80,
     -68,    -7,    60,   -68,    89,     1,    73,    54,    -3,   -68,
      91,    42,   -68,   -68,   -68,    75,    93,   -68,    42,    19,
     -68,   -68,    48,   -68,    48,    60,   -68,    84,   -68,    84,
     -68,   -68,    13,    61,   -68,   -68,    48,    51,   -68
  };

  /* YYDEFACT[S] -- default rule to reduce with in state S when YYTABLE
     doesn't specify something else to do.  Zero means the default is an
     error.  */
  const unsigned char
  Parser::yydefact_[] =
  {
        38,     0,     1,     0,     0,     0,    39,     3,     4,     0,
       2,     0,     0,     0,     0,     0,     0,     0,     5,     0,
      18,    19,     0,    33,    13,     0,     9,     0,     8,     0,
      16,    17,     0,    32,     6,     0,     0,     0,     0,    11,
      12,     0,     7,     0,     0,     0,    37,     8,     0,     0,
      28,     0,     0,    36,     0,     0,     0,     0,     0,    29,
       0,     0,    35,    13,    15,     0,     0,    27,    34,     0,
      14,    27,    21,    10,    21,     0,    25,    31,    26,    30,
       8,    20,    23,     0,    22,    27,    21,     0,    24
  };

  /* YYPGOTO[NTERM-NUM].  */
  const signed char
  Parser::yypgoto_[] =
  {
       -68,   -68,   -68,   -22,   -44,   -18,   -68,    35,   -20,   -42,
     -63,   -68,   -68,   -67,    52,    85,   -68,    56,   -68,   -68,
     -68
  };

  /* YYDEFGOTO[NTERM-NUM].  */
  const signed char
  Parser::yydefgoto_[] =
  {
        -1,     8,     9,    20,    35,    13,    40,    32,    21,    22,
      77,    83,    78,    72,    51,    14,    15,    45,    46,     4,
       1
  };

  /* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
     positive, shift that token.  If negative, reduce the rule which
     number is the opposite.  If zero, do what YYDEFACT says.  */
  const signed char Parser::yytable_ninf_ = -1;
  const unsigned char
  Parser::yytable_[] =
  {
        30,    28,    31,    55,    74,    59,    57,    49,    50,    59,
      39,    79,    17,    42,    18,    24,     5,    60,    86,    68,
      47,    66,    63,    87,    17,    12,    18,    37,     6,    17,
      17,    18,    18,    42,    61,    30,    82,    31,    38,    38,
      17,    73,    18,     7,    29,    10,    30,    39,    31,    19,
      76,    11,    76,    17,    17,    18,    18,    80,    56,    17,
      42,    18,    19,    19,    76,    17,    81,    18,    75,    65,
      17,     2,    18,    88,    19,    41,     3,    84,    12,    16,
      23,    26,    85,    33,    27,    34,    36,    43,    44,    48,
      52,    53,    50,    62,    67,    64,    71,    70,    69,    81,
      25,    58,    54
  };

  /* YYCHECK.  */
  const unsigned char
  Parser::yycheck_[] =
  {
        22,    19,    22,    47,    71,    12,    48,    11,    12,    12,
      32,    74,    11,    35,    13,     8,    14,    24,    85,    61,
      38,    24,    21,    86,    11,    18,    13,     9,     0,    11,
      11,    13,    13,    55,    52,    57,    80,    57,    20,    20,
      11,    22,    13,     6,    15,    13,    68,    69,    68,    20,
      72,     7,    74,    11,    11,    13,    13,    75,    15,    11,
      82,    13,    20,    20,    86,    11,    15,    13,    20,    15,
      11,     0,    13,    22,    20,    16,     5,    16,    18,    13,
       4,    19,    21,     4,    17,    13,     3,    23,    14,    21,
      25,     4,    12,     4,     3,    22,     3,    22,    63,    15,
      15,    49,    46
  };

  /* STOS_[STATE-NUM] -- The (internal number of the) accessing
     symbol of state STATE-NUM.  */
  const unsigned char
  Parser::yystos_[] =
  {
         0,    46,     0,     5,    45,    14,     0,     6,    27,    28,
      13,     7,    18,    31,    41,    42,    13,    11,    13,    20,
      29,    34,    35,     4,     8,    41,    19,    17,    31,    15,
      29,    34,    33,     4,    13,    30,     3,     9,    20,    29,
      32,    16,    29,    23,    14,    43,    44,    31,    21,    11,
      12,    40,    25,     4,    43,    30,    15,    35,    40,    12,
      24,    31,     4,    21,    22,    15,    24,     3,    35,    33,
      22,     3,    39,    22,    39,    20,    29,    36,    38,    36,
      31,    15,    30,    37,    16,    21,    39,    36,    22
  };

#if YYDEBUG
  /* TOKEN_NUMBER_[YYLEX-NUM] -- Internal symbol number corresponding
     to YYLEX-NUM.  */
  const unsigned short int
  Parser::yytoken_number_[] =
  {
         0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,    42,   123,   125,
      40,   124,    41,    91,    93,    58
  };
#endif

  /* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
  const unsigned char
  Parser::yyr1_[] =
  {
         0,    26,    27,    27,    28,    29,    29,    30,    30,    31,
      32,    33,    33,    33,    34,    34,    35,    35,    35,    35,
      36,    36,    37,    37,    38,    39,    39,    39,    40,    40,
      41,    41,    42,    42,    43,    44,    44,    45,    46,    46
  };

  /* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
  const unsigned char
  Parser::yyr2_[] =
  {
         0,     2,     2,     0,     1,     1,     3,     2,     0,     3,
       6,     2,     2,     0,     8,     7,     2,     2,     1,     1,
       2,     0,     2,     0,     8,     2,     2,     0,     1,     2,
      11,    10,     3,     2,     4,     3,     2,    10,     0,     3
  };

#if YYDEBUG || YYERROR_VERBOSE || YYTOKEN_TABLE
  /* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
     First, the terminals, then, starting at \a yyntokens_, nonterminals.  */
  const char*
  const Parser::yytname_[] =
  {
    "\"end of file\"", "error", "$undefined", "\">>>\"", "\"%%\"",
  "\"%model\"", "\"%alphabet\"", "\"%rules\"", "\"%term\"",
  "\"%monitors\"", "\"%end\"", "\"integer\"", "\"double\"",
  "\"atom_label\"", "\"qstring\"", "\"tvar\"", "\"wvar\"", "'*'", "'{'",
  "'}'", "'('", "'|'", "')'", "'['", "']'", "':'", "$accept", "alphabet_",
  "alphabet", "atom", "atoms_multiset", "compartment_type", "compartment",
  "terms_multiset", "pcompartment", "patterns_multiset", "tvar_multiset",
  "wvar_multiset", "ocompartment", "openterm_multiset", "parameters",
  "rule", "rules_list", "monitor", "monitors_list", "model", "start", 0
  };
#endif

#if YYDEBUG
  /* YYRHS -- A `-1'-separated list of the rules' RHS.  */
  const Parser::rhs_number_type
  Parser::yyrhs_[] =
  {
        46,     0,    -1,    27,    13,    -1,    -1,    27,    -1,    13,
      -1,    11,    17,    13,    -1,    30,    29,    -1,    -1,    18,
      13,    19,    -1,    20,    31,    30,    21,    33,    22,    -1,
      33,    29,    -1,    33,    32,    -1,    -1,    20,    31,    30,
      16,    21,    35,    15,    22,    -1,    20,    31,    30,    16,
      21,    15,    22,    -1,    35,    29,    -1,    35,    34,    -1,
      29,    -1,    34,    -1,    36,    15,    -1,    -1,    37,    16,
      -1,    -1,    20,    31,    30,    37,    21,    39,    36,    22,
      -1,    39,    29,    -1,    39,    38,    -1,    -1,    12,    -1,
      40,    12,    -1,    31,    35,    15,     3,    23,    11,    40,
      24,     3,    39,    36,    -1,    31,    35,    15,     3,    23,
      40,    24,     3,    39,    36,    -1,    42,    41,     4,    -1,
      41,     4,    -1,    14,    25,    31,    35,    -1,    44,    43,
       4,    -1,    43,     4,    -1,     5,    14,     6,    28,     7,
      42,     8,    33,     9,    44,    -1,    -1,    46,    45,     0,
      -1
  };

  /* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
     YYRHS.  */
  const unsigned char
  Parser::yyprhs_[] =
  {
         0,     0,     3,     6,     7,     9,    11,    15,    18,    19,
      23,    30,    33,    36,    37,    46,    54,    57,    60,    62,
      64,    67,    68,    71,    72,    81,    84,    87,    88,    90,
      93,   105,   116,   120,   123,   128,   132,   135,   146,   147
  };

  /* YYRLINE[YYN] -- Source line where rule number YYN was defined.  */
  const unsigned short int
  Parser::yyrline_[] =
  {
         0,   166,   166,   170,   175,   185,   190,   201,   205,   210,
     227,   238,   243,   247,   256,   261,   266,   271,   275,   281,
     291,   292,   294,   295,   297,   302,   306,   309,   316,   319,
     325,   352,   380,   387,   397,   404,   407,   413,   426,   427
  };

  // Print the state stack on the debug stream.
  void
  Parser::yystack_print_ ()
  {
    *yycdebug_ << "Stack now";
    for (state_stack_type::const_iterator i = yystate_stack_.begin ();
	 i != yystate_stack_.end (); ++i)
      *yycdebug_ << ' ' << *i;
    *yycdebug_ << std::endl;
  }

  // Report on the debug stream that the rule \a yyrule is going to be reduced.
  void
  Parser::yy_reduce_print_ (int yyrule)
  {
    unsigned int yylno = yyrline_[yyrule];
    int yynrhs = yyr2_[yyrule];
    /* Print the symbols being reduced, and their result.  */
    *yycdebug_ << "Reducing stack by rule " << yyrule - 1
	       << " (line " << yylno << "):" << std::endl;
    /* The symbols being reduced.  */
    for (int yyi = 0; yyi < yynrhs; yyi++)
      YY_SYMBOL_PRINT ("   $" << yyi + 1 << " =",
		       yyrhs_[yyprhs_[yyrule] + yyi],
		       &(yysemantic_stack_[(yynrhs) - (yyi + 1)]),
		       &(yylocation_stack_[(yynrhs) - (yyi + 1)]));
  }
#endif // YYDEBUG

  /* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
  Parser::token_number_type
  Parser::yytranslate_ (int t)
  {
    static
    const token_number_type
    translate_table[] =
    {
           0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
      20,    22,    17,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    25,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,    23,     2,    24,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    18,    21,    19,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16
    };
    if ((unsigned int) t <= yyuser_token_number_max_)
      return translate_table[t];
    else
      return yyundef_token_;
  }

  const int Parser::yyeof_ = 0;
  const int Parser::yylast_ = 102;
  const int Parser::yynnts_ = 21;
  const int Parser::yyempty_ = -2;
  const int Parser::yyfinal_ = 2;
  const int Parser::yyterror_ = 1;
  const int Parser::yyerrcode_ = 256;
  const int Parser::yyntokens_ = 26;

  const unsigned int Parser::yyuser_token_number_max_ = 271;
  const Parser::token_number_type Parser::yyundef_token_ = 2;


} // scwc

/* Line 1054 of lalr1.cc  */
#line 1397 "parser.cpp"


/* Line 1056 of lalr1.cc  */
#line 431 "parser.yy"
 /*** Additional Code ***/

void scwc::Parser::error(const Parser::location_type& l,
			 const std::string& m)
{
  driver.error(l, m);
}

