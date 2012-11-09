/* A Bison parser, made by GNU Bison 2.5.1.  */

/* Skeleton implementation for Bison LALR(1) parsers in C++
   
      Copyright (C) 2002-2012 Free Software Foundation, Inc.
   
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

/* Line 298 of lalr1.cc  */
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

  typedef map<string, unsigned int> symbol_table;

  symbol_table atom_table;
  symbol_table type_table;
  symbol_table variable_table;
  symbol_adress last_type_free = TOP_LEVEL + 1;
  variable_adress last_variable_free = VARIABLE_ADRESS_RESERVED + 1;

  void print_symbol_table() {
    for(symbol_table::iterator it = atom_table.begin(); it != atom_table.end(); it++) {
      cout << it->first << ": " << it->second << endl;
    }
  }

  unsigned int ode_offset = 0;

  

/* Line 298 of lalr1.cc  */
#line 79 "parser.cpp"


#include "parser.h"

/* User implementation prologue.  */

/* Line 304 of lalr1.cc  */
#line 165 "parser.yy"


#include "Driver.h"
#include "scanner.h"

  /* this "connects" the bison parser in the driver to the flex scanner class
   * object. it defines the yylex() function call to pull the next token from the
   * current lexer object of the driver context. */
#undef yylex
#define yylex driver.lexer->lex

  

/* Line 304 of lalr1.cc  */
#line 102 "parser.cpp"

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

/* YYLLOC_DEFAULT -- Set CURRENT to span from RHS[1] to RHS[N].
   If N is 0, then set CURRENT to the empty location which ends
   the previous symbol: RHS[0] (always defined).  */

#define YYRHSLOC(Rhs, K) ((Rhs)[K])
#ifndef YYLLOC_DEFAULT
# define YYLLOC_DEFAULT(Current, Rhs, N)                               \
 do                                                                    \
   if (N)                                                              \
     {                                                                 \
       (Current).begin = YYRHSLOC (Rhs, 1).begin;                      \
       (Current).end   = YYRHSLOC (Rhs, N).end;                        \
     }                                                                 \
   else                                                                \
     {                                                                 \
       (Current).begin = (Current).end = YYRHSLOC (Rhs, 0).end;        \
     }                                                                 \
 while (false)
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

/* Line 387 of lalr1.cc  */
#line 188 "parser.cpp"

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
    std::ostream& yyo = debug_stream ();
    std::ostream& yyoutput = yyo;
    YYUSE (yyoutput);
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

  inline bool
  Parser::yy_pact_value_is_default_ (int yyvalue)
  {
    return yyvalue == yypact_ninf_;
  }

  inline bool
  Parser::yy_table_value_is_error_ (int yyvalue)
  {
    return yyvalue == yytable_ninf_;
  }

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
    
/* Line 573 of lalr1.cc  */
#line 65 "parser.yy"
{
    // initialize the initial location object
    yylloc.begin.filename = yylloc.end.filename = &driver.streamname;
  }

/* Line 573 of lalr1.cc  */
#line 385 "parser.cpp"

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
    if (yy_pact_value_is_default_ (yyn))
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
	if (yy_table_value_is_error_ (yyn))
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

/* Line 698 of lalr1.cc  */
#line 182 "parser.yy"
    {
  (yysemantic_stack_[(2) - (1)].alphabet)->push_back(*(yysemantic_stack_[(2) - (2)].atomLabel));
  delete (yysemantic_stack_[(2) - (2)].atomLabel);
 }
    break;

  case 3:

/* Line 698 of lalr1.cc  */
#line 186 "parser.yy"
    {
  //empty
  (yyval.alphabet) = new vector<string>;
 }
    break;

  case 4:

/* Line 698 of lalr1.cc  */
#line 191 "parser.yy"
    {
  reverse_symbol_table &rst = *(yysemantic_stack_[(1) - (1)].alphabet);
  sort(rst.begin(), rst.end());
  for(symbol_adress i=0; i<rst.size(); i++) {
    atom_table.insert(pair<string, symbol_adress>(rst[i], i));
  }
  type_table.insert(pair<string, symbol_adress>(string("T"), TOP_LEVEL));
  type_table.insert(pair<string, symbol_adress>(string("ALL"), ANY_TYPE));
  //last_type_free = TOP_LEVEL + 1;
 }
    break;

  case 5:

/* Line 698 of lalr1.cc  */
#line 202 "parser.yy"
    {
  //singleton
  (yyval.atomNode) = new pair<symbol_adress, multiplicityType>(atom_table[*(yysemantic_stack_[(1) - (1)].atomLabel)], 1);
  delete (yysemantic_stack_[(1) - (1)].atomLabel);
 }
    break;

  case 6:

/* Line 698 of lalr1.cc  */
#line 207 "parser.yy"
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

/* Line 698 of lalr1.cc  */
#line 218 "parser.yy"
    {
  (yysemantic_stack_[(2) - (1)].atomsMultiset)->add((yysemantic_stack_[(2) - (2)].atomNode)->first, (yysemantic_stack_[(2) - (2)].atomNode)->second);
  delete (yysemantic_stack_[(2) - (2)].atomNode);
 }
    break;

  case 8:

/* Line 698 of lalr1.cc  */
#line 222 "parser.yy"
    {
  //empty
  (yyval.atomsMultiset) = new Species(atom_table.size());
 }
    break;

  case 9:

/* Line 698 of lalr1.cc  */
#line 227 "parser.yy"
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

/* Line 698 of lalr1.cc  */
#line 244 "parser.yy"
    {
  //ground compartment
  (yyval.compartmentNode) = new Compartment((yysemantic_stack_[(6) - (3)].atomsMultiset), (yysemantic_stack_[(6) - (5)].termsMultiset)->first, (yysemantic_stack_[(6) - (5)].termsMultiset)->second, (yysemantic_stack_[(6) - (2)].compartmentType));
  delete (yysemantic_stack_[(6) - (5)].termsMultiset);
 }
    break;

  case 11:

/* Line 698 of lalr1.cc  */
#line 255 "parser.yy"
    { 
  //species
  (yysemantic_stack_[(2) - (1)].termsMultiset)->first->add((yysemantic_stack_[(2) - (2)].atomNode)->first, (yysemantic_stack_[(2) - (2)].atomNode)->second);
  delete (yysemantic_stack_[(2) - (2)].atomNode);
 }
    break;

  case 12:

/* Line 698 of lalr1.cc  */
#line 260 "parser.yy"
    {
  //compartment
  (yysemantic_stack_[(2) - (1)].termsMultiset)->second->push_back((yysemantic_stack_[(2) - (2)].compartmentNode));
 }
    break;

  case 13:

/* Line 698 of lalr1.cc  */
#line 264 "parser.yy"
    {
  //empty
  (yyval.termsMultiset) = new pair<Species *, vector<Compartment *> *>(new Species(atom_table.size()), new vector<Compartment *>);
  }
    break;

  case 14:

/* Line 698 of lalr1.cc  */
#line 273 "parser.yy"
    {
  if(!variable_table.count(*(yysemantic_stack_[(1) - (1)].termVariable)))
	variable_table[*(yysemantic_stack_[(1) - (1)].termVariable)] = last_variable_free++;
  (yyval.termVariable_adress) = variable_table[*(yysemantic_stack_[(1) - (1)].termVariable)];
  delete (yysemantic_stack_[(1) - (1)].termVariable);
}
    break;

  case 15:

/* Line 698 of lalr1.cc  */
#line 280 "parser.yy"
    {
  if(!variable_table.count(*(yysemantic_stack_[(1) - (1)].wrapVariable)))
	variable_table[*(yysemantic_stack_[(1) - (1)].wrapVariable)] = last_variable_free++;
  (yyval.wrapVariable_adress) = variable_table[*(yysemantic_stack_[(1) - (1)].wrapVariable)];
  delete (yysemantic_stack_[(1) - (1)].wrapVariable);	    
}
    break;

  case 16:

/* Line 698 of lalr1.cc  */
#line 287 "parser.yy"
    { (yysemantic_stack_[(2) - (1)].termVariablesAdresses)->push_back((yysemantic_stack_[(2) - (2)].termVariable_adress));}
    break;

  case 17:

/* Line 698 of lalr1.cc  */
#line 288 "parser.yy"
    { (yyval.termVariablesAdresses) = new vector<variable_adress>; }
    break;

  case 18:

/* Line 698 of lalr1.cc  */
#line 290 "parser.yy"
    { (yysemantic_stack_[(2) - (1)].wrapVariablesAdresses)->push_back((yysemantic_stack_[(2) - (2)].wrapVariable_adress));}
    break;

  case 19:

/* Line 698 of lalr1.cc  */
#line 291 "parser.yy"
    { (yyval.wrapVariablesAdresses) = new vector<variable_adress>; }
    break;

  case 20:

/* Line 698 of lalr1.cc  */
#line 305 "parser.yy"
    {
  //non-empty-content compartment
  (yyval.pcompartmentNode) = new PCompartment((yysemantic_stack_[(8) - (3)].atomsMultiset), (yysemantic_stack_[(8) - (6)].patternsMultiset)->first, (yysemantic_stack_[(8) - (6)].patternsMultiset)->second, (yysemantic_stack_[(8) - (4)].wrapVariable_adress), (yysemantic_stack_[(8) - (7)].termVariable_adress), (yysemantic_stack_[(8) - (2)].compartmentType));
  delete (yysemantic_stack_[(8) - (6)].patternsMultiset);
 }
    break;

  case 21:

/* Line 698 of lalr1.cc  */
#line 318 "parser.yy"
    {
  //species
  (yysemantic_stack_[(2) - (1)].patternsMultiset)->first->add((yysemantic_stack_[(2) - (2)].atomNode)->first, (yysemantic_stack_[(2) - (2)].atomNode)->second);
  delete (yysemantic_stack_[(2) - (2)].atomNode);
 }
    break;

  case 22:

/* Line 698 of lalr1.cc  */
#line 323 "parser.yy"
    {
  //compartment
  (yysemantic_stack_[(2) - (1)].patternsMultiset)->second->push_back((yysemantic_stack_[(2) - (2)].pcompartmentNode));
 }
    break;

  case 23:

/* Line 698 of lalr1.cc  */
#line 327 "parser.yy"
    {
  (yyval.patternsMultiset) = new pair<Species *, vector<PCompartment *> *>(new Species(atom_table.size()), new vector<PCompartment *>);
  }
    break;

  case 24:

/* Line 698 of lalr1.cc  */
#line 348 "parser.yy"
    {
  (yyval.ocompartmentNode) = new OCompartment((yysemantic_stack_[(8) - (3)].atomsMultiset), (yysemantic_stack_[(8) - (6)].opentermMultiset)->first, (yysemantic_stack_[(8) - (6)].opentermMultiset)->second, *(yysemantic_stack_[(8) - (4)].wrapVariablesAdresses), *(yysemantic_stack_[(8) - (7)].termVariablesAdresses), (yysemantic_stack_[(8) - (2)].compartmentType));
  delete (yysemantic_stack_[(8) - (6)].opentermMultiset);
 }
    break;

  case 25:

/* Line 698 of lalr1.cc  */
#line 353 "parser.yy"
    {
  (yysemantic_stack_[(2) - (1)].opentermMultiset)->first->add((yysemantic_stack_[(2) - (2)].atomNode)->first, (yysemantic_stack_[(2) - (2)].atomNode)->second);
  delete (yysemantic_stack_[(2) - (2)].atomNode);
 }
    break;

  case 26:

/* Line 698 of lalr1.cc  */
#line 357 "parser.yy"
    {
  (yysemantic_stack_[(2) - (1)].opentermMultiset)->second->push_back((yysemantic_stack_[(2) - (2)].ocompartmentNode));
 }
    break;

  case 27:

/* Line 698 of lalr1.cc  */
#line 360 "parser.yy"
    {
  (yyval.opentermMultiset) = new pair<Species *, vector<OCompartment *> *>(new Species(atom_table.size()), new vector<OCompartment *>);
  }
    break;

  case 28:

/* Line 698 of lalr1.cc  */
#line 367 "parser.yy"
    {
  (yyval.constantRate) = (yysemantic_stack_[(1) - (1)].constantRate);
}
    break;

  case 29:

/* Line 698 of lalr1.cc  */
#line 370 "parser.yy"
    {
  (yyval.constantRate) = (double)(yysemantic_stack_[(1) - (1)].multiplicity);
}
    break;

  case 30:

/* Line 698 of lalr1.cc  */
#line 374 "parser.yy"
    {
	    (yyval.parametersList) = new vector<double>(1, (yysemantic_stack_[(1) - (1)].constantRate));
	    }
    break;

  case 31:

/* Line 698 of lalr1.cc  */
#line 377 "parser.yy"
    {
  (yysemantic_stack_[(2) - (1)].parametersList)->push_back((yysemantic_stack_[(2) - (2)].constantRate));
}
    break;

  case 32:

/* Line 698 of lalr1.cc  */
#line 381 "parser.yy"
    {
  (yysemantic_stack_[(2) - (2)].speciesOccsList)->push_back(atom_table[*(yysemantic_stack_[(2) - (1)].atomLabel)]);
  (yyval.speciesOccsList) = (yysemantic_stack_[(2) - (2)].speciesOccsList);
}
    break;

  case 33:

/* Line 698 of lalr1.cc  */
#line 386 "parser.yy"
    {
  (yyval.speciesOccsList) = new vector<symbol_adress>();
  }
    break;

  case 34:

/* Line 698 of lalr1.cc  */
#line 391 "parser.yy"
    {
  vector<double> *p = new vector<double>(1, (yysemantic_stack_[(10) - (6)].constantRate));
  (yyval.ruleNode) = new Rule(*(yysemantic_stack_[(10) - (2)].patternsMultiset)->first, *(yysemantic_stack_[(10) - (2)].patternsMultiset)->second, *(yysemantic_stack_[(10) - (9)].opentermMultiset)->first, *(yysemantic_stack_[(10) - (9)].opentermMultiset)->second, 0, *p, NULL, (yysemantic_stack_[(10) - (3)].termVariable_adress), *(yysemantic_stack_[(10) - (10)].termVariablesAdresses), (yysemantic_stack_[(10) - (1)].compartmentType));
  delete (yysemantic_stack_[(10) - (2)].patternsMultiset);
  delete (yysemantic_stack_[(10) - (9)].opentermMultiset);
  delete p;
}
    break;

  case 35:

/* Line 698 of lalr1.cc  */
#line 398 "parser.yy"
    {		 
  (yyval.ruleNode) = new Rule(*(yysemantic_stack_[(11) - (2)].patternsMultiset)->first, *(yysemantic_stack_[(11) - (2)].patternsMultiset)->second, *(yysemantic_stack_[(11) - (10)].opentermMultiset)->first, *(yysemantic_stack_[(11) - (10)].opentermMultiset)->second, (yysemantic_stack_[(11) - (6)].multiplicity), *(yysemantic_stack_[(11) - (7)].parametersList), NULL, (yysemantic_stack_[(11) - (3)].termVariable_adress), *(yysemantic_stack_[(11) - (11)].termVariablesAdresses), (yysemantic_stack_[(11) - (1)].compartmentType));
  delete (yysemantic_stack_[(11) - (2)].patternsMultiset);
  delete (yysemantic_stack_[(11) - (10)].opentermMultiset);
  delete (yysemantic_stack_[(11) - (7)].parametersList);
}
    break;

  case 36:

/* Line 698 of lalr1.cc  */
#line 404 "parser.yy"
    {
  (yyval.ruleNode) = new Rule(*(yysemantic_stack_[(13) - (2)].patternsMultiset)->first, *(yysemantic_stack_[(13) - (2)].patternsMultiset)->second, *(yysemantic_stack_[(13) - (12)].opentermMultiset)->first, *(yysemantic_stack_[(13) - (12)].opentermMultiset)->second, (yysemantic_stack_[(13) - (7)].multiplicity), *(yysemantic_stack_[(13) - (8)].parametersList), (yysemantic_stack_[(13) - (9)].speciesOccsList), (yysemantic_stack_[(13) - (3)].termVariable_adress), *(yysemantic_stack_[(13) - (13)].termVariablesAdresses), (yysemantic_stack_[(13) - (1)].compartmentType));
  delete (yysemantic_stack_[(13) - (2)].patternsMultiset);
  delete (yysemantic_stack_[(13) - (12)].opentermMultiset);
  delete (yysemantic_stack_[(13) - (8)].parametersList);
}
    break;

  case 37:

/* Line 698 of lalr1.cc  */
#line 412 "parser.yy"
    {
  if((yysemantic_stack_[(3) - (2)].ruleNode)->is_biochemical) {
    (yysemantic_stack_[(3) - (2)].ruleNode)->ode_index = ode_offset;
    ode_offset += (yysemantic_stack_[(3) - (2)].ruleNode)->parameters.size();
  }
  (yysemantic_stack_[(3) - (1)].rulesList)->push_back((yysemantic_stack_[(3) - (2)].ruleNode));
  //reset variables symbol-table
  variable_table.clear();
  last_variable_free = VARIABLE_ADRESS_RESERVED + 1;
 }
    break;

  case 38:

/* Line 698 of lalr1.cc  */
#line 422 "parser.yy"
    {
  vector<Rule *> *v = new vector<Rule *>;
  if((yysemantic_stack_[(2) - (1)].ruleNode)->is_biochemical) {
    (yysemantic_stack_[(2) - (1)].ruleNode)->ode_index = ode_offset;
    ode_offset += (yysemantic_stack_[(2) - (1)].ruleNode)->parameters.size();
  }
  v->push_back((yysemantic_stack_[(2) - (1)].ruleNode));
  (yyval.rulesList) = v;
  //reset variables symbol-table
  variable_table.clear(); //reset variables symbol-table
  last_variable_free = VARIABLE_ADRESS_RESERVED + 1;
 }
    break;

  case 39:

/* Line 698 of lalr1.cc  */
#line 435 "parser.yy"
    {
  string *t = new string((yysemantic_stack_[(4) - (1)].qstring)->substr(1, (yysemantic_stack_[(4) - (1)].qstring)->size() - 2)); //cut quotes
  (yyval.monitorNode) = new Monitor(*t, (yysemantic_stack_[(4) - (4)].patternsMultiset)->first, (yysemantic_stack_[(4) - (4)].patternsMultiset)->second, (yysemantic_stack_[(4) - (3)].compartmentType));
  delete (yysemantic_stack_[(4) - (4)].patternsMultiset);
  delete (yysemantic_stack_[(4) - (1)].qstring);
  //reset variables symbol-table
  variable_table.clear(); //reset variables symbol-table
  last_variable_free = VARIABLE_ADRESS_RESERVED + 1;
 }
    break;

  case 40:

/* Line 698 of lalr1.cc  */
#line 445 "parser.yy"
    {
  (yysemantic_stack_[(3) - (1)].monitorsList)->push_back((yysemantic_stack_[(3) - (2)].monitorNode));
 }
    break;

  case 41:

/* Line 698 of lalr1.cc  */
#line 448 "parser.yy"
    {
  vector<Monitor *> *v = new vector<Monitor *>;
  v->push_back((yysemantic_stack_[(2) - (1)].monitorNode));
  (yyval.monitorsList) = v;
 }
    break;

  case 42:

/* Line 698 of lalr1.cc  */
#line 459 "parser.yy"
    {
  //print_symbol_table();
  /*
  for(unsigned int i=0; i<$6->size(); ++i)
    cerr << *($6->at(i)) << endl;
    */
  string *title = new string((yysemantic_stack_[(10) - (2)].qstring)->substr(1, (yysemantic_stack_[(10) - (2)].qstring)->size() - 2)); //cut quotes
  (yyval.modelNode) = new Model(*title, *(yysemantic_stack_[(10) - (4)].alphabet), *(yysemantic_stack_[(10) - (6)].rulesList), ode_offset, *(yysemantic_stack_[(10) - (8)].termsMultiset)->first, *(yysemantic_stack_[(10) - (8)].termsMultiset)->second, *(yysemantic_stack_[(10) - (10)].monitorsList));
  delete (yysemantic_stack_[(10) - (2)].qstring);
  delete (yysemantic_stack_[(10) - (8)].termsMultiset);
}
    break;

  case 44:

/* Line 698 of lalr1.cc  */
#line 472 "parser.yy"
    { driver.model = (yysemantic_stack_[(3) - (2)].modelNode); }
    break;



/* Line 698 of lalr1.cc  */
#line 963 "parser.cpp"
	default:
          break;
      }
    /* User semantic actions sometimes alter yychar, and that requires
       that yytoken be updated with the new translation.  We take the
       approach of translating immediately before every use of yytoken.
       One alternative is translating here after every semantic action,
       but that translation would be missed if the semantic action
       invokes YYABORT, YYACCEPT, or YYERROR immediately after altering
       yychar.  In the case of YYABORT or YYACCEPT, an incorrect
       destructor might then be invoked immediately.  In the case of
       YYERROR, subsequent parser actions might lead to an incorrect
       destructor call or verbose syntax error message before the
       lookahead is translated.  */
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
    /* Make sure we have latest lookahead translation.  See comments at
       user semantic actions for why this is necessary.  */
    yytoken = yytranslate_ (yychar);

    /* If not already recovering from an error, report this error.  */
    if (!yyerrstatus_)
      {
	++yynerrs_;
	if (yychar == yyempty_)
	  yytoken = yyempty_;
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
	if (!yy_pact_value_is_default_ (yyn))
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
      {
        /* Make sure we have latest lookahead translation.  See comments
           at user semantic actions for why this is necessary.  */
        yytoken = yytranslate_ (yychar);
        yydestruct_ ("Cleanup: discarding lookahead", yytoken, &yylval,
                     &yylloc);
      }

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
  Parser::yysyntax_error_ (int yystate, int yytoken)
  {
    std::string yyres;
    // Number of reported tokens (one for the "unexpected", one per
    // "expected").
    size_t yycount = 0;
    // Its maximum.
    enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
    // Arguments of yyformat.
    char const *yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];

    /* There are many possibilities here to consider:
       - If this state is a consistent state with a default action, then
         the only way this function was invoked is if the default action
         is an error action.  In that case, don't check for expected
         tokens because there are none.
       - The only way there can be no lookahead present (in yytoken) is
         if this state is a consistent state with a default action.
         Thus, detecting the absence of a lookahead is sufficient to
         determine that there is no unexpected or expected token to
         report.  In that case, just report a simple "syntax error".
       - Don't assume there isn't a lookahead just because this state is
         a consistent state with a default action.  There might have
         been a previous inconsistent state, consistent state with a
         non-default action, or user semantic action that manipulated
         yychar.
       - Of course, the expected token list depends on states to have
         correct lookahead information, and it depends on the parser not
         to perform extra reductions after fetching a lookahead from the
         scanner and before detecting a syntax error.  Thus, state
         merging (from LALR or IELR) and default reductions corrupt the
         expected token list.  However, the list is correct for
         canonical LR with one exception: it will still contain any
         token that will not be accepted due to an error action in a
         later state.
    */
    if (yytoken != yyempty_)
      {
        yyarg[yycount++] = yytname_[yytoken];
        int yyn = yypact_[yystate];
        if (!yy_pact_value_is_default_ (yyn))
          {
            /* Start YYX at -YYN if negative to avoid negative indexes in
               YYCHECK.  In other words, skip the first -YYN actions for
               this state because they are default actions.  */
            int yyxbegin = yyn < 0 ? -yyn : 0;
            /* Stay within bounds of both yycheck and yytname.  */
            int yychecklim = yylast_ - yyn + 1;
            int yyxend = yychecklim < yyntokens_ ? yychecklim : yyntokens_;
            for (int yyx = yyxbegin; yyx < yyxend; ++yyx)
              if (yycheck_[yyx + yyn] == yyx && yyx != yyterror_
                  && !yy_table_value_is_error_ (yytable_[yyx + yyn]))
                {
                  if (yycount == YYERROR_VERBOSE_ARGS_MAXIMUM)
                    {
                      yycount = 1;
                      break;
                    }
                  else
                    yyarg[yycount++] = yytname_[yyx];
                }
          }
      }

    char const* yyformat = YY_NULL;
    switch (yycount)
      {
#define YYCASE_(N, S)                         \
        case N:                               \
          yyformat = S;                       \
        break
        YYCASE_(0, YY_("syntax error"));
        YYCASE_(1, YY_("syntax error, unexpected %s"));
        YYCASE_(2, YY_("syntax error, unexpected %s, expecting %s"));
        YYCASE_(3, YY_("syntax error, unexpected %s, expecting %s or %s"));
        YYCASE_(4, YY_("syntax error, unexpected %s, expecting %s or %s or %s"));
        YYCASE_(5, YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s"));
#undef YYCASE_
      }

    // Argument number.
    size_t yyi = 0;
    for (char const* yyp = yyformat; *yyp; ++yyp)
      if (yyp[0] == '%' && yyp[1] == 's' && yyi < yycount)
        {
          yyres += yytnamerr_ (yyarg[yyi++]);
          ++yyp;
        }
      else
        yyres += *yyp;
    return yyres;
  }


  /* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
     STATE-NUM.  */
  const signed char Parser::yypact_ninf_ = -77;
  const signed char
  Parser::yypact_[] =
  {
       -77,    46,   -77,    12,    36,    53,   -77,   -77,    35,    56,
     -77,    59,    58,   -77,    75,    -1,    60,    23,   -77,   -77,
      77,   -77,    65,   -77,   -77,    59,   -77,    79,   -77,    20,
     -77,    70,   -77,    62,    72,    59,   -77,   -77,   -77,    48,
      55,    63,    81,    72,   -77,   -77,   -77,    66,    78,    64,
     -77,    67,    59,   -77,    87,     1,   -77,    64,   -77,   -77,
      -4,    90,   -77,   -77,   -77,    23,    57,    91,   -77,   -77,
      31,    19,    73,    83,    74,   -77,    43,   -77,   -77,   -77,
      92,    43,    59,   -77,    82,   -77,   -77,    82,   -77,   -77,
      43,    61,    82,    39,   -77,   -77,    43,     3,   -77
  };

  /* YYDEFACT[S] -- default reduction number in state S.  Performed when
     YYTABLE doesn't specify something else to do.  Zero means the
     default is an error.  */
  const unsigned char
  Parser::yydefact_[] =
  {
        43,     0,     1,     0,     0,     0,    44,     3,     4,     0,
       2,     0,     0,    23,     0,     0,     0,     0,    38,    13,
       0,     9,     0,     5,    14,     0,    21,     0,    22,     0,
      37,     0,     8,     0,     0,     0,    11,    12,     6,     0,
       0,     0,     0,    42,     8,    15,     7,     0,     0,    29,
      28,     0,     0,    41,     0,     0,    23,     0,    29,    30,
       0,     0,    23,    40,    13,     0,    33,     0,    31,    27,
      39,     0,     0,    33,     0,    27,    17,    10,    20,    32,
       0,    17,     0,    25,    34,    26,    27,    35,     8,    16,
      17,    19,    36,     0,    27,    18,    17,     0,    24
  };

  /* YYPGOTO[NTERM-NUM].  */
  const signed char
  Parser::yypgoto_[] =
  {
       -77,   -77,   -77,   -17,   -41,   -24,   -77,    37,   -15,     7,
     -76,   -77,   -77,   -46,   -77,   -69,   -36,    45,    30,    89,
     -77,    68,   -77,   -77,   -77
  };

  /* YYDEFGOTO[NTERM-NUM].  */
  const signed char
  Parser::yydefgoto_[] =
  {
        -1,     8,     9,    83,    39,    13,    37,    29,    89,    47,
      84,    93,    28,    17,    85,    76,    59,    60,    74,    14,
      15,    42,    43,     4,     1
  };

  /* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
     positive, shift that token.  If negative, reduce the rule which
     number is the opposite.  If YYTABLE_NINF_, syntax error.  */
  const signed char Parser::yytable_ninf_ = -1;
  const unsigned char
  Parser::yytable_[] =
  {
        26,    32,    27,    55,    51,    87,    81,    19,    58,    50,
      65,    44,    36,    22,    92,    23,    70,    90,    12,    24,
      97,    67,    46,    64,    68,    96,    98,     5,    62,    34,
      68,    22,    22,    23,    23,    22,     6,    23,    46,    24,
      35,    35,    77,    22,    25,    23,     2,    91,    26,    10,
      72,     3,    25,    26,    36,    22,    45,    23,    88,     7,
      22,    94,    23,    11,    82,    45,    48,    49,    50,    58,
      50,    73,    16,    22,    46,    23,    58,    50,    12,    18,
      21,    30,    33,    31,    38,    53,    40,    41,    56,    52,
      57,    63,    61,    69,    75,    86,    78,    73,    24,    80,
      95,    71,    66,    79,    20,     0,     0,     0,     0,     0,
       0,    54
  };

  /* YYCHECK.  */
  const signed char
  Parser::yycheck_[] =
  {
        17,    25,    17,    44,    40,    81,    75,     8,    12,    13,
      56,    35,    29,    12,    90,    14,    62,    86,    19,    16,
      96,    25,    39,    22,    60,    94,    23,    15,    52,     9,
      66,    12,    12,    14,    14,    12,     0,    14,    55,    16,
      21,    21,    23,    12,    21,    14,     0,    88,    65,    14,
      65,     5,    21,    70,    71,    12,    17,    14,    82,     6,
      12,    22,    14,     7,    21,    17,    11,    12,    13,    12,
      13,    14,    14,    12,    91,    14,    12,    13,    19,     4,
      20,     4,     3,    18,    14,     4,    24,    15,    22,    26,
      12,     4,    25,     3,     3,     3,    23,    14,    16,    25,
      93,    64,    57,    73,    15,    -1,    -1,    -1,    -1,    -1,
      -1,    43
  };

  /* STOS_[STATE-NUM] -- The (internal number of the) accessing
     symbol of state STATE-NUM.  */
  const unsigned char
  Parser::yystos_[] =
  {
         0,    51,     0,     5,    50,    15,     0,     6,    28,    29,
      14,     7,    19,    32,    46,    47,    14,    40,     4,     8,
      46,    20,    12,    14,    16,    21,    30,    35,    39,    34,
       4,    18,    32,     3,     9,    21,    30,    33,    14,    31,
      24,    15,    48,    49,    32,    17,    30,    36,    11,    12,
      13,    43,    26,     4,    48,    31,    22,    12,    12,    43,
      44,    25,    32,     4,    22,    40,    44,    25,    43,     3,
      40,    34,    35,    14,    45,     3,    42,    23,    23,    45,
      25,    42,    21,    30,    37,    41,     3,    37,    32,    35,
      42,    31,    37,    38,    22,    36,    42,    37,    23
  };

#if YYDEBUG
  /* TOKEN_NUMBER_[YYLEX-NUM] -- Internal symbol number corresponding
     to YYLEX-NUM.  */
  const unsigned short int
  Parser::yytoken_number_[] =
  {
         0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,    42,   123,
     125,    40,   124,    41,    91,    93,    58
  };
#endif

  /* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
  const unsigned char
  Parser::yyr1_[] =
  {
         0,    27,    28,    28,    29,    30,    30,    31,    31,    32,
      33,    34,    34,    34,    35,    36,    37,    37,    38,    38,
      39,    40,    40,    40,    41,    42,    42,    42,    43,    43,
      44,    44,    45,    45,    46,    46,    46,    47,    47,    48,
      49,    49,    50,    51,    51
  };

  /* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
  const unsigned char
  Parser::yyr2_[] =
  {
         0,     2,     2,     0,     1,     1,     3,     2,     0,     3,
       6,     2,     2,     0,     1,     1,     2,     0,     2,     0,
       8,     2,     2,     0,     8,     2,     2,     0,     1,     1,
       1,     2,     2,     0,    10,    11,    13,     3,     2,     4,
       3,     2,    10,     0,     3
  };

#if YYDEBUG || YYERROR_VERBOSE || YYTOKEN_TABLE
  /* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
     First, the terminals, then, starting at \a yyntokens_, nonterminals.  */
  const char*
  const Parser::yytname_[] =
  {
    "\"end of file\"", "error", "$undefined", "\">>>\"", "\"%%\"",
  "\"%model\"", "\"%alphabet\"", "\"%rules\"", "\"%term\"",
  "\"%monitors\"", "\"%end\"", "\"%F\"", "\"integer\"", "\"double\"",
  "\"atom_label\"", "\"qstring\"", "\"tvar\"", "\"wvar\"", "'*'", "'{'",
  "'}'", "'('", "'|'", "')'", "'['", "']'", "':'", "$accept", "alphabet_",
  "alphabet", "atom", "atoms_multiset", "compartment_type", "compartment",
  "terms_multiset", "tvar_adress", "wvar_adress", "tvar_multiset",
  "wvar_multiset", "pcompartment", "patterns_multiset", "ocompartment",
  "openterm_multiset", "parameter", "parameters", "occs", "rule",
  "rules_list", "monitor", "monitors_list", "model", "start", YY_NULL
  };
#endif

#if YYDEBUG
  /* YYRHS -- A `-1'-separated list of the rules' RHS.  */
  const Parser::rhs_number_type
  Parser::yyrhs_[] =
  {
        51,     0,    -1,    28,    14,    -1,    -1,    28,    -1,    14,
      -1,    12,    18,    14,    -1,    31,    30,    -1,    -1,    19,
      14,    20,    -1,    21,    32,    31,    22,    34,    23,    -1,
      34,    30,    -1,    34,    33,    -1,    -1,    16,    -1,    17,
      -1,    37,    35,    -1,    -1,    38,    36,    -1,    -1,    21,
      32,    31,    36,    22,    40,    35,    23,    -1,    40,    30,
      -1,    40,    39,    -1,    -1,    21,    32,    31,    38,    22,
      42,    37,    23,    -1,    42,    30,    -1,    42,    41,    -1,
      -1,    13,    -1,    12,    -1,    43,    -1,    44,    43,    -1,
      14,    45,    -1,    -1,    32,    40,    35,     3,    24,    43,
      25,     3,    42,    37,    -1,    32,    40,    35,     3,    24,
      12,    44,    25,     3,    42,    37,    -1,    32,    40,    35,
       3,    24,    11,    12,    44,    45,    25,     3,    42,    37,
      -1,    47,    46,     4,    -1,    46,     4,    -1,    15,    26,
      32,    40,    -1,    49,    48,     4,    -1,    48,     4,    -1,
       5,    15,     6,    29,     7,    47,     8,    34,     9,    49,
      -1,    -1,    51,    50,     0,    -1
  };

  /* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
     YYRHS.  */
  const unsigned char
  Parser::yyprhs_[] =
  {
         0,     0,     3,     6,     7,     9,    11,    15,    18,    19,
      23,    30,    33,    36,    37,    39,    41,    44,    45,    48,
      49,    58,    61,    64,    65,    74,    77,    80,    81,    83,
      85,    87,    90,    93,    94,   105,   117,   131,   135,   138,
     143,   147,   150,   161,   162
  };

  /* YYRLINE[YYN] -- Source line where rule number YYN was defined.  */
  const unsigned short int
  Parser::yyrline_[] =
  {
         0,   182,   182,   186,   191,   202,   207,   218,   222,   227,
     244,   255,   260,   264,   273,   280,   287,   288,   290,   291,
     305,   318,   323,   327,   348,   353,   357,   360,   367,   370,
     374,   377,   381,   386,   391,   398,   404,   412,   422,   435,
     445,   448,   454,   471,   472
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
      21,    23,    18,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    26,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,    24,     2,    25,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    19,    22,    20,     2,     2,     2,     2,
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
      15,    16,    17
    };
    if ((unsigned int) t <= yyuser_token_number_max_)
      return translate_table[t];
    else
      return yyundef_token_;
  }

  const int Parser::yyeof_ = 0;
  const int Parser::yylast_ = 111;
  const int Parser::yynnts_ = 25;
  const int Parser::yyempty_ = -2;
  const int Parser::yyfinal_ = 2;
  const int Parser::yyterror_ = 1;
  const int Parser::yyerrcode_ = 256;
  const int Parser::yyntokens_ = 27;

  const unsigned int Parser::yyuser_token_number_max_ = 272;
  const Parser::token_number_type Parser::yyundef_token_ = 2;


} // scwc

/* Line 1144 of lalr1.cc  */
#line 1533 "parser.cpp"

/* Line 1145 of lalr1.cc  */
#line 476 "parser.yy"
 /*** Additional Code ***/

void scwc::Parser::error(const Parser::location_type& l,
			 const std::string& m)
{
  driver.error(l, m);
}

