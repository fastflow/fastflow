%{/*** C/C++ Declarations ***/

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

  %}

/*** yacc/bison Declarations ***/

/* Require bison 2.4.1 or later */
%require "2.4.1"

/* add debug output code to generated parser. disable this for release
 * versions. */
%debug

 /* start symbol is named "start" */
%start start

 /* write out a header file containing the token defines */
%defines

 /* use newer C++ skeleton file */
%skeleton "lalr1.cc"

 /* namespace to enclose parser in */
%name-prefix="scwc"

			   /* set the parser's class identifier */
			   %define "parser_class_name" "Parser"

			   /* keep track of the current position within the input */
			   %locations
			   %initial-action
  {
    // initialize the initial location object
    @$.begin.filename = @$.end.filename = &driver.streamname;
  };

/* The driver is passed by reference to the parser and to the scanner. This
 * provides a simple but effective pure interface, not relying on global
 * variables. */
%parse-param { class Driver& driver }

/* verbose error messages */
%error-verbose

 /*** Change the grammar's tokens below ***/

%union {
  /*unsigned*/ long long multiplicity;
  double constantRate;
  std::string *atomLabel;
  unsigned int compartmentType;
  std::string *qstring;
  std::string *termVariable;
  //std::vector<std::string *> *termVariablesMultiset;
  std::string *wrapVariable;
  //std::vector<std::string *> *wrapVariablesMultiset;

  unsigned int termVariable_adress;
  std::vector<unsigned int> *termVariablesAdresses;
  unsigned int wrapVariable_adress;
  std::vector<unsigned int> *wrapVariablesAdresses;

  std::vector<std::string> *alphabet;
  std::pair<unsigned int, /*unsigned*/ long long> *atomNode;
  class Species *atomsMultiset;
  class Compartment *compartmentNode;
  std::pair<class Species *, std::vector<class Compartment *> *> *termsMultiset;
  class PCompartment *pcompartmentNode;
  std::pair<class Species *, std::vector<class PCompartment *> *> *patternsMultiset;
  class OCompartment *ocompartmentNode;
  std::pair<class Species *, std::vector<class OCompartment *> *> *opentermMultiset;
  class Rule *ruleNode;
  std::vector<class Rule *> *rulesList; 
  class Monitor *monitorNode;
  std::vector<class Monitor *> *monitorsList;
  class Model *modelNode;
  std::vector<double> *parametersList;
  std::vector<unsigned int> *speciesOccsList;
  int semantics;
}

%token                  		END 				0 "end of file"
//%token                		EOL         			"end of line"
%token                  		RSEP        			">>>"
%token                  		LSEP        			"%%"
%token                  		MODEL_      			"%model"
%token                  		ALPHABET_   			"%alphabet"
%token                  		RULES_      			"%rules"
%token                  		TERM_       			"%term"
%token                  		MONITORS_   			"%monitors"
%token                  		ENDF        			"%end"
%token					FUNCTION_			"%F"
%token <multiplicity>   		INTEGER     			"integer"
%token <constantRate>   		DOUBLE      			"double"
%token <atomLabel>      		ATOM_LABEL			"atom_label"
%token <qstring>        		QSTRING     			"qstring"

%token <termVariable>   		TVAR        			"tvar"
%token <wrapVariable>   		WVAR        			"wvar"


%type <alphabet>			alphabet
%type <alphabet>			alphabet_
%type <atomNode>			atom
%type <atomsMultiset>			atoms_multiset
%type <compartmentType>			compartment_type
%type <compartmentNode>			compartment
%type <termsMultiset>			terms_multiset
%type <pcompartmentNode>		pcompartment
%type <patternsMultiset>		patterns_multiset

%type <termVariable_adress>		tvar_adress
%type <wrapVariable_adress>		wvar_adress
//%type <termVariablesMultiset>		tvar_multiset
//%type <wrapVariablesMultiset>		wvar_multiset
%type <termVariablesAdresses>		tvar_multiset
%type <wrapVariablesAdresses>		wvar_multiset

%type <ocompartmentNode>		ocompartment
%type <opentermMultiset>		openterm_multiset
%type <ruleNode>			rule
%type <rulesList>			rules_list
%type <monitorNode>			monitor
%type <monitorsList>			monitors_list
%type <modelNode>			model
%type <constantRate>			parameter
%type <parametersList>			parameters
%type <speciesOccsList>			occs

 /*** Change the grammar's tokens above ***/

%{

#include "Driver.h"
#include "scanner.h"

  /* this "connects" the bison parser in the driver to the flex scanner class
   * object. it defines the yylex() function call to pull the next token from the
   * current lexer object of the driver context. */
#undef yylex
#define yylex driver.lexer->lex

  %}

%% /*** Grammar Rules ***/

 /*** Change the grammar rules below ***/

alphabet_ : alphabet_ ATOM_LABEL {
  $1->push_back(*$2);
  delete $2;
 }
| {
  //empty
  $$ = new vector<string>;
 }

alphabet : alphabet_ {
  reverse_symbol_table &rst = *$1;
  sort(rst.begin(), rst.end());
  for(symbol_adress i=0; i<rst.size(); i++) {
    atom_table.insert(pair<string, symbol_adress>(rst[i], i));
  }
  type_table.insert(pair<string, symbol_adress>(string("T"), TOP_LEVEL));
  type_table.insert(pair<string, symbol_adress>(string("ALL"), ANY_TYPE));
  //last_type_free = TOP_LEVEL + 1;
 }

atom : ATOM_LABEL {
  //singleton
  $$ = new pair<symbol_adress, multiplicityType>(atom_table[*$1], 1);
  delete $1;
 }
| INTEGER '*' ATOM_LABEL {
  //bag
  if($1 <= 0) {
    error(yyloc, string("Negative multiplicity"));
    YYERROR;
  } else {
    $$ = new pair<symbol_adress, multiplicityType>(atom_table[*$3], $1);
    delete $3;
  }
  }

atoms_multiset : atoms_multiset atom {
  $1->add($2->first, $2->second);
  delete $2;
 }
| {
  //empty
  $$ = new Species(atom_table.size());
 }

compartment_type : '{' ATOM_LABEL '}' {
  //compartment type-label
  map<string, symbol_adress>::iterator it;
  it = type_table.find(*$2);
  if(it != type_table.end()) {
    $$ = it->second;
  }
  else {
    $$ = (type_table[*$2] = last_type_free++);
  }
  delete $2;
 }



/*** TERM ***/

compartment : '(' compartment_type  atoms_multiset '|' terms_multiset ')' {
  //ground compartment
  $$ = new Compartment($3, $5->first, $5->second, $2);
  delete $5;
 }

/*
  term : atom { $$ = new pair<pair<symbol_adress, multiplcityType> *, Compartment *>(); }
  | compartment { $$ = $1; }
*/

terms_multiset : terms_multiset atom { 
  //species
  $1->first->add($2->first, $2->second);
  delete $2;
 }
| terms_multiset compartment {
  //compartment
  $1->second->push_back($2);
 }
| {
  //empty
  $$ = new pair<Species *, vector<Compartment *> *>(new Species(atom_table.size()), new vector<Compartment *>);
  }



/*** VARIABLES ***/

tvar_adress : TVAR {
  if(!variable_table.count(*$1))
	variable_table[*$1] = last_variable_free++;
  $$ = variable_table[*$1];
  delete $1;
}

wvar_adress : WVAR {
  if(!variable_table.count(*$1))
	variable_table[*$1] = last_variable_free++;
  $$ = variable_table[*$1];
  delete $1;	    
}

tvar_multiset : tvar_multiset tvar_adress { $1->push_back($2);}
| /* empty */ { $$ = new vector<variable_adress>; }

wvar_multiset : wvar_multiset wvar_adress { $1->push_back($2);}
| /* empty */ { $$ = new vector<variable_adress>; }


//tvar_multiset : tvar_multiset TVAR { $1->push_back($2);}
//| /* empty */ { $$ = new vector<string *>; }

//wvar_multiset : wvar_multiset WVAR { $1->push_back($2);}
//| /* empty */ { $$ = new vector<string *>; }



/*** PATTERN ***/

//pcompartment : '(' compartment_type atoms_multiset WVAR '|' patterns_multiset TVAR ')' {
pcompartment : '(' compartment_type atoms_multiset wvar_adress '|' patterns_multiset tvar_adress ')' {
  //non-empty-content compartment
  $$ = new PCompartment($3, $6->first, $6->second, $4, $7, $2);
  delete $6;
 }
/*
//| '(' compartment_type atoms_multiset WVAR '|' TVAR ')' {
| '(' compartment_type atoms_multiset wvar_adress '|' tvar_adress ')' {
  //empty-content compartment
  $$ = new PCompartment($3, new Species(atom_table.size()), new vector<PCompartment *>, $4, $6, $2);
  }
*/

patterns_multiset : patterns_multiset atom {
  //species
  $1->first->add($2->first, $2->second);
  delete $2;
 }
| patterns_multiset pcompartment {
  //compartment
  $1->second->push_back($2);
 }
| /*empty*/ {
  $$ = new pair<Species *, vector<PCompartment *> *>(new Species(atom_table.size()), new vector<PCompartment *>);
  }
/*
| atom {
  Species *species = new Species(atom_table.size());
  species->add($1->first, $1->second);
  delete $1;
  $$ = new pair<Species *, vector<PCompartment *> *>(species, new vector<PCompartment *>);
  }
| pcompartment {
  vector<PCompartment *> *cc = new vector<PCompartment *>;
  cc->push_back($1);
  $$ = new pair<Species *, vector<PCompartment *> *>(new Species(atom_table.size()), cc);
  }
*/



/*** OPEN TERM ***/

ocompartment : '(' compartment_type atoms_multiset wvar_multiset '|' openterm_multiset tvar_multiset ')' {
  $$ = new OCompartment($3, $6->first, $6->second, *$4, *$7, $2);
  delete $6;
 }

openterm_multiset : openterm_multiset atom {
  $1->first->add($2->first, $2->second);
  delete $2;
 }
| openterm_multiset ocompartment {
  $1->second->push_back($2);
 }
| /* empty */ {
  $$ = new pair<Species *, vector<OCompartment *> *>(new Species(atom_table.size()), new vector<OCompartment *>);
  }



/*** RULE, MONITOR, MODEL ***/
parameter: DOUBLE {
  $$ = $1;
}
| INTEGER {
  $$ = (double)$1;
}

parameters: parameter {
	    $$ = new vector<double>(1, $1);
	    }
| parameters parameter {
  $1->push_back($2);
}

occs : ATOM_LABEL occs {
  $2->push_back(atom_table[*$1]);
  $$ = $2;
}
| /* empty */
  {
  $$ = new vector<symbol_adress>();
  }

rule :
compartment_type patterns_multiset tvar_adress RSEP '[' parameter ']' RSEP openterm_multiset tvar_multiset {
  vector<double> *p = new vector<double>(1, $6);
  $$ = new Rule(*$2->first, *$2->second, *$9->first, *$9->second, 0, *p, NULL, $3, *$10, $1);
  delete $2;
  delete $9;
  delete p;
}
| compartment_type patterns_multiset tvar_adress RSEP '[' INTEGER parameters ']' RSEP openterm_multiset tvar_multiset {		 
  $$ = new Rule(*$2->first, *$2->second, *$10->first, *$10->second, $6, *$7, NULL, $3, *$11, $1);
  delete $2;
  delete $10;
  delete $7;
}
| compartment_type patterns_multiset tvar_adress RSEP '[' FUNCTION_ INTEGER parameters occs ']' RSEP openterm_multiset tvar_multiset {
  $$ = new Rule(*$2->first, *$2->second, *$12->first, *$12->second, $7, *$8, $9, $3, *$13, $1);
  delete $2;
  delete $12;
  delete $8;
}


rules_list : rules_list rule LSEP {
  if($2->is_biochemical) {
    $2->ode_index = ode_offset;
    ode_offset += $2->parameters.size();
  }
  $1->push_back($2);
  //reset variables symbol-table
  variable_table.clear();
  last_variable_free = VARIABLE_ADRESS_RESERVED + 1;
 }
| rule LSEP {
  vector<Rule *> *v = new vector<Rule *>;
  if($1->is_biochemical) {
    $1->ode_index = ode_offset;
    ode_offset += $1->parameters.size();
  }
  v->push_back($1);
  $$ = v;
  //reset variables symbol-table
  variable_table.clear(); //reset variables symbol-table
  last_variable_free = VARIABLE_ADRESS_RESERVED + 1;
 }

monitor : QSTRING ':' compartment_type patterns_multiset {
  string *t = new string($1->substr(1, $1->size() - 2)); //cut quotes
  $$ = new Monitor(*t, $4->first, $4->second, $3);
  delete $4;
  delete $1;
  //reset variables symbol-table
  variable_table.clear(); //reset variables symbol-table
  last_variable_free = VARIABLE_ADRESS_RESERVED + 1;
 }

monitors_list : monitors_list monitor LSEP{
  $1->push_back($2);
 }
| monitor LSEP {
  vector<Monitor *> *v = new vector<Monitor *>;
  v->push_back($1);
  $$ = v;
 }

model :	MODEL_ QSTRING
ALPHABET_ alphabet
RULES_ rules_list
TERM_ terms_multiset
MONITORS_ monitors_list
{
  //print_symbol_table();
  /*
  for(unsigned int i=0; i<$6->size(); ++i)
    cerr << *($6->at(i)) << endl;
    */
  string *title = new string($2->substr(1, $2->size() - 2)); //cut quotes
  $$ = new Model(*title, *$4, *$6, ode_offset, *$8->first, *$8->second, *$10);
  delete $2;
  delete $8;
}

start : /* empty */
| start model END { driver.model = $2; }

/*** Change the grammar rules above ***/

%% /*** Additional Code ***/

void scwc::Parser::error(const Parser::location_type& l,
			 const std::string& m)
{
  driver.error(l, m);
}
