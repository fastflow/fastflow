/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */
/* ***************************************************************************
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License version 2 as 
 *  published by the Free Software Foundation.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 *
 *  As a special exception, you may use this file as part of a free software
 *  library without restriction.  Specifically, if other files instantiate
 *  templates or use macros or inline functions from this file, or you compile
 *  this file and link it with other files to produce an executable, this
 *  file does not by itself cause the resulting executable to be covered by
 *  the GNU General Public License.  This exception does not however
 *  invalidate any other reasons why the executable file might be covered by
 *  the GNU General Public License.
 *
 ****************************************************************************
 */

/* 
 * Testing the change_node method for the pipeline, farm all-to-all and combine.
 * 
 */

/*
 * Author: Massimo Torquati
 */ 

#include <ff/ff.hpp>

using namespace ff;

struct Source: ff_monode_t<long> {
	Source(long ntasks): ntasks(ntasks) {}    
	long* svc(long*) {
        for (long i=1;i<=ntasks;++i){
            ff_send_out((long*)i);
        }
		return EOS; 
	}
    long ntasks;
};
struct miNode:ff_minode_t<long> {
    long* svc(long* in) {
		return in;
	}		
};
struct moNode:ff_monode_t<long> {
    long* svc(long* in) {
		return in;
	}		
};
struct Sink: ff_minode_t<long> {
    long* svc(long* in) {
        printf("in=%ld\n", (long)in);
        return GO_ON;
    }    
};



int main() {
	Source source(10);
	Sink   sink;

	miNode* uno = new miNode;
	moNode* due = new moNode;
	ff_pipeline pipe0;
	pipe0.add_stage(uno, true);
	pipe0.add_stage(due, true);
		
	ff_a2a a2a;
	std::vector<ff_node*> W1, W2;
	miNode* tre     = new miNode;
	moNode* quattro = new moNode;
	ff_comb *c1   = new ff_comb(tre, quattro, true, true);
	W1.push_back(c1);
	miNode* cinque  = new miNode;
	moNode* sei     = new moNode;
	ff_comb *c2   = new ff_comb(cinque, sei, true, true);
	W1.push_back(c2);
	a2a.add_firstset(W1, 0, true);
        
	miNode* sette  = new miNode;
	moNode* otto   = new moNode;
	ff_comb *c3   = new ff_comb(sette, otto, true, true);
	W2.push_back(c3);	
	a2a.add_secondset(W2,true);

    ff_farm farm;
    moNode* nove  = new moNode;
    moNode* dieci = new moNode;
    farm.add_workers({nove, dieci});
    farm.cleanup_workers();
    
	
	ff_pipeline pipeMain;
	pipeMain.add_stage(&source);
	pipeMain.add_stage(&pipe0);
	pipeMain.add_stage(&a2a);
    pipeMain.add_stage(&farm);
	pipeMain.add_stage(&sink);

	// changing some nodes
	ff_node* c = getBB(&pipeMain, otto);
	assert(c==c3);
	(reinterpret_cast<ff_comb*>(c))->change_node(otto, new miNode, true);
    
	ff_node* t1 = getBB(&pipeMain, tre);
	assert(t1 == c1);
	(reinterpret_cast<ff_comb*>(t1))->change_node(tre, new moNode, true);

	ff_node* t2 = getBB(&pipeMain, cinque);
	assert(t2 == c2);
	(reinterpret_cast<ff_comb*>(t2))->change_node(cinque, new moNode, true);

    ff_node* t3 = getBB(&pipeMain, c2);
    assert(t3 == &a2a);
    (reinterpret_cast<ff_a2a*>(t3))->change_node(c2, new ff_comb(new moNode, new moNode, true, true), true, true);
    delete c2;

    ff_node* t4 = getBB(&pipeMain, nove);
    assert(t4 == &farm);
    (reinterpret_cast<ff_farm*>(t4))->change_node(nove, new ff_comb(new moNode, new moNode, true, true), true, true);
    (reinterpret_cast<ff_farm*>(t4))->change_node(dieci, new ff_comb(new moNode, new moNode, true, true), true, true);
    
	pipeMain.change_node(&source, new Source(100), true);

	// now run the modified pipeline
	if (pipeMain.run_and_wait_end()<0) {
		error("running pipe\n");
		return -1;
	}
	
	return 0;
}
