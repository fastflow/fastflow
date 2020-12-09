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
 * Example of heterogeneous all-to-all with complex building-blocks nesting.
 * 
 *
 *                                               |  |----->| WR - WRoutH | ---------> |
 *     |-> WL -> WLoutH ->|                      |  |                                 |
 *     |                  |                      |  |                                 |
 *     |                  |->| WL11inH - WL11 |->|  |                                 |
 *     |-> WL -> WLoutH ->|                      |  |-> WR - WRoutH |                 |
 *     |                                         |  |               |                 |
 * Sc->|                                         |->|               |-> WR1 -> WR11 ->|-> Sk
 *     |                                         |  |-> WR - WRoutH |                 |
 *     |                                         |  |                                 |
 *     | ------> | WL - WL11 | ----------------->|  
 *
 */
/*
 *
 * Author: Massimo Torquati
 */


#include <iostream>
#include <ff/ff.hpp>

using namespace ff;

struct S_t {    
    int t;
    float f;
};

struct Source: ff_monode_t<S_t> {
	Source(long ntasks): ntasks(ntasks) {}    
	S_t* svc(S_t*) {
        for (long i=1;i<=ntasks;++i){
            S_t* p = new S_t;
            p->t = i;
            p->f = p->t*1.0;
            ff_send_out_to(p,0);
        }
		return EOS; 
	}
    long ntasks;
};
struct Sink: ff_minode_t<S_t> {
    S_t* svc(S_t* in) {
        printf("in->t=%d from %ld\n", in->t, get_channel_id());
        return GO_ON;
    }    
};

struct WL: ff_node_t<S_t> {
    S_t* svc(S_t* in) {	return in; }
};
struct WLoutH: ff_monode_t<S_t> {
    S_t* svc(S_t* in) { return in; }
};
struct WL1: ff_monode_t<S_t> {
    S_t* svc(S_t* in) { return in; }
};
struct WL11inH: ff_minode_t<S_t> {
    S_t* svc(S_t* in) { return in; }
};	
struct WL11: ff_monode_t<S_t> {
    S_t* svc(S_t* in) {	return in; }
};

struct WR: ff_minode_t<S_t> {
    S_t* svc(S_t* in) { return in; }
};
struct WRoutH: ff_monode_t<S_t> {
    S_t* svc(S_t* in) { return in; }
};
struct WR1: ff_minode_t<S_t> {
    S_t* svc(S_t* in) { return in; }
};
struct WR11: ff_node_t<S_t> {
    S_t* svc(S_t* in) {	return in; }
};


int main(int argc, char* argv[]) {
	std::vector<ff_node*> L;
	std::vector<ff_node*> R;

	ff_pipeline pipeMain;
	Source Sc(100);
	ff_a2a a2aCentral;
	Sink Sk;


	// --- these are the edge nodes of the graph -----
	std::vector<WL*> edgesWL_in;
	edgesWL_in.push_back(new WL);
	edgesWL_in.push_back(new WL);
	auto edgesWL_out = new ff_comb_t(new WL11inH, new WL11);;

	auto    edgesWL_inout = new ff_comb_t(new WL, new WL11);;

	auto edgesWR_inout = new ff_comb_t(new WR, new WRoutH);
	std::vector<ff_comb_t<S_t,S_t,S_t>*> edgesWR_in;
	edgesWR_in.push_back(new ff_comb_t(new WR, new WRoutH));
	edgesWR_in.push_back(new ff_comb_t(new WR, new WRoutH));

	WR11* edgesWR_out = new WR11;
	// ----------------------------------------------
	
	ff_pipeline LeftUp;
	ff_a2a a2a_LeftUp;
	ff_pipeline pipe1;
	pipe1.add_stage(edgesWL_in[0]);
	pipe1.add_stage(new WLoutH);

	ff_pipeline pipe2;
	pipe2.add_stage(edgesWL_in[1]);
	pipe2.add_stage(new WLoutH);

	L.push_back(&pipe1);
	L.push_back(&pipe2);

	R.push_back(edgesWL_out);
	a2a_LeftUp.add_firstset(L);
	a2a_LeftUp.add_secondset(R);
	LeftUp.add_stage(&a2a_LeftUp);
	
	ff_pipeline LeftDown;
	LeftDown.add_stage(edgesWL_inout);

	// --------------------------
	L.clear();
	R.clear();

	ff_pipeline RightUp;
	RightUp.add_stage(edgesWR_inout);

	
	ff_pipeline RightDown;
	ff_a2a a2a_RightDown;
	L.push_back(edgesWR_in[0]);
	L.push_back(edgesWR_in[1]);
	ff_pipeline pipe3;
	pipe3.add_stage(new WR1);
	pipe3.add_stage(edgesWR_out);
	a2a_RightDown.add_firstset(L);
	R.push_back(&pipe3);
	a2a_RightDown.add_secondset(R);
	RightDown.add_stage(&a2a_RightDown);


	// ------------------
	L.clear();
	R.clear();
	L.push_back(&LeftUp);
	L.push_back(&LeftDown); 

	R.push_back(&RightUp); 
	R.push_back(&RightDown);
	a2aCentral.add_firstset(L);
	a2aCentral.add_secondset(R);

	pipeMain.add_stage(&Sc);
	pipeMain.add_stage(&a2aCentral);
	pipeMain.add_stage(&Sk);

	std::cout << "Num Threads= " << pipeMain.numThreads() << "\n";
    pipeMain.run_and_wait_end();
    
    return 0;
}



