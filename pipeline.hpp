/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */
#ifndef _FF_PIPELINE_HPP_
#define _FF_PIPELINE_HPP_
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

#include <vector>
#include <node.hpp>

namespace ff {

class ff_pipeline: public ff_node {
public:
    enum { DEF_IN_BUFF_ENTRIES=512, DEF_OUT_BUFF_ENTRIES=(DEF_IN_BUFF_ENTRIES+128)};

    ff_pipeline(int in_buffer_entries=DEF_IN_BUFF_ENTRIES,
                int out_buffer_entries=DEF_OUT_BUFF_ENTRIES):
        in_buffer_entries(in_buffer_entries),
        out_buffer_entries(out_buffer_entries) {}

    int add_stage(ff_node * s) {        
        nodes_list.push_back(s);
        return 0;
    }

    // the last stage output queue will be connected 
    // to the first stage input queue (feedback channel).
    int wrap_around() {
        if (nodes_list.size()<2) {
            error("PIPE, too few pipeline nodes\n");
            return -1;
        }
        if (create_input_buffer(out_buffer_entries)<0)
            return -1;
        
        if (set_output_buffer(get_in_buffer())<0)
            return -1;

        nodes_list[0]->skip1pop = true;

        return 0;
    }

    int run(bool skip_init=false) {
        int nstages=nodes_list.size();

        if (!skip_init) {            
            // set the initial value for the barrier 
            Barrier::instance()->barrier(cardinality());        
        }
        
        for(int i=1;i<nstages;++i) {
            if (nodes_list[i]->create_input_buffer(in_buffer_entries)<0) {
                error("PIPE, creating input buffer for node %d\n", i);
                return -1;
            }
        }
        
        for(int i=0;i<(nstages-1);++i) {
            if (nodes_list[i]->set_output_buffer(nodes_list[i+1]->get_in_buffer())<0) {
                error("PIPE, setting output buffer to node %d\n", i);
                return -1;
            }
        }
        
        for(int i=0;i<nstages;++i) {
            nodes_list[i]->set_id(i);
            if (nodes_list[i]->run(true)<0) {
                error("ERROR: PIPE, running stage %d\n", i);
                return -1;
            }
        }

        return 0;
    }

    int wait(/* timeval */ ) {
        int ret=0;
        for(unsigned int i=0;i<nodes_list.size();++i)
            if (nodes_list[i]->wait()<0) {
                error("PIPE, waiting stage thread, id = %d\n",nodes_list[i]->get_my_id());
                ret = -1;
            } 

        return ret;
    }

    int wait_freezing(/* timeval */ ) {
        int ret=0;
        for(unsigned int i=0;i<nodes_list.size();++i)
            if (nodes_list[i]->wait_freezing()<0) {
                error("PIPE, waiting freezing of stage thread, id = %d\n",
                      nodes_list[i]->get_my_id());
                ret = -1;
            } 
        
        return ret;
    } 

    void freeze() {
        for(unsigned int i=0;i<nodes_list.size();++i) nodes_list[i]->freeze();
    }

    void thaw() {
        for(unsigned int i=0;i<nodes_list.size();++i) nodes_list[i]->thaw();
    }
    
    int run_and_wait_end() {
        if (run()<0) return -1;           
        wait();
        return 0;
    }

    int   cardinality() const { 
        int card=0;
        for(unsigned int i=0;i<nodes_list.size();++i) 
            card += nodes_list[i]->cardinality();

        return card;
    }

    double ffTime() {
        return diffmsec(nodes_list[nodes_list.size()-1]->getstoptime(),
                        nodes_list[0]->getstarttime());
    }
    
#if defined(TRACE_FASTFLOW)
    void ffStats(std::ostream & out) { 
        out << "--- pipeline:\n";
        for(unsigned int i=0;i<nodes_list.size();++i)
            nodes_list[i]->ffStats(out);
    }
#else
    void ffStats(std::ostream & out) { 
        out << "FastFlow trace not enabled\n";
    }
#endif

protected:

    // ff_node interface
    void* svc(void * task) { return NULL; }
    int   svc_init() { return -1; };
    void  svc_end()  {}
    
    int create_input_buffer(int nentries) { 
        if (in) return -1;
        if (nodes_list[0]->create_input_buffer(nentries)<0) {
            error("PIPE, creating input buffer for node 0\n");
            return -1;
        }
        ff_node::set_input_buffer(nodes_list[0]->get_in_buffer());
        return 0;
    }
    
    int create_output_buffer(int nentries) {
        int last = nodes_list.size()-1;
        if (!last) return -1;

        if (nodes_list[last]->create_output_buffer(nentries)<0) {
            error("PIPE, creating output buffer for node %d\n",last);
            return -1;
        }
        ff_node::set_output_buffer(nodes_list[last]->get_out_buffer());
        return 0;
    }

    int set_output_buffer(SWSR_Ptr_Buffer * const o) {
        int last = nodes_list.size()-1;
        if (!last) return -1;

        if (nodes_list[last]->set_output_buffer(o)<0) {
            error("PIPE, setting output buffer for node %d\n",last);
            return -1;
        }
        return 0;
    }

private:
    int in_buffer_entries;
    int out_buffer_entries;
    std::vector<ff_node *> nodes_list;
};





} // namespace ff

#endif /* _FF_PIPELINE_HPP_ */
