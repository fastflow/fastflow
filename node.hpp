/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */
#ifndef _FF_NODE_HPP_
#define _FF_NODE_HPP_
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


#include <iostream>
#include <buffer.hpp>

namespace ff {


enum { FF_EOS=0xffffffff };


class ff_thread {

protected:
    ff_thread() {}
    virtual ~ff_thread() {}

    static void * thread_routine(void * arg) {
        ff_thread & obj = *(ff_thread *)arg;
        void * r = NULL;
        
        Barrier();        

        if (obj.svc_init(obj.get_args())<0)
            error("ff_thread, svc_init failed, thread exit!!!\n");
        else  r= obj.svc(NULL);
        obj.svc_end(r);

        pthread_exit(r);
    }

    virtual void * get_args() const { return NULL;}

public:
    virtual void* svc(void * task) = 0;
    virtual int   svc_init(void * args) { return 0; };
    virtual void  svc_end(void * result)  {}

    int spawn() {
        if (pthread_create(&th_handle, NULL, // FIX: attr management !
                           ff_thread::thread_routine, this) != 0) {
            perror("pthread_create");
            return -1;
        }
        return 0;
    }
   
    int wait() {
        pthread_join(th_handle, NULL);
        return 0;
    }

    pthread_t get_handle() const { return th_handle;}



private:
    pthread_t      th_handle;
    pthread_attr_t attr; // TODO
};
    


class ff_node: public ff_thread {

    template <typename w_t, typename lb_t, typename gt_t> friend class ff_farm;

public:
    virtual void* svc(void * task) = 0;
    virtual int   svc_init(void * args) { return 0; }
    virtual void  svc_end(void * result) {}

    int get_my_id() { return myid; };
    
    bool push(void * ptr) { return in->push(ptr);}
    bool pop(void ** ptr) { return ((out->pop(ptr)>0)?true:false);} 

    
protected:
    ff_node():in(0),out(0),myid(-1) {};

    virtual  ~ff_node() {
        if (in)  delete in;
        if (out) delete out;
    };

    bool get(void **ptr) { return ((in->pop(ptr)>0)?true:false);} 
    bool put(void * ptr) { return out->push(ptr); }

    int create_input_buffer(int nentries) {
        if (in) return -1;
        in = new SWSR_Ptr_Buffer(nentries);        
        if (!in) return -1;
        return (in->init()?0:-1);
    }
    
    int create_output_buffer(int nentries) {
        if (out) return -1;
        out = new SWSR_Ptr_Buffer(nentries);        
        if (!out) return -1;
        return (out->init()?0:-1);
    }
    
    void set_id(int id) { myid = id;}

private:
    SWSR_Ptr_Buffer * in;
    SWSR_Ptr_Buffer * out;
    int myid;
};


template<typename T>
class thWorker: public T {
    enum {TICKS2WAIT=1000};
protected:
    virtual void * get_args() const { return args;}

public:
    thWorker(void * args):args(args) {}

    bool push(void * task) {
        //register int cnt = 0;
        while (! T::put(task)) {
            // if (ch->thxcore>1) {
            // if (++cnt>PUSH_POP_CNT) { sched_yield(); cnt=0;}
            //    else ticks_wait(TICKS2WAIT);
            //} else 
            ticks_wait(TICKS2WAIT);
        }     
        return true;
    }
    
    bool pop(void ** task) {
        //register int cnt = 0;       
        while (! T::get(task)) {
            //    if (ch->thxcore>1) {
            //if (++cnt>PUSH_POP_CNT) { sched_yield(); cnt=0;}
            //else ticks_wait(TICKS2WAIT);
            //} else 
            ticks_wait(TICKS2WAIT);
        } 
        return true;
    }
    
    virtual void* svc(void * ) {
        void * task = NULL;

        do {
            pop(&task);
            if (task == (void*)FF_EOS) {
                // WARNING: the collector could not have been attached
                // to the output buffer!
                push(task); 
                break;
            }
            void * result = T::svc(task);
            if (!result) continue;
            push(result);
        } while(1);

        return NULL;
    }

    virtual void svc_end(void * result) {
        T::svc_end(result);
        double t= farmTime(STOP_TIME);
        std::cerr << "Worker " << T::get_my_id() << " time= " << t << "\n";
    }

protected:    
    void * args;
};

} // namespace ff

#endif /* _FF_NODE_HPP_ */
