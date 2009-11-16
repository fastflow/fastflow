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
#include <utils.hpp>
#include <buffer.hpp>

namespace ff {


enum { FF_EOS=0xffffffff, FF_GO_ON=0x1};
#define GO_ON (void*)FF_GO_ON


class ff_thread {

protected:
    ff_thread() {}
    virtual ~ff_thread() {}

    static void * thread_routine(void * arg) {
        ff_thread & obj = *(ff_thread *)arg;
        
        Barrier();        

        if (obj.svc_init()<0)
            error("ff_thread, svc_init failed, thread exit!!!\n");
        else  obj.svc(NULL);
        obj.svc_end();

        pthread_exit(NULL);
    }

public:
    virtual void* svc(void * task) = 0;
    virtual int   svc_init() { return 0; };
    virtual void  svc_end()  {}

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
    

class ff_node {
private:

    template <typename lb_t, typename gt_t> friend class ff_farm;
    friend class ff_pipeline;
    friend class ff_loadbalancer;

    void set_id(int id) { myid = id;}
    bool push(void * ptr) { return out->push(ptr); }
    bool pop(void ** ptr) { return ((in->pop(ptr)>0)?true:false);} 
    
    virtual int create_input_buffer(int nentries) {
        if (in) return -1;
        in = new SWSR_Ptr_Buffer(nentries);        
        if (!in) return -1;
        myinbuffer=true;
        return (in->init()?0:-1);
    }
    
    virtual int create_output_buffer(int nentries) {
        if (out) return -1;
        out = new SWSR_Ptr_Buffer(nentries);        
        if (!out) return -1;
        myoutbuffer=true;
        return (out->init()?0:-1);
    }

    virtual int set_output_buffer(SWSR_Ptr_Buffer * const o) {
        if (myoutbuffer) return -1;
        out = o;
        return 0;
    }

    virtual int set_input_buffer(SWSR_Ptr_Buffer * const i) {
        if (myinbuffer) return -1;
        in = i;
        return 0;
    }

    virtual int   run(bool=false) { 
        thread = new thWorker(this);
        if (!thread) return -1;
        return thread->run();
    }

    virtual int  wait() { return thread->wait(); }

    bool skipfirstpop() const { return skip1pop; }


public:
    // If svc returns a NULL value then End-Of-Stream (EOS) is produced
    // on the output channel.
    virtual void* svc(void * task) = 0;
    virtual int   svc_init() { return 0; }
    virtual void  svc_end() {}
    virtual int   get_my_id() const { return myid; };
    virtual bool  put(void * ptr) { return in->push(ptr);}
    virtual bool  get(void **ptr) { return ((out->pop(ptr)>0)?true:false);}
    virtual int   cardinality() const { return 1;}
    virtual SWSR_Ptr_Buffer * const get_in_buffer() const { return in;}
    virtual SWSR_Ptr_Buffer * const get_out_buffer() const { return out;}
    
protected:
    // protected constructor
    ff_node():in(0),out(0),myid(-1),
              myoutbuffer(false),myinbuffer(false),
              skip1pop(false), thread(NULL),callback(NULL) {};

    virtual  ~ff_node() {
        if (in && myinbuffer)  delete in;
        if (out && myoutbuffer) delete out;
    };

    virtual bool ff_send_out(void * task, 
                             unsigned int retry=((unsigned int)-1),
                             unsigned int ticks=thWorker::TICKS2WAIT) {

        if (callback) return  callback(task,retry,ticks,callback_arg);

        for(unsigned int i=0;i<retry;++i) {
            if (push(task)) return true;
            ticks_wait(ticks);
        }     
        return false;
    }

private:

    void registerCallback(bool (*cb)(void *,unsigned int,unsigned int,void *), void * arg) {
        callback=cb;
        callback_arg=arg;
    }

    class thWorker: public ff_thread {
    public:
        enum {TICKS2WAIT=1000};

        thWorker(ff_node * const filter):filter(filter) {}
        
        bool push(void * task) {
            //register int cnt = 0;
            while (! filter->push(task)) {
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
            while (! filter->pop(task)) {
                //    if (ch->thxcore>1) {
                //if (++cnt>PUSH_POP_CNT) { sched_yield(); cnt=0;}
                //else ticks_wait(TICKS2WAIT);
                //} else 
                ticks_wait(TICKS2WAIT);
            } 
            return true;
        }
        
        bool put(void * ptr) { return filter->put(ptr);}
        bool get(void **ptr) { return filter->get(ptr);}
        
        void* svc(void * ) {
            void * task = NULL;
            bool inpresent  = (filter->get_in_buffer() != NULL);
            bool outpresent = (filter->get_out_buffer() != NULL);
            bool skipfirstpop = filter->skipfirstpop(); 
            bool exit=false;
            
            do {
                if (inpresent) {
                    if (!skipfirstpop) pop(&task); 
                    else skipfirstpop=false;
                    if (task == (void*)FF_EOS) { 
                        if (outpresent)  push(task); 
                        break;
                    }
                }
                
                void * result = filter->svc(task);
                if (!result) {
                    // NOTE: The EOS is gonna be produced in the output queue
                    // and the thread is exiting even if there might be some tasks
                    // in the input queue !!!
                    result = (void *)FF_EOS; 
                    exit=true;
                }
                if (outpresent && (result != GO_ON)) push(result);
            } while(!exit);
            
            return NULL;
        }
        
        int svc_init() { return filter->svc_init(); }
        
        virtual void svc_end() {
            filter->svc_end();
            //double t= 
            farmTime(STOP_TIME);
            //std::cerr << "Worker " << T::get_my_id() << " time= " << t << "\n";
        }
        
        int run() { return ff_thread::spawn(); }
        int wait() { return ff_thread::wait();}
        
        int get_my_id() const { return filter->get_my_id(); };
        
    protected:    
        ff_node * const filter;
    };

private:
    SWSR_Ptr_Buffer * in;
    SWSR_Ptr_Buffer * out;
    int               myid;
    bool              myoutbuffer;
    bool              myinbuffer;
    bool              skip1pop;
    thWorker        * thread;
    bool (*callback)(void *,unsigned int,unsigned int, void *);
    void * callback_arg;
};


} // namespace ff

#endif /* _FF_NODE_HPP_ */
