#ifndef _CWC_FF_DEFINITIONS_HPP_
#define _CWC_FF_DEFINITIONS_HPP_

#include <Simulation.h>

//simulation task
struct simulation_task_t {
  simulation_task_t(Simulation *sim, int offset=0):
    simulation(sim),
    id_offset(offset),
    n_sampled(0),
    running_time(0.0),
    to_end(false)
  {}

  ~simulation_task_t() {
    delete simulation;
  }

  Simulation *simulation;
  int id_offset;
  int n_sampled;
  double running_time;
  bool to_end;
};

//sample/ack task
#include <definitions.h>
#ifdef USE_FF_DISTR
#include <ff/dnode.hpp>
#include <ff/d/inter.hpp>
#include <ff/d/zmqTransport.hpp>
#include <ff/d/zmqImpl.hpp>
using namespace ff;
#endif

struct out_task_t {
#ifdef USE_FF_DISTR
  typedef zmqTransportMsg_t msg_t;
#endif

  //sample
  out_task_t(simulation_task_t *st, double t, sample_t *m):
    ack(false),
    simulation_id(st->simulation->get_id()),
    simulation_task_id(st->simulation->get_id() - st->id_offset),
    sample_time(t),
    running_time(0),
    n_sampled(-1),
    monitor(m) {}
  
  //ack
  out_task_t(simulation_task_t *st, double t):
    ack(true),
    simulation_id(st->simulation->get_id()),
    simulation_task_id(st->simulation->get_id() - st->id_offset),
    sample_time(-1),
    running_time(t),
    n_sampled(st->n_sampled),
    monitor(NULL) {}
  
  bool ack;
  int simulation_id;
  int simulation_task_id;
  double sample_time;
  double running_time;
  int n_sampled;
  sample_t *monitor;
#ifdef USE_FF_DISTR
  msg_t *msg;
#endif
};

typedef struct window_task_t {
  window_task_t(double t, sample_t **s):
    sample_time(t), samples(s) {}

  double sample_time;
  sample_t **samples;
} window_task_t;

#ifdef USE_STAT_ACCEL
#include <statistics.hpp>
//statistical task
typedef struct stat_task_t {
  Stat_Window<multiplicityType> *window;
  int mid, sid;
  stat_task_t(Stat_Window<multiplicityType> *w, int mid, int sid) :
    window(w), mid(mid), sid(sid) {}
} stat_task_t;
#endif
#endif
