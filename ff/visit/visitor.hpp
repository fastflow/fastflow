// TODO(vincenzopalazzo) Adding doc here

#ifndef FF_VISITOR_HPP
#define FF_VISITOR_HPP

#include <ff/node.hpp>

namespace ff {

class ff_node;

class ff_visitor {
 public:
  virtual void visit_ff_node(ff_node const *node) = 0;
};

template <class T>
class ff_type_visitor : public ff_visitor {
 public:
  virtual T get_result() = 0;
  virtual void visit_ff_node(ff_node const *node){};
};
};  // namespace ff
#endif
