// TODO(vincenzopalazzo) Adding documentation here

#ifndef STR_VISITOR_HPP
#define STR_VISITOR_HPP

#include <ff/node.hpp>
#include <ff/visit/visitor.hpp>
#include <string>

namespace ff {

class ff_str_visitor : public ff_type_visitor<std::string> {
 private:
  std::string result;

 public:
  // TODO better return a pointer here and not the
  // copy
  std::string get_result() { return result; }

  // TODO: better avoid to implemenet a visit for ff_node
  void visit_ff_node(ff_node const *node) override {
    result = result + "ff_node\n";
  }
};

};  // namespace ff

#endif
