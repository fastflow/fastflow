/*
 * fftree.hpp
 *
 *  Created on: 1 Oct 2014
 *      Author: drocco
 */

#ifndef FFTREE_HPP_
#define FFTREE_HPP_

#include <ff/node.hpp>

#include <vector>
#include <set>

namespace ff {

static std::set<fftree *> roots;

enum fftype {
	FARM, PIPE, EMITTER, WORKER, COLLECTOR
};

union ffnode_ptr {
	ff_node *ffnode;
	ff_thread *ffthread;
};

struct threadcount_t {
	size_t n_emitter;
	size_t n_collector;
	size_t n_workers;
	size_t n_docomp;
};

struct fftree {
	bool isroot, do_comp;
	fftype nodetype;
	std::vector<fftree *> children;
	ffnode_ptr ffnode;

	fftree(ff_node *ffnode_, fftype nodetype_) :
			nodetype(nodetype_) {
		do_comp = (nodetype == WORKER);
		ffnode.ffnode = ffnode_;
		isroot = ispattern();
		if (isroot)
			roots.insert(this);
	}

	fftree(ff_thread *ffnode_, fftype nodetype_) :
			nodetype(nodetype_) {
		do_comp = (nodetype == WORKER);
		ffnode.ffthread = ffnode_;
		isroot = ispattern();
		if (isroot)
			roots.insert(this);
	}

	~fftree() {
		for (size_t i = 0; i < children.size(); ++i)
			if (children[i] && !children[i]->ispattern())
				delete children[i];
		if (isroot)
			roots.erase(this);
	}

	void add_child(fftree *t) {
		children.push_back(t);
		if (t && t->isroot)
			roots.erase(t);
	}

	std::string fftype_tostr(fftype t) {
		switch (t) {
		case FARM:
			return "FARM";
		case PIPE:
			return "PIPE";
		case EMITTER:
			return "EMITTER";
		case WORKER:
			return "WORKER";
		case COLLECTOR:
			return "COLLECTOR";
		}
		return std::string("puppa"); //never reached
	}

	bool ispattern() {
		return nodetype == FARM || nodetype == PIPE;
	}

	void print(std::ostream &os) {
		os << (ispattern() ? "[" : "(") << fftype_tostr(nodetype);
		if (do_comp)
			os << " docomp";
		for (unsigned long i = 0; i < children.size(); ++i) {
			if (children[i]) {
				os << " ";
				children[i]->print(os);
			}
		}
		os << (ispattern() ? "]" : ")");
	}

	size_t threadcount(threadcount_t *tc) {
		size_t cnt = !ispattern();
		if (cnt) {
			tc->n_emitter += nodetype == EMITTER;
			tc->n_emitter += nodetype == COLLECTOR;
			tc->n_emitter += nodetype == WORKER;
			tc->n_docomp += do_comp;
		}
		for (size_t i = 0; i < children.size(); ++i)
			if (children[i])
				cnt += children[i]->threadcount(tc);
		return cnt;
	}
};

static void print_fftrees(std::ostream &os) {
	size_t t = 0, total = 0;
	for (std::set<fftree *>::const_iterator it = roots.begin();
			it != roots.end(); ++it) {
		threadcount_t tc;
		os << "> ff tree @" << *it << "\n";
		(*it)->print(os);
		total += (t = (*it)->threadcount(&tc));
		os << "\nthread count = " << t << "\n";
	}
	os << "total thread count = " << total << "\n";
}

}
;
//end namespace ff
#endif /* FFTREE_HPP_ */
