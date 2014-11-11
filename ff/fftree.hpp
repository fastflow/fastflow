/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */


/* ***************************************************************************
 *
 *  This program is free software; you can redistribute it and/or modify it
 *  under the terms of the GNU Lesser General Public License version 3 as
 *  published by the Free Software Foundation.
 *
 *  This program is distributed in the hope that it will be useful, but WITHOUT
 *  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 *  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
 *  License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with this program; if not, write to the Free Software Foundation,
 *  Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 *
 ****************************************************************************
 */

/*
 * fftree.hpp
 *
 *  Created on: 1 Oct 2014
 *      Author: drocco
 */

#ifndef FFTREE_HPP_
#define FFTREE_HPP_

#include <ostream>
#include <vector>
#include <set>
#include <ff/node.hpp>

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
	std::vector<std::pair<fftree *, bool> > children;
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
			if (children[i].first && !children[i].second)
				delete children[i].first;
		if (isroot)
			roots.erase(this);
	}

	void add_child(fftree *t) {
		children.push_back(std::make_pair(t,t?t->ispattern():false));
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
			if (children[i].first) {
				os << " ";
				children[i].first->print(os);
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
			if (children[i].first)
				cnt += children[i].first->threadcount(tc);
		return cnt;
	}
};

static inline void print_fftrees(std::ostream &os) {
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

} // namespace

#endif /* FFTREE_HPP_ */
