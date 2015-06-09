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

// Massimo Torquati October 2013

#include <cstdio>
#include <string>

#include <ff/gsearch.hpp>

using namespace ff;


int main(int argc, char *argv[]) {
    std::string input = "abc";
    if (argc>1) {
        if (argc<2) {
            printf("use: %s string\n", argv[0]);
            return -1;
        }
        input = std::string(argv[1]);
    }

    // create some nodes
    gnode_t<std::string> *node1  = new gnode_t<std::string>(1,  "aaa");
    gnode_t<std::string> *node2  = new gnode_t<std::string>(2,  "aacb");
    gnode_t<std::string> *node3  = new gnode_t<std::string>(8,  "bbcb");
    gnode_t<std::string> *node4  = new gnode_t<std::string>(4,  "baac");
    gnode_t<std::string> *node5  = new gnode_t<std::string>(7,  "bba");
    gnode_t<std::string> *node6  = new gnode_t<std::string>(6,  "bacb");
    gnode_t<std::string> *node7  = new gnode_t<std::string>(3,  "aaa");
    gnode_t<std::string> *node8  = new gnode_t<std::string>(5,  "abb");
    gnode_t<std::string> *node9  = new gnode_t<std::string>(9,  "abbc");
    gnode_t<std::string> *node10  = new gnode_t<std::string>(10,  "ccacbb");
    gnode_t<std::string> *node11  = new gnode_t<std::string>(11,  "acbccb");
    gnode_t<std::string> *node12  = new gnode_t<std::string>(12,  "cacbbc");
    gnode_t<std::string> *node13  = new gnode_t<std::string>(13,  "aaa");
    gnode_t<std::string> *node14  = new gnode_t<std::string>(14,  "abbc");


    // connect them 
    node1->add_node(node2);    node1->add_node(node3);
    node2->add_node(node4);    node2->add_node(node3);
    node3->add_node(node7);
    node4->add_node(node1);
    node5->add_node(node3);    node5->add_node(node13);     node5->add_node(node12);
    node6->add_node(node2);    node6->add_node(node7);      node6->add_node(node9);    node6->add_node(node5);    node6->add_node(node10);
    node7->add_node(node8);    node7->add_node(node5);      node7->add_node(node6);
    node8->add_node(node4);
    node9->add_node(node10);   node9->add_node(node14);
    node10->add_node(node4);    node10->add_node(node13);
    node11->add_node(node1);    node11->add_node(node3);    node11->add_node(node5);
    node12->add_node(node2);    node12->add_node(node3);    node12->add_node(node10);
    node14->add_node(node7);

    // this is the one to search
    gnode_t<std::string> s(0, input);
    
    // create the pattern
    ff_graphsearch<gnode_t<std::string> > gs;

    // where to store the results
    std::deque<gnode_t<std::string>*> result;

    // one-shot search
    bool found = gs.search(node1, &s, result, 2);

    if (found) { 
        printf("FOUND\n"); 
        for(auto r: result) 
            printf("Id=%ld\n", r->getId());
    } else printf("NOT FOUND\n");
    
    return 0;
}
