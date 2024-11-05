/* ***************************************************************************
 *
 *  FastFlow is free software; you can redistribute it and/or modify it
 *  under the terms of the GNU Lesser General Public License version 3 as
 *  published by the Free Software Foundation.
 *  Starting from version 3.0.1 FastFlow is dual licensed under the GNU LGPLv3
 *  or MIT License (https://github.com/ParaGroup/WindFlow/blob/vers3.x/LICENSE.MIT)
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
 *
 * Authors: 
 *   Nicolo' Tonci
 *   Massimo Torquati
 */

#ifndef DDEFINES_HPP
#define DDEFINES_HPP

#include <tuple>
#include <utility>
#include <ff/ff.hpp>

namespace ff {

enum ChannelType {FWD, FBK};

enum ChannelLocality {LOCAL, REMOTE};

using queueLength = int;

using nodeXChannelType_t = std::tuple<ff_node*, ChannelType, ChannelLocality, queueLength>;

using EgressChannels_t = std::vector<nodeXChannelType_t>;
using IngressChannels_t = EgressChannels_t;
using IngressEgressChannels_t = std::pair<IngressChannels_t, EgressChannels_t>;

struct message2_t {
    int src, dest;
    ChannelType type;
    ChannelLocality locality;
    dataBuffer data;

    message2_t(char *rd, size_t size, bool cleanup=true) : data(rd,size,cleanup){}
    message2_t() = default;

    inline static message2_t* make_logical_EOS(int src, int dest = -1){
        message2_t* out = new message2_t;
        out->dest = dest;
        out->src = src;
        return out;
    }

    /* this is used between receiver and sender to signal that no more messages are coming from that group */
    inline static message2_t* make_pyshical_EOS(){
        message2_t* out = new message2_t;
        out->dest = -1;
        out->src = -1;
        return out;
    }

    bool isLogicalEOS(){
        return (dest == -1 && data.getLen() == 0);
    }
};

struct inputMapRecord_t {
    int index;
    ChannelType type;
    ChannelLocality locality;
    bool eos_received = false;

    inputMapRecord_t(int i, ChannelType t, ChannelLocality l) : index(i), type(t), locality(l) {}
    inputMapRecord_t() = default;
};

struct outputMapRecord_t {
    int identifier;
    ChannelType type;
    ChannelLocality locality;
    int localIndex;

    outputMapRecord_t(int i, ChannelType t, ChannelLocality l, int localIndex = -1) : identifier(i), type(t), locality(l), localIndex(localIndex) {}
};




}

#endif