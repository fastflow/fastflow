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

class ff_node;

enum ChannelType : char {FWD, FBK};

enum ChannelLocality : char {LOCAL, REMOTE};

using addr_t = short;

using sizeDFF_t = size_t;

using queueLength = int;

using nodeXChannelType_t = std::tuple<ff_node*, ChannelType, ChannelLocality, queueLength>;

using EgressChannels_t = std::vector<nodeXChannelType_t>;
using IngressChannels_t = EgressChannels_t;
using IngressEgressChannels_t = std::pair<IngressChannels_t, EgressChannels_t>;

using blobReleaseF_t = void(*)(void*, char*, size_t);

// Manual serialization descriptor.
//
// copied == true:
//   data points to a runtime-owned buffer and the original task can be freed
//   immediately after enqueueing the message.
//
// copied == false:
//   data is borrowed until the transport is done with the message. The optional
//   owner/release pair describes how to release the object that keeps data alive.
struct serializedBuffer_t {
    char* data = nullptr;
    sizeDFF_t size = 0;
    bool copied = true;
    void* owner = nullptr;
    blobReleaseF_t release = nullptr;

    serializedBuffer_t() = default;
    serializedBuffer_t(char* data, sizeDFF_t size, bool copied=true,
                       void* owner=nullptr, blobReleaseF_t release=nullptr):
        data(data), size(size), copied(copied), owner(owner), release(release) {}
};

struct message2_t {
    int src, dest;
    ChannelType type;
    ChannelLocality locality;

    sizeDFF_t size = 0;
    char* data = nullptr;
    bool cleanup = false;
    // Sender-side release hook for non-copied serialized buffers. It is stored
    // per message because different messages may carry different owners.
    void* blobOwner = nullptr;
    blobReleaseF_t freeCallback = nullptr;

    message2_t(char *rd, sizeDFF_t size, bool cleanup=true) :  size(size), data(rd), cleanup(cleanup) {}
    message2_t() = default;


    ~message2_t(){ cleanContent(); }

    inline void setBuff(std::pair<char*, sizeDFF_t>&& buffPair){
        data = buffPair.first; 
        size = buffPair.second;
        blobOwner = nullptr;
        freeCallback = nullptr;
    }

    inline void cleanContent(){
        if (data && cleanup){
            // Use the owner-aware release hook when manual serialization
            // supplied one; otherwise keep the default copied-buffer policy.
            if (freeCallback) freeCallback(blobOwner, data, size);
            else free(data);
        }
        data = nullptr;
        cleanup = false;
        size = 0;
        blobOwner = nullptr;
        freeCallback = nullptr;
    }

    inline bool isFlush(){
        return dest == -2;
    }

    bool isLogicalEOS(){
        return (dest == -1 && size == 0);
    }
};

static const size_t headerSize = sizeof(addr_t) + sizeof(addr_t) + sizeof(ChannelType) + sizeof(sizeDFF_t); 

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