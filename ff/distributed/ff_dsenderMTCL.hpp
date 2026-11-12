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
 */
/* Authors:
 *   Nicolo' Tonci
 *   Massimo Torquati
 */

#ifndef FF_DSENDER_H
#define FF_DSENDER_H

#include <ff/distributed/dsender/sender.hpp>

namespace ff {

// Compatibility boundary for existing callers and build-time defaults.
class ff_dsenderMTCL2 : public distributed::sender::Sender {
    static distributed::SenderOptions options(int records, std::size_t bytes, int tokens) {
        if (records <= 0 || tokens <= 0)
            throw std::invalid_argument("batch size and send token limit must be positive");
        distributed::SenderOptions result;
        result.batchRecords = static_cast<std::size_t>(records);
        result.batchBytes = bytes;
        result.sendTokens = static_cast<std::size_t>(tokens);
        result.sendBuffers = SEND_BUFFERS_PREALLOCATE;
        result.connectRetries = MAX_RETRIES;
        result.protocol.separatePayloadAbove = SINGLE_SEND_SIZE_THRESHOLD;
#ifdef DFF_ACK
        result.protocol.flowControl = distributed::FlowControl::Tokens;
#endif
#ifdef RANDOM_LOADBALANCING
        result.loadBalancing = distributed::LoadBalancing::Random;
#endif
        result.validate();
        return result;
    }
public:
    ff_dsenderMTCL2(const distributed::sender::Endpoints& endpoints,
                   const distributed::sender::Channels& channels,
                   int batchSize = DEFAULT_BATCH_SIZE,
                   std::size_t batchBytes = DEFAULT_BATCH_BYTE_SIZE,
                   int sendTokens = DEFAULT_MESSAGE_OTF)
        : Sender(endpoints, channels, options(batchSize, batchBytes, sendTokens)) {}
};

} // namespace ff
#endif
