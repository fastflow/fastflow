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

#ifndef FF_DRECEIVER_H
#define FF_DRECEIVER_H

#include <ff/distributed/dreceiver/receiver.hpp>

namespace ff {

class ff_dreceiverMTCL2 : public distributed::receiver::Receiver {
    static distributed::ProtocolOptions options() {
        distributed::ProtocolOptions result;
        result.separatePayloadAbove = SINGLE_SEND_SIZE_THRESHOLD;
#ifdef DFF_ACK
        result.flowControl = distributed::FlowControl::Tokens;
#endif
        return result;
    }
public:
    // The third parameter remains accepted for source compatibility.
    ff_dreceiverMTCL2(std::string& endpoint, std::size_t inputChannels, bool = false)
        : Receiver(endpoint, inputChannels, options()) {}
};

} // namespace ff
#endif
