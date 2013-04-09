/* ***************************************************************************
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU Lesser General Public License version 3 as 
 *  published by the Free Software Foundation.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 *
 *
 ****************************************************************************
 */

#ifndef _FF_OFEDSETTINGS_HPP_
#define _FF_OFEDSETTINGS_HPP_

#define MAX_SEND_WR 				11000
#define MAX_RECV_WR 				11000
#define NUMBER_OF_POST_RECV 		11000
#define CQE_ENTRIES 				22000
#define NUM_COMP_VECTORS 			22000
#define MAX_INLINE_DATA				450

/*
#define MAX_SEND_WR 				20
#define MAX_RECV_WR 				20
#define NUMBER_OF_POST_RECV 		20
#define CQE_ENTRIES 				40
#define NUM_COMP_VECTORS 			40
#define MAX_INLINE_DATA				20
*/

#define BUFFER_SIZE					100000000

// #define MEMCPY_MESSAGES				1
const int TIMEOUT_IN_MS = 120000; /* ms */

// #define DEBUG 1


#endif /* _OFEDSETTINGS_HPP_INCLUDE_ */
