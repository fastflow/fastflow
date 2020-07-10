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

/**

 Preliminary test for cross-platform testing
 
*/
#include <vector>
#include <iostream>
#include <stdint.h>

#if defined(_MSC_VER)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

int main() {

	std::cout << "char " << sizeof(char) << "\n";
	std::cout << "short " << sizeof(short) << "\n";
	std::cout << "int " << sizeof(int) << "\n";
	std::cout << "long " << sizeof(long) << "\n";
	std::cout << "long long " << sizeof(long long) << "\n";
	std::cout << "float " << sizeof(float) << "\n";
	std::cout << "double " << sizeof(double) << "\n";
	std::cout << "void * " << sizeof(void *) << "\n";
	std::cout << "size_t " << sizeof(size_t) << "\n";
	std::cout << "ssize_t " << sizeof(ssize_t) << "\n";
    
    return 0;
}
