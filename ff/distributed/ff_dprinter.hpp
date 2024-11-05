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

#ifndef FF_DPRINTER_H
#define FF_DPRINTER_H

#include <iostream>
namespace ff {

class prefixbuf : public std::streambuf
{
	std::string     prefix;
	std::streambuf* sbuf;
	bool            need_prefix = true;
    
	int sync() { return this->sbuf->pubsync();}
	int overflow(int c) {
		if (c != std::char_traits<char>::eof()) {
			if (this->need_prefix
				&& !this->prefix.empty()
				&& (int)(this->prefix.size()) != this->sbuf->sputn(&this->prefix[0], this->prefix.size())) {
				return std::char_traits<char>::eof();
			}
			this->need_prefix = c == '\n';
		}
		return this->sbuf->sputc(c);
	}
    
public:
	prefixbuf(std::string const& p, std::streambuf* sbuf) : prefix(std::string("[" + p + "] ")), sbuf(sbuf) {}
	void setPrefix(std::string const& p){this->prefix = std::string("[" + p + "] ");}
};
	
class Printer : private virtual prefixbuf, public std::ostream {
public:
	Printer(std::string const& p, std::ostream& out) : prefixbuf(p, out.rdbuf()), std::ios(static_cast<std::streambuf*>(this)), std::ostream(static_cast<std::streambuf*>(this)) {}
	void setPrefix(std::string const& p){prefixbuf::setPrefix(p);}
};
    
template<class CharT, class Traits>
auto& endl(std::basic_ostream<CharT, Traits>& os){return std::endl(os);}
    
Printer cout("undefined", std::cout);

} // namespace

#endif
