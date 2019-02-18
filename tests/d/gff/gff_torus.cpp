/* ***************************************************************************
 *
 *  This file is part of gam.
 *
 *  gam is free software; you can redistribute it and/or modify it
 *  under the terms of the GNU Lesser General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  gam is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *  See the GNU Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with gam. If not, see <http://www.gnu.org/licenses/>.
 *
 ****************************************************************************
 */

/**
 * 
 * @file        gff_bb_torus.cpp
 * @brief       a simple gff torus
 * @author      Maurizio Drocco
 *
 */

#include <iostream>
#include <cassert>
#include <cmath>

#include <ff/d/gff/gff.hpp>

#define STREAMLEN             1024
#define NSTAGES               8

/*
 ***************************************************************************
 *
 * pipeline stages
 *
 ***************************************************************************
 */

class Stage1Logic {
public:
    gff::token_t svc(gam::private_ptr<int> &in, gff::OneToOne &c)
    {
        if(++cnt == STREAMLEN)
        	return gff::eos;
        c.emit(std::move(in));
        return gff::go_on;
    }

    void svc_init(gff::OneToOne &c)
    {
    	c.emit(gam::private_ptr<int>(gam::make_private<int>(42)));
    }

    void svc_end(gff::OneToOne &c)
    {
    }

private:
    size_t cnt{0};
};

typedef gff::Filter<gff::OneToOne, gff::OneToOne, //
        gam::private_ptr<int>, gam::private_ptr<int>, //
		Stage1Logic> Stage1;

class Stage2Logic {
public:
    gff::token_t svc(gam::private_ptr<int> &in, gff::OneToOne &c)
    {
    	printf("> [Stage2] in\n");
    	++cnt;
        c.emit(std::move(in));
        return gff::go_on;
    }

    void svc_init(gff::OneToOne &c)
    {
    }

    void svc_end(gff::OneToOne &c)
    {
    	assert(cnt == STREAMLEN);
    }

private:
    size_t cnt{0};
};

typedef gff::Filter<gff::OneToOne, gff::OneToOne, //
        gam::private_ptr<int>, gam::private_ptr<int>, //
		Stage2Logic> Stage2;

/*
 *******************************************************************************
 *
 * main
 *
 *******************************************************************************
 */
int main(int argc, char * argv[])
{
	std::vector<gff::OneToOne> comms(NSTAGES);

	gff::add(Stage1(comms.back(), comms[0]));

	for(size_t i = 1; i < NSTAGES; ++i)
		gff::add(Stage2(comms[i-1], comms[i]));

    gff::run();

    return 0;
}
