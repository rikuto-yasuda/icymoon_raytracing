#include "StdAfx.h"
#include "test_model.h"

using namespace rtc;
using namespace model;

// test_null model ////////////////////////////////////////////////////////
double plasma::test_null_plasma::getDensity(const vector &point) const
{
	return 0;
}

double plasma::test_simple::getDensity(const vector &point) const ///////////////?��V?��?��?��?��?��v?��?��?��Y?��}?��?��?��f?��?��?��iz?��?��?��?��?��?��?��?��exp?��Ō�?��?��?��j
{
	const double
		z = 2 * 6.4e6; // 単位m
	const double
		n = 1.15e8; // 単�? / m3
	const double
		h = std::fabs(n * pow((point(2) / z), -3));

	return h;
	/*
	return 1.0e3;
	*/
}

double plasma::europa_plume::getDensity(const vector &point) const ///////////////?��V?��?��?��?��?��v?��?��?��Y?��}?��?��?��f?��?��?��iz?��?��?��?��?��?��?��?��exp?��Ō�?��?��?��j
{
	const double
		r = std::sqrt((pow(point(0), 2.0)) + (pow(point(1), 2.0)) + (pow(point(2) + 1.5608e6, 2.0)));
	const double
		rxy = std::sqrt((pow(point(0), 2.0)) + (pow(point(1), 2.0)));
	const double
		plume = std::fabs(1.0e12 * exp(-(r - 1.5608e6) / 1.5e5) * exp(-((atan2(rxy, point(2))) / 0.261799) * ((atan2(rxy, point(2))) / 0.261799)));
	const double
		t = std::fabs(9e9 * exp(-(r - 1.5608e6) / 2.4e5)); //////////////?��G?��E?��?��?��p?��Ð�?��?��?��?��?��s?��?��?��f?��?�� ?��n?��\?��ʂ�9.0*10^3(/cc) ?��X?��P?��[?��?��?��n?��C?��g240km
	const double
		d = t + plume;
	;

	return d;
}

double plasma::europa_nonplume::getDensity(const vector &point) const ///////////////?��V?��?��?��?��?��v?��?��?��Y?��}?��?��?��f?��?��?��iz?��?��?��?��?��?��?��?��exp?��Ō�?��?��?��j
{
	const double
		r = std::sqrt((pow(point(0), 2.0)) + (pow(point(1), 2.0)) + (pow(point(2) + 1.5608e6, 2.0)));
	const double
		rxy = std::sqrt((pow(point(0), 2.0)) + (pow(point(1), 2.0)));
	const double
		t = std::fabs(4.5e8 * exp(-(r - 1.5608e6) / 6e5)); //////////////?��G?��E?��?��?��p?��Ð�?��?��?��?��?��s?��?��?��f?��?�� ?��n?��\?��ʂ�4.0*10^2(/cc) ?��X?��P?��[?��?��?��n?��C?��g600km
	;

	return t;
}

double plasma::europa_clare3D::getDensity(const vector &point) const ///////////////Europa 3D ionospheric moddel in Clare's poster
{
	const double
		a0 = 19.5731379e6; // unit .. m^-3

	const double
		a1 = 294.081049e6; // unit .. m^-3
	const double
		a2 = 132.767864e6; // unit .. m^-3

	const double
		a3 = 1.36171699; // unit .. rad

	const double
		a4 = -2.53645889; // unit .. rad

	const double
		a5 = 1.23674495e-01; // unit .. none

	const double
		a6 = 8.19197342e-03; // unit .. none

	const double
		a7 = 8.61898769e-03; // unit .. none

	const double
		a8 = 1.17890873e-01; // unit .. none

	const double
		a9 = 0.0996583124; // unit .. length (Re)

	const double
		a10 = 2.27077240; // unit .. none

	const double
		r = (std::sqrt(point(0) * point(0) + point(1) * point(1) + point(2) * point(2)) - 1.5608e6) / 1.5608e6; // unit .. Re

	const double
		theta_row = std::atan(point(2), sqrt(point(0) * point(0) + point(1) * point(1))); // unit .. rad

	const double
		phi = std::atan2(point(1), point(0)) - a4; // unit .. rad

	const double nx = std::cos(theta_row);
	const double ny = std::sin(theta_row);
	const double n0x = std::cos(a3);
	const double n0y = std::sin(a3);

	const double theta = std::asin((nx * n0y - ny * n0x) /
								   (std::sqrt(nx * nx + ny * ny) *
									std::sqrt(n0x * n0x + n0y * n0y)));
	const double theta_theta = theta * theta;
	const double theta_phi = theta * phi;
	const double phi_phi = phi * phi;

	const double determinant = a5 * a8 - a6 * a7;

	const double f = a2 * std::exp((a8 * theta_theta - (a6 + a7) * theta_phi + a5 * phi_phi) / (-2.0 * determinant)) /
					 std::sqrt(std::abs(determinant));

	const double total = a0 + (a1 + f) * std::exp(-1.0 * std::pow(r / a9, std::abs(a10)));

	return total;
}

/*
double plasma::ganymede_nonplume::getDensity( const vector& point ) const               ///////////////?��V?��?��?��?��?��v?��?��?��Y?��}?��?��?��f?��?��?��iz?��?��?��?��?��?��?��?��exp?��Ō�?��?��?��j
{
	const double
		r = std::sqrt((pow(point(0),2.0))+(pow(point(1),2.0))+(pow(point(2)+2.6341e6,2.0)));
	const double
		rxy = std::sqrt((pow(point(0),2.0))+(pow(point(1),2.0)));
	const double
		t = std::fabs(4e8*exp(-(r-2.6341e6)/6.0e5));                                  //////////////?��K?��j?��?��?��f?��Ð�?��?��?��?��?��s?��?��?��f?��?�� ?��n?��\?��ʂ�4.0*10^2(/cc) ?��X?��P?��[?��?��?��n?��C?��g600km  ?��?��?��?��?��Over?��ς�
		;

	return t;
}
*/
double plasma::ganymede_nonplume::getDensity(const vector &point) const ///////////////?��V?��?��?��?��?��v?��?��?��Y?��}?��?��?��f?��?��?��iz?��?��?��?��?��?��?��?��exp?��Ō�?��?��?��j
{
	const double
		r = std::sqrt((pow(point(0), 2.0)) + (pow(point(1), 2.0)) + (pow(point(2) + 2.6341e6, 2.0)));
	const double
		rxy = std::sqrt((pow(point(0), 2.0)) + (pow(point(1), 2.0)));
	const double
		t = std::fabs(1.5e8 * exp(-(r - 2.6341e6) / 6e5)); // ?��K?��j?��?��?��f?��Ð�?��?��?��?��?��s?��?��?��f?��?�� ?��n?��\?��ʂł̖�?��x?��?�� ?��n?��\?��ʂ�3.5*10^2(/cc) ?��X?��P?��[?��?��?��n?��C?��g100km
	;

	return t;
}

double plasma::callisto_nonplume::getDensity(const vector &point) const ///////////////?��V?��?��?��?��?��v?��?��?��Y?��}?��?��?��f?��?��?��iz?��?��?��?��?��?��?��?��exp?��Ō�?��?��?��j
{
	const double
		r = std::sqrt((pow(point(0), 2.0)) + (pow(point(1), 2.0)) + (pow(point(2) + 2.4103e6, 2.0)));
	const double
		rxy = std::sqrt((pow(point(0), 2.0)) + (pow(point(1), 2.0)));
	const double
		t = std::fabs(1.5e8 * exp(-(r - 2.4103e6) / 4e5)); // ?��J?��?��?��X?��g?��Ð�?��?��?��?��?��s?��?��?��f?��?�� ?��n?��\?��ʂł̖�?��x?��?�� ?��n?��\?��ʂ�3.5*10^2(/cc) ?��X?��P?��[?��?��?��n?��C?��g100km
	;
	return t;
}