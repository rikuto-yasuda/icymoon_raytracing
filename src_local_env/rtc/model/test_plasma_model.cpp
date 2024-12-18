#include "StdAfx.h"
#include "test_model.h"

using namespace rtc;
using namespace model;

// test_null model ////////////////////////////////////////////////////////
double plasma::test_null_plasma::getDensity(const vector &point) const
{
	return 0;
}

double plasma::test_simple::getDensity(const vector &point) const ///////////////?申V?申?申?申?申?申v?申?申?申Y?申}?申?申?申f?申?申?申iz?申?申?申?申?申?申?申?申exp?申��鐃�?申?申?申j
{
	const double
		z = 2 * 6.4e6; // ���篏�m
	const double
		n = 1.15e8; // ���篏? / m3
	const double
		h = std::fabs(n * pow((point(2) / z), -3));

	return h;
	/*
	return 1.0e3;
	*/
}

double plasma::europa_plume::getDensity(const vector &point) const ///////////////?申V?申?申?申?申?申v?申?申?申Y?申}?申?申?申f?申?申?申iz?申?申?申?申?申?申?申?申exp?申��鐃�?申?申?申j
{
	const double
		r = std::sqrt((pow(point(0), 2.0)) + (pow(point(1), 2.0)) + (pow(point(2) + 1.5608e6, 2.0)));
	const double
		rxy = std::sqrt((pow(point(0), 2.0)) + (pow(point(1), 2.0)));
	const double
		plume = std::fabs(1.0e12 * exp(-(r - 1.5608e6) / 1.5e5) * exp(-((atan2(rxy, point(2))) / 0.261799) * ((atan2(rxy, point(2))) / 0.261799)));
	const double
		t = std::fabs(9e9 * exp(-(r - 1.5608e6) / 2.4e5)); //////////////?申G?申E?申?申?申p?申��鐃�?申?申?申?申?申s?申?申?申f?申?申 ?申n?申\?申��鐃�9.0*10^3(/cc) ?申X?申P?申[?申?申?申n?申C?申g240km
	const double
		d = t + plume;
	;

	return d;
}

double plasma::europa_nonplume::getDensity(const vector &point) const ///////////////?申V?申?申?申?申?申v?申?申?申Y?申}?申?申?申f?申?申?申iz?申?申?申?申?申?申?申?申exp?申��鐃�?申?申?申j
{
	const double
		r = std::sqrt((pow(point(0), 2.0)) + (pow(point(1), 2.0)) + (pow(point(2) + 1.5608e6, 2.0)));
	const double
		rxy = std::sqrt((pow(point(0), 2.0)) + (pow(point(1), 2.0)));
	const double
		t = std::fabs(4.5e8 * exp(-(r - 1.5608e6) / 6e5)); //////////////?申G?申E?申?申?申p?申��鐃�?申?申?申?申?申s?申?申?申f?申?申 ?申n?申\?申��鐃�4.0*10^2(/cc) ?申X?申P?申[?申?申?申n?申C?申g600km
	;

	return t;
}
/*
double plasma::ganymede_nonplume::getDensity( const vector& point ) const               ///////////////?申V?申?申?申?申?申v?申?申?申Y?申}?申?申?申f?申?申?申iz?申?申?申?申?申?申?申?申exp?申��鐃�?申?申?申j
{
	const double
		r = std::sqrt((pow(point(0),2.0))+(pow(point(1),2.0))+(pow(point(2)+2.6341e6,2.0)));
	const double
		rxy = std::sqrt((pow(point(0),2.0))+(pow(point(1),2.0)));
	const double
		t = std::fabs(4e8*exp(-(r-2.6341e6)/6.0e5));                                  //////////////?申K?申j?申?申?申f?申��鐃�?申?申?申?申?申s?申?申?申f?申?申 ?申n?申\?申��鐃�4.0*10^2(/cc) ?申X?申P?申[?申?申?申n?申C?申g600km  ?申?申?申?申?申Over?申��鐃�
		;

	return t;
}
*/
double plasma::ganymede_nonplume::getDensity(const vector &point) const ///////////////?申V?申?申?申?申?申v?申?申?申Y?申}?申?申?申f?申?申?申iz?申?申?申?申?申?申?申?申exp?申��鐃�?申?申?申j
{
	const double
		r = std::sqrt((pow(point(0), 2.0)) + (pow(point(1), 2.0)) + (pow(point(2) + 2.6341e6, 2.0)));
	const double
		rxy = std::sqrt((pow(point(0), 2.0)) + (pow(point(1), 2.0)));
	const double
		t = std::fabs(1.5e8 * exp(-(r - 2.6341e6) / 6e5)); // ?申K?申j?申?申?申f?申��鐃�?申?申?申?申?申s?申?申?申f?申?申 ?申n?申\?申������鐃�?申x?申?申 ?申n?申\?申��鐃�3.5*10^2(/cc) ?申X?申P?申[?申?申?申n?申C?申g100km
	;

	return t;
}

double plasma::callisto_nonplume::getDensity(const vector &point) const ///////////////?申V?申?申?申?申?申v?申?申?申Y?申}?申?申?申f?申?申?申iz?申?申?申?申?申?申?申?申exp?申��鐃�?申?申?申j
{
	const double
		r = std::sqrt((pow(point(0), 2.0)) + (pow(point(1), 2.0)) + (pow(point(2) + 2.4103e6, 2.0)));
	const double
		rxy = std::sqrt((pow(point(0), 2.0)) + (pow(point(1), 2.0)));
	const double
		t = std::fabs(1.5e8 * exp(-(r - 2.4103e6) / 4e5)); // ?申J?申?申?申X?申g?申��鐃�?申?申?申?申?申s?申?申?申f?申?申 ?申n?申\?申������鐃�?申x?申?申 ?申n?申\?申��鐃�3.5*10^2(/cc) ?申X?申P?申[?申?申?申n?申C?申g100km
	;
	return t;
}