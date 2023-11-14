#include "StdAfx.h"
#include "test_model.h"

using namespace rtc;
using namespace model;

// test_null model ////////////////////////////////////////////////////////
double plasma::test_null_plasma::getDensity(const vector &point) const
{
	return 0;
}

double plasma::test_simple::getDensity(const vector &point) const ///////////////?øΩV?øΩ?øΩ?øΩ?øΩ?øΩv?øΩ?øΩ?øΩY?øΩ}?øΩ?øΩ?øΩf?øΩ?øΩ?øΩiz?øΩ?øΩ?øΩ?øΩ?øΩ?øΩ?øΩ?øΩexp?øΩ≈åÔøΩ?øΩ?øΩ?øΩj
{
	const double
		z = 2 * 6.4e6; // Âçò‰Ωçm
	const double
		n = 1.15e8; // Âçò‰Ω? / m3
	const double
		h = std::fabs(n * pow((point(2) / z), -3));

	return h;
	/*
	return 1.0e3;
	*/
}

double plasma::europa_plume::getDensity(const vector &point) const ///////////////?øΩV?øΩ?øΩ?øΩ?øΩ?øΩv?øΩ?øΩ?øΩY?øΩ}?øΩ?øΩ?øΩf?øΩ?øΩ?øΩiz?øΩ?øΩ?øΩ?øΩ?øΩ?øΩ?øΩ?øΩexp?øΩ≈åÔøΩ?øΩ?øΩ?øΩj
{
	const double
		r = std::sqrt((pow(point(0), 2.0)) + (pow(point(1), 2.0)) + (pow(point(2) + 1.5608e6, 2.0)));
	const double
		rxy = std::sqrt((pow(point(0), 2.0)) + (pow(point(1), 2.0)));
	const double
		plume = std::fabs(1.0e12 * exp(-(r - 1.5608e6) / 1.5e5) * exp(-((atan2(rxy, point(2))) / 0.261799) * ((atan2(rxy, point(2))) / 0.261799)));
	const double
		t = std::fabs(9e9 * exp(-(r - 1.5608e6) / 2.4e5)); //////////////?øΩG?øΩE?øΩ?øΩ?øΩp?øΩ√êÔøΩ?øΩ?øΩ?øΩ?øΩ?øΩs?øΩ?øΩ?øΩf?øΩ?øΩ ?øΩn?øΩ\?øΩ ÇÔøΩ9.0*10^3(/cc) ?øΩX?øΩP?øΩ[?øΩ?øΩ?øΩn?øΩC?øΩg240km
	const double
		d = t + plume;
	;

	return d;
}

double plasma::europa_nonplume::getDensity(const vector &point) const ///////////////?øΩV?øΩ?øΩ?øΩ?øΩ?øΩv?øΩ?øΩ?øΩY?øΩ}?øΩ?øΩ?øΩf?øΩ?øΩ?øΩiz?øΩ?øΩ?øΩ?øΩ?øΩ?øΩ?øΩ?øΩexp?øΩ≈åÔøΩ?øΩ?øΩ?øΩj
{
	const double
		r = std::sqrt((pow(point(0), 2.0)) + (pow(point(1), 2.0)) + (pow(point(2) + 1.5608e6, 2.0)));
	const double
		rxy = std::sqrt((pow(point(0), 2.0)) + (pow(point(1), 2.0)));
	const double
		t = std::fabs(4.5e8 * exp(-(r - 1.5608e6) / 6e5)); //////////////?øΩG?øΩE?øΩ?øΩ?øΩp?øΩ√êÔøΩ?øΩ?øΩ?øΩ?øΩ?øΩs?øΩ?øΩ?øΩf?øΩ?øΩ ?øΩn?øΩ\?øΩ ÇÔøΩ4.0*10^2(/cc) ?øΩX?øΩP?øΩ[?øΩ?øΩ?øΩn?øΩC?øΩg600km
	;

	return t;
}
/*
double plasma::ganymede_nonplume::getDensity( const vector& point ) const               ///////////////?øΩV?øΩ?øΩ?øΩ?øΩ?øΩv?øΩ?øΩ?øΩY?øΩ}?øΩ?øΩ?øΩf?øΩ?øΩ?øΩiz?øΩ?øΩ?øΩ?øΩ?øΩ?øΩ?øΩ?øΩexp?øΩ≈åÔøΩ?øΩ?øΩ?øΩj
{
	const double
		r = std::sqrt((pow(point(0),2.0))+(pow(point(1),2.0))+(pow(point(2)+2.6341e6,2.0)));
	const double
		rxy = std::sqrt((pow(point(0),2.0))+(pow(point(1),2.0)));
	const double
		t = std::fabs(4e8*exp(-(r-2.6341e6)/6.0e5));                                  //////////////?øΩK?øΩj?øΩ?øΩ?øΩf?øΩ√êÔøΩ?øΩ?øΩ?øΩ?øΩ?øΩs?øΩ?øΩ?øΩf?øΩ?øΩ ?øΩn?øΩ\?øΩ ÇÔøΩ4.0*10^2(/cc) ?øΩX?øΩP?øΩ[?øΩ?øΩ?øΩn?øΩC?øΩg600km  ?øΩ?øΩ?øΩ?øΩ?øΩOver?øΩœÇÔøΩ
		;

	return t;
}
*/
double plasma::ganymede_nonplume::getDensity(const vector &point) const ///////////////?øΩV?øΩ?øΩ?øΩ?øΩ?øΩv?øΩ?øΩ?øΩY?øΩ}?øΩ?øΩ?øΩf?øΩ?øΩ?øΩiz?øΩ?øΩ?øΩ?øΩ?øΩ?øΩ?øΩ?øΩexp?øΩ≈åÔøΩ?øΩ?øΩ?øΩj
{
	const double
		r = std::sqrt((pow(point(0), 2.0)) + (pow(point(1), 2.0)) + (pow(point(2) + 2.6341e6, 2.0)));
	const double
		rxy = std::sqrt((pow(point(0), 2.0)) + (pow(point(1), 2.0)));
	const double
		t = std::fabs(1e8 * exp(-(r - 2.6341e6) / 10e5)); // ?øΩK?øΩj?øΩ?øΩ?øΩf?øΩ√êÔøΩ?øΩ?øΩ?øΩ?øΩ?øΩs?øΩ?øΩ?øΩf?øΩ?øΩ ?øΩn?øΩ\?øΩ Ç≈ÇÃñÔøΩ?øΩx?øΩ?øΩ ?øΩn?øΩ\?øΩ ÇÔøΩ3.5*10^2(/cc) ?øΩX?øΩP?øΩ[?øΩ?øΩ?øΩn?øΩC?øΩg100km
	;

	return t;
}

double plasma::callisto_nonplume::getDensity(const vector &point) const ///////////////?øΩV?øΩ?øΩ?øΩ?øΩ?øΩv?øΩ?øΩ?øΩY?øΩ}?øΩ?øΩ?øΩf?øΩ?øΩ?øΩiz?øΩ?øΩ?øΩ?øΩ?øΩ?øΩ?øΩ?øΩexp?øΩ≈åÔøΩ?øΩ?øΩ?øΩj
{
	const double
		r = std::sqrt((pow(point(0), 2.0)) + (pow(point(1), 2.0)) + (pow(point(2) + 2.4103e6, 2.0)));
	const double
		rxy = std::sqrt((pow(point(0), 2.0)) + (pow(point(1), 2.0)));
	const double
		t = std::fabs(1.5e8 * exp(-(r - 2.4103e6) / 4e5)); // ?øΩJ?øΩ?øΩ?øΩX?øΩg?øΩ√êÔøΩ?øΩ?øΩ?øΩ?øΩ?øΩs?øΩ?øΩ?øΩf?øΩ?øΩ ?øΩn?øΩ\?øΩ Ç≈ÇÃñÔøΩ?øΩx?øΩ?øΩ ?øΩn?øΩ\?øΩ ÇÔøΩ3.5*10^2(/cc) ?øΩX?øΩP?øΩ[?øΩ?øΩ?øΩn?øΩC?øΩg100km
	;
	return t;
}