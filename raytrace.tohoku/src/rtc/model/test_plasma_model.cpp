#include "StdAfx.h"
#include "test_model.h"

using namespace rtc;
using namespace model;

// test_null model ////////////////////////////////////////////////////////
double plasma::test_null_plasma::getDensity(const vector &point) const
{
	return 0;
}

double plasma::test_simple::getDensity(const vector &point) const ///////////////�V�����v���Y�}���f���iz��������exp�Ō����j
{
	/*	const double
		h = std::fabs(0.5e5*point(2));

	return h;
*/
	return 1.0e3;
}

double plasma::europa_plume::getDensity(const vector &point) const ///////////////�V�����v���Y�}���f���iz��������exp�Ō����j
{
	const double
		r = std::sqrt((pow(point(0), 2.0)) + (pow(point(1), 2.0)) + (pow(point(2) + 1.601e6, 2.0)));
	const double
		rxy = std::sqrt((pow(point(0), 2.0)) + (pow(point(1), 2.0)));
	const double
		plume = std::fabs(1.0e12 * exp(-(r - 1.601e6) / 1.5e5) * exp(-((atan2(rxy, point(2))) / 0.261799) * ((atan2(rxy, point(2))) / 0.261799)));
	const double
		t = std::fabs(9e9 * exp(-(r - 1.601e6) / 2.4e5)); //////////////�G�E���p�Ð������s���f�� �n�\�ʂ�9.0*10^3(/cc) �X�P�[���n�C�g240km
	const double
		d = t + plume;
	;

	return d;
}

double plasma::europa_nonplume::getDensity(const vector &point) const ///////////////�V�����v���Y�}���f���iz��������exp�Ō����j
{
	const double
		r = std::sqrt((pow(point(0), 2.0)) + (pow(point(1), 2.0)) + (pow(point(2) + 1.601e6, 2.0)));
	const double
		rxy = std::sqrt((pow(point(0), 2.0)) + (pow(point(1), 2.0)));
	const double
		t = std::fabs(4e8 * exp(-(r - 1.601e6) / 4e5)); //////////////�G�E���p�Ð������s���f�� �n�\�ʂ�4.0*10^2(/cc) �X�P�[���n�C�g600km
	;

	return t;
}
/*
double plasma::ganymede_nonplume::getDensity( const vector& point ) const               ///////////////�V�����v���Y�}���f���iz��������exp�Ō����j
{
	const double
		r = std::sqrt((pow(point(0),2.0))+(pow(point(1),2.0))+(pow(point(2)+2.6341e6,2.0)));
	const double
		rxy = std::sqrt((pow(point(0),2.0))+(pow(point(1),2.0)));
	const double
		t = std::fabs(4e8*exp(-(r-2.6341e6)/6.0e5));                                  //////////////�K�j���f�Ð������s���f�� �n�\�ʂ�4.0*10^2(/cc) �X�P�[���n�C�g600km  �����Over�ς�
		;

	return t;
}
*/
double plasma::ganymede_nonplume::getDensity(const vector &point) const ///////////////�V�����v���Y�}���f���iz��������exp�Ō����j
{
	const double
		r = std::sqrt((pow(point(0), 2.0)) + (pow(point(1), 2.0)) + (pow(point(2) + 2.6341e6, 2.0)));
	const double
		rxy = std::sqrt((pow(point(0), 2.0)) + (pow(point(1), 2.0)));
	const double
		t = std::fabs(3.5e8 * exp(-(r - 2.6341e6) / 1e5)); //�K�j���f�Ð������s���f�� �n�\�ʂł̖��x�� �n�\�ʂ�3.5*10^2(/cc) �X�P�[���n�C�g100km
	;

	return t;
}