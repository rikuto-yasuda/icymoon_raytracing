#include "StdAfx.h"
#include "test_model.h"

using namespace rtc;
using namespace model;

// europa_surface model ////////////////////////////////////////////////////////

double reflection::europa_surface::getDensity( const vector& point ) const               ///////////////新しいプラズマモデル（z軸方向にexpで減少）
{
	const double
		r = std::sqrt((pow(point(0),2.0))+(pow(point(1),2.0))+(pow(point(2)+1.601e6,2.0)));
	const double
		rxy = std::sqrt((pow(point(0),2.0))+(pow(point(1),2.0)));
	const double
		plume = std::fabs(1.0e12*exp(-(r-1.601e6)/1.5e5)*exp(-((atan2(rxy,point(2)))/0.261799)*((atan2(rxy,point(2)))/0.261799)));
	const double
		t = std::fabs(9e9*exp(-(r-1.601e6)/2.4e5));                                  //////////////エウロパ静水圧平行モデル 地表面で9.0*10^3(/cc) スケールハイト240km
	const double
		d = t+plume;
		;

	return d;
}
