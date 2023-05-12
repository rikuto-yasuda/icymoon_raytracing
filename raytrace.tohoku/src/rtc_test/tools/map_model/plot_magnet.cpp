#include "main.h"

void plot_magnet(
   const char* pref,
   const rtc::basic_magnet_model& m,
   const double mlt,
   const double mlat
){
	std::ofstream out(pref);

	const double theta = rtc::mlt2rad(mlt);

	// x軸を theta だけ回転
	rtc::vector y(3);
	y[0] = std::sin(theta);
	y[1] = -std::cos(theta);
	y[2] = 0;

	const double arg = rtc::mlat2rad(mlat);
	const double
		Re = m.getMother().getRadius(),
		c = std::cos(arg/2),
		s = std::sin(arg/2);

	rtc::quaternion 
		qr( c, +s*y[0],+s*y[1], +s*y[2] ),
		qc( c, -s*y[0],-s*y[1], -s*y[2] );

	const rtc::vector& mm = m.getMagneticMoment();
	rtc::quaternion k(
		  0, mm(0), mm(1), mm(2)
	);

	k = qr*k*qc *( Re/rtc::norm_2(mm) );

	// 前方向に進む。
	rtc::vector p(3);
	p(0) = k.R_component_2();
	p(1) = k.R_component_3();
	p(2) = -k.R_component_4();
	std::cout << p << std::endl;

	while( rtc::norm_2(p) > Re - 1e-2*Re )
	{
		out
			<< p(0)/Re << " "
			<< p(1)/Re << " "
			<< p(2)/Re << "\n";
		
		const rtc::vector B = -m(p);
		p += B * (1e-2*Re/rtc::norm_2(B));
	}
	out << std::endl;
}
