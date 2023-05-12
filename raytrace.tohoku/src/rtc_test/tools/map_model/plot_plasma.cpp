#include <fstream>
#include "../../../rtc/raytrace.h"
#include "misc.h"
////////////////////////////////////////////////////氷衛星簡易モデルのために一部修正

void trace_plasm(
	const rtc::basic_plasma_model& p,
	const rtc::vector& start_ptr,
	const rtc::vector&   end_ptr,
	const int               step
){
	double dens = 0.0;
	std::ofstream f( "pltr" );
	rtc::vector ds = end_ptr - start_ptr;
	ds /= step;

	rtc::vector ptr = start_ptr;
	for( int s = 0; s < step; ++s, ptr+=ds )
	{
		dens = p(ptr);
		f << s << " " << dens << " "
		  << ptr[0]/p.getMother().getRadius() << " "
		  << ptr[1]/p.getMother().getRadius() << " "
		  << ptr[2]/p.getMother().getRadius() << "\n";
	}
}

void plot_plasma_V( 
	const char* pref,
	const rtc::basic_plasma_model& p,
	const double mlt
){
	std::string 
		name_fp(pref),
		name_frx(pref),
		name_fpfc(pref);
	name_frx  += "_frx";
	name_fpfc += "_fpfc";

	std::ofstream
		fp(name_fp.c_str()),
		fp_frx( name_frx.c_str() ),
		fp_fpfc( name_fpfc.c_str() );

	const double Re = p.getMother().getRadius();
//	const double theta = rtc::mlt2rad(mlt);                              /////元の設定
	const double theta = 0;                                              /////単純化するため左のように変更
	const double
//		range = 5*Re,                                                    /////元の設定
		range = 500*Re,                                                   /////計算範囲の関係でプロット範囲の広さも変える必要があった
		step  = range/500;

	for( double r = -range*3.0 ; r < range*3.0; r += step )
	{
		const double
			x = r*std::cos(theta),
			y = r*std::sin(theta);

//		for( double z = -range; z < range; z += step )                   /////地下のプラズマ密度を求めても仕方ないので・・
		for( double z = -range; z < range*3.0; z += step )
		{
			rtc::vector ptr(3);
			ptr[0] = x; ptr[1] = y; ptr[2] = z;

			plot_ptr( fp,     ptr, p(ptr)                               );
			plot_ptr( fp_frx, ptr, freq_RxCutoff(ptr)/(2*rtc::cnst::pi) );
			plot_ptr( fp_fpfc,ptr, freq_fp(ptr) / freq_fc(ptr)          );
		}
		fp      << std::endl;
		fp_frx  << std::endl;
		fp_fpfc << std::endl;

		std::clog << ".";
	}
	std::clog << std::endl;
}

void plot_plasma_H(
	const char* pref,
	const rtc::basic_plasma_model& p,
	const double height
){
	std::string 
		name_fp(pref),
		name_frx(pref),
		name_fpfc(pref);
	name_frx  += "_frx";
	name_fpfc += "_fpfc";

	std::ofstream
		fp(name_fp.c_str()),
		fp_frx( name_frx.c_str() ),
		fp_fpfc( name_fpfc.c_str() );

	const double
		Re    = p.getMother().getRadius(),
		range = 7*Re,
		step  = 2*range/100;

	for( double x = -range; x < range; x += step )
	{
		for( double y = -range; y < range; y += step )
		{
			rtc::vector ptr(3);
			ptr[0] = x;
			ptr[1] = y;
			ptr[2] = height*Re;
			
			plot_ptr(fp,      x,y, p(ptr)                               );
			plot_ptr(fp_frx,  x,y, freq_RxCutoff(ptr)/(2*rtc::cnst::pi) );
			plot_ptr(fp_fpfc, x,y, freq_fp(ptr) / freq_fc(ptr)          );
		}
		fp      << std::endl;
		fp_frx  << std::endl;
		fp_fpfc << std::endl;

		std::clog << ".";
	}
	std::clog << "\n";
}
