#include "../../raytrace.h"
#include <cmath>
#include <fstream>

double freq_fc( const rtc::vector& r )
{ return std::sqrt(
	rtc::getCosmos().getSquaredCycloFreq(r)
  );
}

double freq_fp( const rtc::vector& r )
{ return std::sqrt(
	rtc::getCosmos().getSquaredPlasmaFreq(r)
  );
}

double freq_RxCutoff( const rtc::vector& r )
{
	const double fp = rtc::getCosmos().getSquaredPlasmaFreq(r);
	const double fc = rtc::getCosmos().getSquaredCycloFreq(r);
	
	return 0.5*std::sqrt(fc) + std::sqrt(fp + fc/4);
}

double freq_LoCutoff( const rtc::vector& r )
{
	const double fp = rtc::getCosmos().getSquaredPlasmaFreq(r);
	const double fc = rtc::getCosmos().getSquaredCycloFreq(r);
	
	const double flo = -0.5*std::sqrt(fc) + std::sqrt(fp + fc/4);

	return std::max( std::max( flo, std::sqrt(fp) ), std::sqrt(fc) );
}

double freq_UHR( const rtc::vector& r )
{
	return std::sqrt(
		  rtc::getCosmos().getSquaredPlasmaFreq(r)
		+ rtc::getCosmos().getSquaredCycloFreq(r)
	);
}

void plot_ptr(
    std::ostream& o,
    const rtc::vector& v,
	const double   p
){
  const double Re = rtc::getCosmos().getPlanet().getRadius();
  for( rtc::vector::const_iterator it = v.begin(); it != v.end(); ++it )
    o << *it/Re << " ";
  o << p << "\n";
}
    

void plot_ptr( 
    std::ostream& o,
	const double x,
	const double y,
	const double p
 ){
	const double Re = rtc::getCosmos().getPlanet().getRadius();
	o 
		<< x/Re << " "
		<< y/Re << " "
		<< p << "\n";
}
