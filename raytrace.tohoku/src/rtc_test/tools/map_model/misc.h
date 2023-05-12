#ifndef WORK_RAYTRACE_MISC_MISC_H
#define WORK_RAYTRACE_MISC_MISC_H

double freq_fc( const rtc::vector& r );
double freq_fp( const rtc::vector& r );
double freq_RxCutoff( const rtc::vector& r );
double freq_LoCutoff( const rtc::vector& r );
double freq_UHR( const rtc::vector& r );

void plot_ptr(
    std::ostream& o,
    const rtc::vector& v,
	const double   p
);

void plot_ptr(
    std::ostream& o,
	const double x,
	const double y,
	const double p
);

#endif
