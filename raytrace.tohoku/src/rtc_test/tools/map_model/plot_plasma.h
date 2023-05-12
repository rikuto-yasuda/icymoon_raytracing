#ifndef RAYTRACE_MISC_PLASMA_H
#define RAYTRACE_MISC_PLASMA_H

void trace_plasma(
	const rtc::basic_plasma_model& p,
	const rtc::vector& start_ptr,
	const rtc::vector&   end_ptr,
    const int          step
);

void plot_plasma_V( 
	const char* pref,
	const rtc::basic_plasma_model& p,
	const double mlt
);

void plot_plasma_H(
	const char* pref,
	const rtc::basic_plasma_model& p,
	const double height
);


#endif
