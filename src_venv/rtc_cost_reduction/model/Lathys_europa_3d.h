////////////////////////////////////////////////////////////////////////
// Lathys_europa_3d.h
//	This is a library for software uses RAY-TRACING.
#ifndef RTC_RAYTRACE_LATHYS_EUROPA_3D_H
#define RTC_RAYTRACE_LATHYS_EUROPA_3D_H

namespace rtc { namespace model { namespace plasma {

	class lathys_europa_3d : public basic_plasma_model
	{
	private:
		std::vector<float> buffer;
		size_t dimlen[3];
		void loadData(const std::string& filename);

	protected:
		double getDensity( const vector& point ) const;

	public:
		lathys_europa_3d();
	};

}}}// namespace rtc

#endif//RTC_RAYTRACE_MODEL_SATO_EARTH_H
