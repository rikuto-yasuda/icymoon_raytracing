////////////////////////////////////////////////////////////////////////
// Lathys_europa_3d.h
//	This is a library for software uses RAY-TRACING.
#ifndef RTC_RAYTRACE_LATHYS_EUROPA_3D_H
#define RTC_RAYTRACE_LATHYS_EUROPA_3D_H

namespace rtc { namespace model { namespace plasma {

	/********************************************
	class sato_earth
	　このモデルは、佐藤さん(2000年卒業,PPARC)によって
	観測された電子密度モデルを元に作成しています。

	*********************************************/
	class lathys_europa_3d : public basic_plasma_model
	{

	protected:
		double getDensity( const vector& point ) const;
	
	};

}}}// namespace rtc

#endif//RTC_RAYTRACE_MODEL_SATO_EARTH_H
