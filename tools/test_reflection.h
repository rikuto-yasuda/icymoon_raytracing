////////////////////////////////////////////////////////////////////////
// test_model.h
//	This is a library for software uses RAY-TRACING.
//	Copyright (C) 2005 Miyamoto Luisch
#ifndef RTC_RAYTRACE_TEST_MODEL_H
#define RTC_RAYTRACE_TEST_MODEL_H

namespace rtc { namespace model {

namespace reflection {

	// europa_plume model ---------------------------------------
	// Europa�̐Ð������t�v���Y�}�ƃv���[�����f����g�ݍ��킹�����f���B
	class europa_surface : public basic_reflection
	{
	protected:
		double getDensity( const vector& point )   const;
	};

}}}// namespace rtc; ---------------------------------------------------

#endif//RTC_RAYTRACE_TEST_MODEL_H
