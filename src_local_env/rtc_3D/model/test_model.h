////////////////////////////////////////////////////////////////////////
// test_model.h
//	This is a library for software uses RAY-TRACING.
//	Copyright (C) 2005 Miyamoto Luisch
#ifndef RTC_RAYTRACE_TEST_MODEL_H
#define RTC_RAYTRACE_TEST_MODEL_H

namespace rtc
{
	namespace model
	{

		namespace magnet
		{

			// test_null model -----------------------------------------
			// test_null���f���ł́A����͑��݂��Ȃ��B
			// ��� 0 ��Ԃ��B
			class test_null_magnet : public basic_magnet_model
			{
			protected:
				vector getField(const vector &pos) const;

			public:
				matrix getDerivativeB(const vector &pos) const;
			};

			// test_simpte model --------------------------------------////////���K�p�iz�������Ɉ��̎�������j

			class test_simple : public basic_magnet_model
			{
			public:
				test_simple();

				vector getFootPrint(
					const vector &sp,
					double) const;

				vector getEquatorPrint(
					const vector &sp,
					double) const;

			protected:
				vector getField(const vector &pos) const;
			};

		}
		namespace plasma
		{

			// test_null model -----------------------------------------
			// test_null���f���ł́A�v���Y�}�͑��݂��Ȃ��B
			// ��� 0 ��Ԃ��B
			class test_null_plasma : public basic_plasma_model
			{
			protected:
				double getDensity(const vector &point) const;
			};

			// test_simple model ---------------------------------------
			class test_simple : public basic_plasma_model
			{
			protected:
				double getDensity(const vector &point) const;
			};

			// europa_plume model ---------------------------------------
			// Europa�̐Ð������t�v���Y�}�ƃv���[�����f����g�ݍ��킹�����f���B
			class europa_plume : public basic_plasma_model
			{
			protected:
				double getDensity(const vector &point) const;
			};

			// europa_nonplume model ---------------------------------------
			// Europa�̐Ð������t�v���Y�}�݂̂̃��f���B
			class europa_nonplume : public basic_plasma_model
			{
			protected:
				double getDensity(const vector &point) const;
			};

			// europa_clare3D model ---------------------------------------
			// Clare����̍����Europa�̃v���Y�}���f�����֐��t�B�b�e�B���O�������f���B
			class europa_clare3D : public basic_plasma_model
			{
			protected:
				double getDensity(const vector &point) const;
			};

			// ganymede_nonplume model ---------------------------------------
			// Ganymede�̐Ð������t�v���Y�}�݂̂̃��f���B
			class ganymede_nonplume : public basic_plasma_model
			{
			protected:
				double getDensity(const vector &point) const;
			};

			// callisto_nonplume model ---------------------------------------
			// Callisto�̐Ð������t�v���Y�}�݂̂̃��f���B
			class callisto_nonplume : public basic_plasma_model
			{
			protected:
				double getDensity(const vector &point) const;
			};

		}
	}
} // namespace rtc; ---------------------------------------------------

#endif // RTC_RAYTRACE_TEST_MODEL_H
