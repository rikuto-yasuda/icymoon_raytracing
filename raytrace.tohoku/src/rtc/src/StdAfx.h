// stdafx.h : 標準のシステム インクルード ファイル、
//            または参照回数が多く、かつあまり変更されない
//            プロジェクト専用のインクルード ファイルを記述します。
//

#if !defined(AFX_STDAFX_H__89C5ABD0_D5FE_4AB7_8153_71EF8F228A65__INCLUDED_)
#define AFX_STDAFX_H__89C5ABD0_D5FE_4AB7_8153_71EF8F228A65__INCLUDED_

#if defined _WIN32 && _MSC_VER > 1000
#pragma once

#define WIN32_LEAN_AND_MEAN		// Windows ヘッダーから殆ど使用されないスタッフを除外します


// TODO: プログラムで必要なヘッダー参照を追加してください。

//非MFCでメモリリーク報告するには、以下のコードを追加して
//main()の先頭に「_CrtSetDbgFlag( _CrtSetDbgFlag(_CRTDBG_REPORT_FLAG) | _CRTDBG_LEAK_CHECK_DF);」を書く。
#ifdef _DEBUG
#include <cstdlib>
#include <new>
#include <memory>

#include <crtdbg.h>
#define _CRTDBG_MAP_ALLOC

#define new  ::new(_NORMAL_BLOCK, __FILE__, __LINE__)
#endif
#endif // _MSC_VER > 1000

// 標準ライブラリ ///////////////////////////////
#include <cstdlib>
#include <cfloat>

#include <vector>
#include <list>
#include <set>

#if defined __INTEL_COMPILER
#include "../iccmath.hpp"
#else
#include <cmath>
#include <complex>
#endif

// boost library ////////////////////////////////
#include <boost/format.hpp>
#include <boost/math/quaternion.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/operation.hpp>
#include <boost/thread.hpp>

// MSVC6では使えない
#if (defined (_MSC_VER) && _MSC_VER > 1200) || !defined _MSC_VER
#	include <boost/numeric/ublas/lu.hpp>
#endif

// namespace ////////////////////////////////////
#if defined (_MSC_VER) && _MSC_VER < 1300
namespace std {
	using ::memset;
	using ::memcpy;
}
#endif

// libraytrace //////////////////////////////////
#include "../raytrace.h"

// STLportのチェック ///////////////////////////////////////////////////
#ifndef NDEBUG
#	if !defined _STLP_USE_DYNMIC_LIB || !defined _STLP_DEBUG
#	pragma message("DEBUG版で _STLP_USE_DYNMIC_LIB と _STLP_DEBUG が定義されてないよ？")
#	endif
#endif

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ は前行の直前に追加の宣言を挿入します。

#endif // !defined(AFX_STDAFX_H__89C5ABD0_D5FE_4AB7_8153_71EF8F228A65__INCLUDED_)
