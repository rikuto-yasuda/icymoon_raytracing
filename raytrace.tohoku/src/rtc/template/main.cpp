////////////////////////////////////////////////////////////////////////
// RTC template main.cpp
//  This program has been written in C++.
//  Copyright (C) 2006 Miyamoto Luisch.

// include 文は これ以外を書いてはいけません。
// 追加の include は、すべてstdlib.hに書いてください。
#include "stdlib.h"

// NOTE:
// これは、RTCを用いたコードを書くときの「ひな形」となるように
// 必要最低限のコードを記述されたテンプレートです。
// RTCを使用する場合、template にあるファイルをそのままコピーして
// これを基に変更を加えていくと便利です。
//
// コンパイルで生成される実行可能ファイルの名前は
//  common.mak 中の LIBNAME = 文で定義されています。
//
// makefileは、コンパイラに応じて
//  GNU gcc              : makefile.gcc
//  Microsoft Visual C++ : makefile.vc
//  Intel Compiler       : makefile.icc
// を使用してください。コマンドラインからコンパイルするときは、
//
//  $ make -f makefile.gcc
//
// の様に使います。UNIX上で使用する場合は、例えば
//
//  $ ln -s makefile.gcc makefile
//
// の様に makefileからシンボリックリンクを張ると便利です。
//
// あたらしいソースファイルが増えた場合は、common.mak中の
// SRC = 文に追加してください。
// また、常にソースコードの先頭で stdlib.h をインクルードするよう
// 気をつけてください。これをしない場合、一部のコンパイラで
// コンパイルが失敗します。


// main 関数 -------------------------------------
int main()
{
	// RTCでは、トレース中のエラー処理には例外処理を用います。
	// すべてのエラーを確実に補足するため、mainの先頭に try節を書きます。
	try {
		// rtc::cosmos のみ、グローバル変数に置いてもかまいません。
		// が、構築時にエラーが発生した場合「Segmentation fault」が発生します。
		// try 文の中に記述する事で、それをある程度防ぐ事ができます。
		rtc::cosmos c(1980,5,6, 0,0,0);
		
		// これ以降、rtc::getCosmos() で 直前に作成した
		// cosomos オブジェクトが得られるようになります。
		assert( &c == &rtc::getCosmos() );
		
		// モデルは、自作の物も含め、グローバル変数に置かないでください。
		// グローバルで構築した場合、致命的エラーが発生することがあります。
		// なお、モデルはレイトレースを実行する間、破壊しないでください。
		// planetはモデルをコピーせず、与えられたインスタンスを直接使用します。
		rtc::model::magnet::simple m;
		rtc::model::plasma::simple p;
		
		// 惑星は、自作の物も含め、グローバル変数に置かないでください。
		// グローバルで構築した場合、致命的エラーが発生することがあります。
		rtc::planet::earth e(m,p);
		
		// 完成した惑星を宇宙に登録します。
		c.registerPlanet(e);
		
		// レイトレースの初期位置は次のコードで取得出来ます。
		rtc::vector start_ptr = e.getLocation( 70.0/*ILAT*/, 0.0/*MLT*/, 3e7/*Altitude[m]*/ );
		
		// レイトレースするためには「光」を作る必要があります。
		// RTCではcosmosが ray オブジェクトを生成し、そのポインタを返す事ができます。
		// 次のコードで、400[kHz]のR-Xモード波動を作成することができます。
		rtc::ray* r = c.createRay( rtc::wave_parameter::RX_MODE, 400e3 );
		
		// 次のコードで生成された光の向かう初期ベクトル（kベクトル）を設定することができます。
		r->initialize( start_ptr, 90.0/*pitch*/, 180.0/*round*/ );
		
		// 5000ステップだけトレースし、その結果を標準出力に出します。
		for( int n = 0; n < 5000; ++n )
		{
			// ray::take_a_step()で、必要な一ステップ進める事ができます。
			// ray::take_a_step()は、一ステップ間に経過した時刻を返します。
			r->take_a_step();
			
			// ray の現在位置は ray::getR()で取得する事ができます。
			const rtc::vector& pos = r->getR();
			std::cout << pos[0] << " " << pos[1] << " " << pos[2] << "\n";
		}
		
		// 生成した光は、次のコードで後始末することを推奨します。
		// 必須ではありません。
		c.eraseRay(r);
	}
	catch( std::exception& e )
	{
		// RTC内部から投げられた例外エラーはここでキャッチされ
		// エラーの内容を表示して終了します。
		std::cerr << e.what() << std::endl;
	}

	return 0;
}
