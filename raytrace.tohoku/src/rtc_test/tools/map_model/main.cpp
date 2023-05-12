#include "main.h"

/////////////////////////////////////////////////////氷衛星の簡易モデルのプロットのため一部修正
int plot_plasma(rtc::basic_plasma_model &p)
{
	std::clog << "plotting plasma-xy";
	plot_plasma_H("pxy_normal", p, 0);

	std::clog << "plotting plasma-xz";
	plot_plasma_V("pxz-normal", p, 0);
	/*                                                   ////////キャビティーは存在しないので無視（コメントアウト）
		const double Re = p.getMother().getRadius();
		p.addCavity( rtc::cavity(
			0.3,
			58,15,
			0, 6,
			7.0*Re,
			0.8*Re
		));
		std::clog << "plotting plasma-xy";
		plot_plasma_H("pxy_cav", p, 0 );

		std::clog << "plotting plasma-xz";
		plot_plasma_V("pxz-cav", p, 0 );
	*/
	return 0;
}

int plot_magnet(rtc::basic_magnet_model &m)
{
	plot_magnet("m0", m, 0, 55);
	return 0;
}

int main()
{
	rtc::cosmos c(1980, 5, 6, 0, 0, 0);
	rtc::model::magnet::IGRF m(4);
	//	rtc::model::plasma::test_simple p;                 ////////プラズマモデルをテストモデルに差し替え
	//	rtc::model::plasma::europa_nonplume p;                 ////////プラズマモデルをテストモデルに差し替え
	//	rtc::model::plasma::callisto_nonplume p;                 ////////プラズマモデルをテストモデルに差し替え
	rtc::model::plasma::ganymede_nonplume p;
	rtc::planet::benchmark e(m, p); ////////惑星モデルearth->benchmarkと差し替え（プロットの単位が惑星半径のため
	const double Re = e.getRadius();
	/*p.addCavity( rtc::cavity(
		0.03,
		60,10,
		0,6,
		7.0*Re,
		0.8*Re
		));*/
	c.registerPlanet(e);

	// plot_magnet(m);
	plot_plasma(p);

	return 0;
}
