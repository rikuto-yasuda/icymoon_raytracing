object_name = 'ganymede'  # ganydeme/
highest_plasma = '4e2'  # 単位は(/cc) 2e2/4e2/16e2
plasma_scaleheight = '9e2'  # 単位は(km) 1.5e2/3e2/6e2

ft_modeling_result = np.loadtxt(
Radio_name_cdf='../result_for_yasudaetal2022/tracing_range_' +
    object_name+'/para_'+highest_plasma+'_'+plasma_scaleheight+'.csv'
Radio_Range=pd.read_csv(Radio_name_cdf, header=0)

Radio_observer_position=np.loadtxt(
    '/Users/yasudarikuto/research/icymoon_raytracing/tools/result_for_yasudaetal2022/R_P_'+object_name+'_fulldata.txt',)  # 電波源の経度を含む


galdata=np.loadtxt(
    '/Users/yasudarikuto/research/icymoon_raytracing/tools/result_sgepss_2021/GLL_GAN_2.txt')


n=len(Radio_observer_position)
Freq_str=['3.984813988208770752e5', '4.395893216133117676e5', '4.849380254745483398e5', '5.349649786949157715e5', '5.901528000831604004e5', '6.510338783264160156e5',
            '7.181954979896545410e5', '7.922856807708740234e5', '8.740190267562866211e5', '9.641842246055603027e5', '1.063650846481323242e6',
            '1.173378825187683105e6', '1.294426321983337402e6', '1.427961349487304688e6', '1.575271964073181152e6', '1.737779378890991211e6',
            '1.917051434516906738e6', '2.114817380905151367e6', '2.332985162734985352e6', '2.573659420013427734e6', '2.839162111282348633e6',
            '3.132054328918457031e6', '3.455161809921264648e6', '3.811601638793945312e6', '4.204812526702880859e6', '4.638587474822998047e6',
            '5.117111206054687500e6', '5.644999980926513672e6', ]

date=np.arange(0, 10801, 60)  # エクスプレスコードで計算している時間幅（sec)を60で割る
