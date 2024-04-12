@echo off
rem プロファイリングを開始する関数を、testing.mapからコピペする。
rem prep /om /ft /sf _main  Release\testing.exe
    prep /om /ft /sf ?getFootPrint@basic_magnet_model@rtc@@QBE?AV?$vector@NV?$unbounded_array@N@ublas@numeric@boost@@@ublas@numeric@boost@@ABV3456@NN@Z Release\testing.exe
if errorlevel 1 goto done

profile Release\testing --plot ray --source-z 10e6 --freq 300e3 --pitch 90 --ray-length 1e4
if errorlevel 1 goto done

prep /m Release\testing
if errorlevel 1 goto done

plist Release\testing > profile.txt
:done
