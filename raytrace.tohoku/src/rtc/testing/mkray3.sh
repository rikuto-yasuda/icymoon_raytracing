#for HIG in -500 -400 -300 -200 -100 0 100 500 600 1000 1100 1500 1600 2000 2100 2500 2600 #3000 3100 3500 3600 4000 4100 4500 4600 5000 5100 5500 5600 6000 6100 6500 6600 7000 7100 7500 7600 8000 8100
for HIG in `seq -200 2 1600`
#for HIG in `seq -100 2 100`
do

#for FREQ in 3.612176179885864258e5 3.984813988208770752e5 4.395893216133117676e5 4.849380254745483398e5 5.349649786949157715e5 5.901528000831604004e5 6.510338783264160156e5 7.181954979896545410e5 7.922856807708740234e5 8.740190267562866211e5 9.641842246055603027e5 1.063650846481323242e6 1.173378825187683105e6 1.294426321983337402e6 1.427961349487304688e6 1.575271964073181152e6 1.737779378890991211e6 1.917051434516906738e6 2.114817380905151367e6 2.332985162734985352e6 2.573659420013427734e6 2.839162111282348633e6 3.132054328918457031e6 3.455161809921264648e6 3.811601638793945312e6 4.204812526702880859e6 4.638587474822998047e6 5.117111206054687500e6 5.644999980926513672e6
for FREQ in 3.612176179885864258e5 3.984813988208770752e5 4.395893216133117676e5 4.849380254745483398e5 5.349649786949157715e5 5.901528000831604004e5 6.510338783264160156e5 7.181954979896545410e5 7.922856807708740234e5 8.740190267562866211e5 9.641842246055603027e5 1.063650846481323242e6 1.173378825187683105e6 1.294426321983337402e6
#for FREQ in 1.427961349487304688e6 1.575271964073181152e6 1.737779378890991211e6 1.917051434516906738e6 2.114817380905151367e6 2.332985162734985352e6 2.573659420013427734e6 2.839162111282348633e6 3.132054328918457031e6 3.455161809921264648e6 3.811601638793945312e6 4.204812526702880859e6 4.638587474822998047e6 5.117111206054687500e6 5.644999980926513672e6

do

RAYTRACE='./testing'

## �ѥ�᡼�������� ##############################
## ȯ��������� #��
# euclid����ξ��ϡ�����Ⱦ��ñ�̤ǡ�
# polar����ξ��ϡ�MKSñ�̤ǡ�
#COORD="polar"   # (euclid|polar)
COORD="euclid" # (euclid|polar9
#SX=70           # (source.x|MLAT)
#SY=23           # (source.y|MLT)
#SZ=2e6         # (source.z|altitude)
#SZ=1.1e8        # (source.z|altitude)

SX=-10000
SY=0
SZ=${HIG}

## ��ǥ������ ##
PLASMA="callisto_nonplume"    #(null|test_null|simple|test_simple|europa_plume|europa_nonplume|ganymede_nonplume|sato|nsumei|devine_garrett)
MAGNET="test_simple"         #(null|test_null|simple|test_simple|igrf|igrf4|vip4)
PLANET="benchmark"        #(earth(?)|jupiter|benchmark)

## ����λ������� ##
DATE="2000/1/1"  # year/month/day
TIME="0:0.0"     # hour:minutes.sec

## ��ư���������� ##
FREQ=${FREQ}      # ���ȿ�[Hz]
MODE="LO"       # ��ư�⡼��(LO|RX)
RAY_L=4e8     # �ȥ졼���������θ�ϩĹ
PITCH=0        # ������Ф���ԥå���
SEGMENT=300     # ���Ϥ����ϩ������ο�
#MAX_STEP=1300 # �ȥ졼�������ƥåפκ����
MAX_STEP=45000 # �ȥ졼�������ƥåפκ����
#STEP_LENGTH=100  # �����ƥåפǿʤ����θ�ϩĹ (1step��˿ʤ�����Ĺ��[m]�ǻ��ꤹ��)
STEP_LENGTH=10000000  # �����ƥåפǿʤ����θ�ϩĹ (1step��˿ʤ�����Ĺ��[m]�ǻ��ꤹ��)
PRECISION="10000"  # �����ƥå״֤Υ٥��ȥ�����ε���Ψ
#TIME_RANGE="1e-4:1e-6"  # �����ƥå״֤λ���ʬ��ǽ���  (1step��˿ʤ����֤κ����͡��Ǿ��ͤ���ꤹ�롣)
TIME_RANGE="4e-6:1e-13"  # �����ƥå״֤λ���ʬ��ǽ���  (1step��˿ʤ����֤κ����͡��Ǿ��ͤ���ꤹ�롣)

## plasma cavity ##
# --cavity [fp/fc]/[ilat]:[ilat range]/[mlt]:[mlt range]/[height upper]:[height bottom]
# ���٤�MKSAñ�̷�
CAVITY_LIST=(                      \
  '--cavity 0.03/70:3/0:1/3e7:5e6' \
) # cavity�ο��������ץ��������

## ���ϥե�����̾����ꤹ�롣
OUTPUT="ray-P${PLASMA}_1.5e2_4e2-M${MAGNET}-${PLANET}-${MODE}-Z${SZ}-FR${FREQ}"

##OUTPUT="ray-P${PLASMA}-M${MAGNET}-${PLANET}-${MODE}-X${SX}-FR${FREQ}-PITCH${PITCH}"
LOG="${0}.log"

## ��λ���˥᡼������� ##
MAIL_TO="" # ��λ�᡼������������ꤹ�롣
MAIL_FROM="${USER}" # ������From���������뤳�Ȥ��Ǥ��롣
MAIL_SUBJECT="[mkray] ${OUTPUT} was completed."

##################################################

send_mail()
{
	if [ ${MAIL_TO} ]; then
		echo "${MAIL_SUBJECT}" | mail "${MAIL_TO}" -s "${MAIL_SUBJECT}" -- -f ${MAIL_FROM}
	fi
}

## main ##########################################
	echo "BEGIN (${OUTPUT}) at " `date` >> ${LOG}

	$RAYTRACE \
	  --plot ray                    \
	  --verbose                     \
	  --source-coord ${COORD}       \
	  --source-x     ${SX}          \
	  --source-y     ${SY}          \
	  --source-z     ${SZ}          \
	  --plasma-model ${PLASMA}      \
	  --magnet-model ${MAGNET}      \
	  --freq         ${FREQ}        \
	  --ray-mode     ${MODE}        \
	  --ray-length   ${RAY_L}       \
	  --step-count   ${MAX_STEP}    \
	  --step-length  ${STEP_LENGTH} \
	  --time-range   ${TIME_RANGE}  \
	  --precision    ${PRECISION}   \
	  --pitch        ${PITCH}       \
	  --ray-path-segment ${SEGMENT} \
	  --planet       ${PLANET}      \
	  ${CAVITY_LIST}                \
	  2>&1                          \
	  1> ${OUTPUT}                  \
	  | tee ${LOG}

#�ȤäƤ��ʤ����ץ����
#	  --back-trace                  \
#	  --without-plot-startptr       \
#	  --parallel                    \


	echo "END (${OUTPUT}) at " `date` >> ${LOG}
	send_mail

done

done