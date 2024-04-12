
RAYTRACE='./testing'


## �ѥ�᡼�������� ##############################
## ȯ��������� ##
# euclid����ξ��ϡ�����Ⱦ��ñ�̤ǡ�
# polar����ξ��ϡ�MKSñ�̤ǡ�
#COORD="polar"   # (euclid|polar)
COORD="euclid" # (euclid|polar)
#SX=60           # (source.x|MLAT)
#SX=70           # (source.x|MLAT)
#SY=23           # (source.y|MLT)
#SZ=2e6         # (source.z|altitude)
#SZ=1.1e8        # (source.z|altitude)

SX=0
SY=0
SZ=7400


## ��ǥ������ ##
PLASMA="test_simple"    #(null|test_null|simple|test_simple|europa_plume|europa_nonplume|ganymede_nonplume|sato|nsumei|devine_garrett)
MAGNET="test_simple"         #(null|test_null|simple|test_simple|igrf|igrf4|vip4)
PLANET="benchmark"        #(earth(?)|jupiter|benchmark)

## ����λ������� ##
DATE="2000/1/1"  # year/month/day
TIME="0:0.0"     # hour:minutes.sec

## ��ư���������� ##
FREQ=1e6       # ���ȿ�[Hz]
MODE="LO"       # ��ư�⡼��(LO|RX)
RAY_L=30000     # �ȥ졼���������θ�ϩĹ
PITCH=80        # ������Ф���ԥå���
SEGMENT=100     # ���Ϥ����ϩ������ο�
MAX_STEP=100   # �ȥ졼�������ƥåפκ����
STEP_LENGTH=100000  # �����ƥåפǿʤ����θ�ϩĹ (1step��˿ʤ�����Ĺ��[m]�ǻ��ꤹ��)
PRECISION="10000"  # �����ƥå״֤Υ٥��ȥ�����ε���Ψ
TIME_RANGE="1e-6:1e-13"  # �����ƥå״֤λ���ʬ��ǽ���  (1step��˿ʤ����֤κ����͡��Ǿ��ͤ���ꤹ�롣)

## plasma cavity ##
# --cavity [fp/fc]/[ilat]:[ilat range]/[mlt]:[mlt range]/[height upper]:[height bottom]
# ���٤�MKSAñ�̷�
CAVITY_LIST=(                      \
  '--cavity 0.03/70:3/0:1/3e7:5e6' \
) # cavity�ο��������ץ��������

## ���ϥե�����̾����ꤹ�롣
OUTPUT="ray-P${PLASMA}-M${MAGNET}-${PLANET}-${MODE}-X${SX}"
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