## Makefile for GCC g++ ##########################
CC = g++
LD = g++
OBJECT_EXP = .o
PCH_EXP    = .gch
EXEC_EXP   =
LIBRT      = \
    ../lib/core.a   \
    ../lib/model.a  \
    ../lib/planet.a \
    /opt/homebrew/lib/libboost_thread-mt.a \
#    /opt//homebrew/lib/libboost_thread-mt.dylib


INCLUDE_OPT=
STDLIB     = pthread boost_thread-mt

PCH = $(STDHEADER)$(PCH_EXP)

# コンパイル時のフラグを指定
CC_FLAG = -c -O2          \
    $(INCLUDE_OPT:%=-I%)  \
    -DNDEBUG              \
    -D_THREAD_SAFE        \
    -D_REENTRANT          \
    -finline-functions    \
    -funroll-loops        \

#    -march=`arch`

CC_OUTPUT_FLAG = -o
LD_FLAG     = $(STDLIB:%=-l%)
LD_OUTPUT   = -o $(OUTPUT)

include common.mak

##################################################
$(PCH):
	$(CC) $(PCH_FLAG) $(STDHEADER) -o $@


local-clean:
	rm -f ./$(PCH)

