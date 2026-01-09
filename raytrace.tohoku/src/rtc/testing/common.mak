# 共通設定
CC = clang++
LD = clang++

CFLAGS   =
CPPFLAGS =
LDFLAGS  = -L/opt/homebrew/lib -Wl,-rpath,/opt/homebrew/lib -Wl,-search_paths_first
LDLIBS   =

STDHEADER = testing.h
LIBNAME   = testing
OBJ_PATH  = object
VPATH     = $(OBJ_PATH)
LEX       = flex
AR      = ar
ARFLAGS = rcs

SRC = \
	main.cpp    \
	raytrace.cpp\
	lex.cmd.cpp \

OBJ = $(SRC:%.cpp=$(OBJ_PATH)/%$(OBJECT_EXP))
.SUFFIXES:$(OBJECT_EXP).cpp

OUTPUT = $(LIBNAME)$(EXEC_EXP)

##################################################
all : $(OUTPUT)

# 実行ファイル(testing)をリンク
$(OUTPUT): $(OBJ) $(LIBRT)
	$(LD) $(OBJ) $(LIBRT) \
		-L/opt/homebrew/lib -Wl,-rpath,/opt/homebrew/lib -Wl,-search_paths_first \
		-lboost_thread -lboost_chrono -lboost_atomic -lpthread \
		-o $@

$(OBJ_PATH)/%$(OBJECT_EXP) :: %.cpp
	mkdir -p $(OBJ_PATH)
	$(CC) $(CC_FLAG) $< $(CC_OUTPUT_FLAG) $@

$(OBJ): $(STDHEADER) $(PCH)

.PHONY: clean local-clean
clean: local-clean
	rm -f ./$(OBJ_PATH)/*$(OBJECT_EXP)
	rm -f ./$(OBJ_PATH)/*.idb
	rm -f $(OUTPUT)
