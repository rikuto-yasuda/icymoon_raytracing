STDHEADER = stdlib.h
LIBNAME   = template
OBJ_PATH  = object
VPATH     = $(OBJ_PATH)

# ここにソースコードのファイル名を指定する。
SRC = \
    main.cpp    \


OBJ = $(SRC:%.cpp=$(OBJ_PATH)/%$(OBJECT_EXP))
.SUFFIXES:$(OBJECT_EXP).cpp

OUTPUT = $(EXEC_PREF)$(LIBNAME)$(EXEC_EXP)

##################################################
all : $(OUTPUT)

$(OUTPUT): $(OBJ) $(RES_OBJ) $(LIBRT)
	$(LD) $(LD_FLAG) $^ $(LD_OUTPUT_FLAG)$@

$(OBJ_PATH)/%$(OBJECT_EXP) :: %.cpp
	mkdir -p $(OBJ_PATH)
	$(CC) $(CC_FLAG) $< $(CC_OUTPUT_FLAG)$@

$(OBJ): $(STDHEADER) $(PCH)

.PHONY: clean local-clean
clean: local-clean
	rm -f ./$(OBJ_PATH)/*$(OBJECT_EXP)
	rm -f ./$(OBJ_PATH)/*.idb
	rm -f $(OUTPUT)

