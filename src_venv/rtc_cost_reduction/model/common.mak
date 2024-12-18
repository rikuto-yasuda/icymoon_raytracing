STDHEADER = StdAfx.h
LIBNAME   = model
OBJ_PATH  = object
VPATH     = $(OBJ_PATH)

SRC = \
    sample_magnet_model.cpp \
    test_magnet_model.cpp \
    sample_plasma_model.cpp \
	parameter_plasma_model.cpp \
    test_plasma_model.cpp \
    model_sato_earth.cpp    \
    model_nsumei_earth.cpp  \
	Lathys_europa_3d.cpp    \
	DevineGarrett.cpp       \
    IGRF.cpp                \
	VIP4.cpp                \


OBJ = $(SRC:%.cpp=$(OBJ_PATH)/%$(OBJECT_EXP))
.SUFFIXES:$(OBJECT_EXP).cpp

OUTPUT = ../lib/$(LIBNAME)$(EXEC_EXP)

##################################################
all : $(OUTPUT)

$(OUTPUT): $(OBJ) $(RES_OBJ) $(LIBRT)
	$(LD) $(LD_FLAG) $(LD_OUTPUT) $^

$(OBJ_PATH)/%$(OBJECT_EXP) :: %.cpp
	mkdir -p $(OBJ_PATH)
	$(CC) $(CC_FLAG) $< $(CC_OUTPUT_FLAG) $@

$(OBJ): $(STDHEADER) $(PCH)

.PHONY: clean local-clean
clean: local-clean
	rm -f ./$(OBJ_PATH)/*$(OBJECT_EXP)
	rm -f ./$(OBJ_PATH)/*.idb
	rm -f $(OUTPUT)

