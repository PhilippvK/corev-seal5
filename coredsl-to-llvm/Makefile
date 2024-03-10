BIN ?= cdsl2llvm
BIN_JIT ?= cdsljit
LLVM_DIR ?= llvm/llvm
LLVM_BUILD_DIR ?= llvm/llvm/build
BUILD_DIR ?= build
PATTERN_FILES ?= XCoreVSIMDInstrFormat.td XCoreVSIMD.td XCoreVALUInstrFormat.td XCoreVALU.td XCoreVSIMD_manual.td

LLVM_CONFIG = $(LLVM_BUILD_DIR)/bin/llvm-config
LLVM_CONFIG_JIT = $(LLVM_CONFIG)#llvm-config

INCLUDE_DIRS = \
	$(LLVM_DIR)/lib/Target/RISCV/ \
	$(LLVM_BUILD_DIR)/lib/Target/RISCV/

SRC_FILES_COMMON = $(wildcard src/lib/*.cpp)
SRC_FILES_CDSL2LLVM =  $(wildcard src/CoreDSL2LLVM/*.cpp)
SRC_FILES_CDSLJIT =  $(wildcard src/CoreDSLJIT/*.cpp)

OBJ = $(SRC_FILES_COMMON:%.cpp=$(BUILD_DIR)/%.o) \
	  $(SRC_FILES_CDSL2LLVM:%.cpp=$(BUILD_DIR)/%.o) \
	  $(SRC_FILES_CDSLJIT:%.cpp=$(BUILD_DIR)/%.o)
DEP = $(OBJ:%.o=%.d)

CXXFLAGS ?= -g3 -march=native
CXXFLAGS += -std=c++17 -fno-rtti

LDFLAGS_BIN = `$(LLVM_CONFIG) --ldflags` `$(LLVM_CONFIG) --libs` -lzstd -lz -lcurses -lpthread -ldl
LDFLAGS_BIN_JIT = `$(LLVM_CONFIG_JIT) --ldflags` `$(LLVM_CONFIG_JIT) --libs` -lzstd -lz -lcurses -lpthread

# Generic Targets
all: $(LLVM_CONFIG) $(BIN)
# $(BIN_JIT)

release: CXXFLAGS += -O3
release: $(LLVM_CONFIG) $(BIN) $(BIN_JIT)

debug: CXXFLAGS += -fsanitize=address
debug: $(LLVM_CONFIG) $(BIN) $(BIN_JIT)

$(BIN): $(SRC_FILES_COMMON:%.cpp=$(BUILD_DIR)/%.o) $(SRC_FILES_CDSL2LLVM:%.cpp=$(BUILD_DIR)/%.o)
	mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(LDFLAGS_BIN)

$(BIN_JIT): $(SRC_FILES_COMMON:%.cpp=$(BUILD_DIR)/%.o) $(SRC_FILES_CDSLJIT:%.cpp=$(BUILD_DIR)/%.o)
	mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(LDFLAGS_BIN_JIT)

-include $(DEP)
$(BUILD_DIR)/%.o: %.cpp
	mkdir -p $(@D)
	$(CXX) -MMD $(addprefix -I, $(INCLUDE_DIRS)) -o $@ `$(LLVM_CONFIG) --cppflags` $(CXXFLAGS) -c $<

# LLVM build to use with CDSL2LLVM
$(LLVM_CONFIG):
	# git submodule update --init --recursive
# do not include custom instructions in pattern gen LLVM build (create empty files instead)
	cd $(LLVM_DIR)/lib/Target/RISCV && $(RM) -r $(PATTERN_FILES) && touch $(PATTERN_FILES)
	mkdir -p $(LLVM_DIR)/build
	cd $(LLVM_DIR)/build && \
		cmake -G Ninja \
			-DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
			-DLLVM_TARGETS_TO_BUILD=RISCV -DLLVM_ENABLE_PROJECTS=clang -DLLVM_PARALLEL_LINK_JOBS=8 -DOPTIMIZED_TABLEGEN=ON ..
	cd $(LLVM_DIR)/build && ninja

# LLVM build to use with generated patterns
.PHONY: flow
flow: $(BIN)
# (build and) run coredsl2llvm
	./$(BIN) core_descs/XCoreVSIMD.core_desc Xcvsimd
	./$(BIN) core_descs/XCoreVALU.core_desc Xcvalu
# copy generated files to llvm
	cp -f $(addprefix core_descs/, $(PATTERN_FILES)) $(LLVM_DIR)/lib/Target/RISCV
	mkdir -p $(LLVM_DIR)/build2
# could use different CMake config for final LLVM build here
	cd $(LLVM_DIR)/build2 && \
		cmake -G Ninja \
			-DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
			-DLLVM_TARGETS_TO_BUILD=RISCV -DLLVM_ENABLE_PROJECTS=clang -DLLVM_PARALLEL_LINK_JOBS=8 -DOPTIMIZED_TABLEGEN=ON ..
	cd $(LLVM_DIR)/build2 && ninja
# create symlink for convenience
	ln -fs $(LLVM_DIR)/build2 out


.PHONY: clean
clean:
	$(RM) -r $(BIN) $(BIN_JIT) $(BUILD_DIR)
