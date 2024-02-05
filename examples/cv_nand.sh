#!/bin/bash

set -e

# Script needs to be called from seal5 directory
DIR=$(pwd)


# Configure
EXT="XCoreVNand"  # used in Coredsl
EXT_="XCVNand"  # used by LLVM
INSN="CV_NAND_BITWISE"
CORE="RV32IMACFDXCoreV"
M2ISAR_LOG_LEVEL=info

# Definitions
ETISS_ARCH_DIR=$DIR/etiss_arch_riscv
TOP=$ETISS_ARCH_DIR/top.core_desc
CORE_DESC=$ETISS_ARCH_DIR/rv_xcorev/XCoreVNand.core_desc
CORE_DESC_FIXED=$CORE_DESC.2
# Workaround to remove includes from coredsl files (unsupported by coredsl2tablegen)
# Currently using manually edited file as input
# cat $CORE_DESC | sed "s/^import/\/\/ import/g" > $CORE_DESC_FIXED
YAML_DIR=$DIR/yaml
YAML=$YAML_DIR/$EXT.yml
M2ISAR_DIR=$DIR/M2-ISA-R
COREDSL2LLVM_DIR=$DIR/coredsl-to-llvm
SCRIPTS_DIR=$DIR/scripts
WORK_DIR=$DIR/work_dir
GEN_DIR=$WORK_DIR/gen
# LLVM_DIR=$WORK_DIR/llvm-project
LLVM_DIR=$DIR/llvm-project
LLVM_BUILD_DIR=$LLVM_DIR/build
# LLVM_BUILD_DIR=$WORK_DIR/build
LLVM_INSTALL_DIR=$LLVM_DIR/install
# LLVM_INSTALL_DIR=$WORK_DIR/install

# Limit number of parallel link jobs
MAX_LINK_JOBS=$(free --giga | grep Mem | awk '{print int($2 / 16)}')

# Prepare environment
export PYTHONPATH=$M2ISAR_DIR:$PYTHONPATH
# export LLVM_REPO=https://github.com/llvm/llvm-project.git
export LLVM_REF=ae42196bc493ffe877a7e3dff8be32035dea4d07

# echo "[FLOW] setup submodules"
# git -C $DIR submodule update --init
# git -C $ETISS_ARCH_DIR submodule update --init rv_xcorev

# optional (unsafe operation if GEN_DIR not set!)
# echo "[FLOW] cleanup previous outputs"
# rm -rf $GEN_DIR
# rm -rf $ETISS_ARCH_RISCV/gen_model
# rm -rf $ETISS_ARCH_RISCV/gen_output

echo "[FLOW] init directories"
mkdir -p $WORK_DIR
mkdir -p $GEN_DIR

echo "[FLOW] prepare llvm repo"
test -d $LLVM_DIR || git clone $LLVM_REPO $LLVM_DIR
git -C $LLVM_DIR checkout $LLVM_REF

echo "[FLOW] initial llvm build"
cmake -B "$LLVM_BUILD_DIR" "$LLVM_DIR/llvm/" -G Ninja -DLLVM_ENABLE_PROJECTS="clang;lld" "-DCMAKE_BUILD_TYPE=Release" -DCMAKE_INSTALL_PREFIX=$LLVM_INSTALL_DIR -DLLVM_TARGETS_TO_BUILD="X86;RISCV" -DLLVM_OPTIMIZED_TABLEGEN=ON -DLLVM_ENABLE_ASSERTIONS=OFF -DLLVM_CCACHE_BUILD=False -DLLVM_PARALLEL_LINK_JOBS=$MAX_LINK_JOBS -DLLVM_BUILD_TOOLS=ON
cmake --build $LLVM_BUILD_DIR
# optional install step
# cmake --install $LLVM_BUILD_DIR

echo "[FLOW] add llvm markers"
git -C $LLVM_DIR apply $DIR/insert_markers.patch
git -C $LLVM_DIR add --all
git -C $LLVM_DIR commit -m "[Patcher] Insert Markers"
# cmake --build $DIR/work_dir/llvm-project/build

echo "[FLOW] generate m2isar metamodel"
python -m m2isar.frontends.coredsl2.parser --log $M2ISAR_LOG_LEVEL $TOP

echo "[FLOW] generate baseline patches"
mkdir -p $GEN_DIR/0
python -m m2isar.backends.llvmgen.writer $ETISS_ARCH_DIR/gen_model/top.m2isarmodel --template RISCV --core $CORE --log $M2ISAR_LOG_LEVEL --set $EXT --insn $INSN --gen-features --gen-insn-info --gen-insn-formats --gen-asm-parser --gen-isa-info --gen-index --yaml $YAML --output-dir $GEN_DIR/0/
cd $LLVM_DIR
python $SCRIPTS_DIR/inject_extensions_2.py $GEN_DIR/0/index.yaml --msg "Add $EXT baseline patches" --append --out-file $GEN_DIR/0/diff.patch
cd -
git -C $LLVM_DIR am $GEN_DIR/0/diff.patch

echo "[FLOW] llvm build after baseline $EXT patches"
cmake --build $LLVM_BUILD_DIR
# optional
# cmake --install $LLVM_BUILD_DIR

echo "[FLOW] setup cdsl2llvm"
cd $COREDSL2LLVM_DIR
make clean all LLVM_DIR=$LLVM_DIR/llvm LLVM_BUILD_DIR=$LLVM_BUILD_DIR
cd -

echo "[FLOW] run cdsl2llvm ($EXT)"
# coredsl2llvm currently has no way to specify the output directory
mkdir -p $GEN_DIR/1/
CORE_DESC_FIXED2=$GEN_DIR/1/$EXT.core_desc
cp $CORE_DESC_FIXED $CORE_DESC_FIXED2
$COREDSL2LLVM_DIR/cdsl2llvm $CORE_DESC_FIXED2 $EXT_
mkdir -p $GEN_DIR/1/llvm/lib/Target/RISCV/
cp $GEN_DIR/1/$EXT.td $GEN_DIR/1/llvm/lib/Target/RISCV/
# cp $GEN_DIR/1/${EXT}InstrFormat.td $GEN_DIR/1/llvm/lib/Target/RISCV/
tee $GEN_DIR/1/llvm/lib/Target/RISCV/RISCVInstrInfo.td.insn_info_includes << END
include "${EXT}.td"
// include "${EXT}InstrFormat.td"
END
tee $GEN_DIR/1/index.yaml << END
artifacts:
- append: true
  key: insn_info_includes
  path: llvm/lib/Target/RISCV/RISCVInstrInfo.td
  src: llvm/lib/Target/RISCV/RISCVInstrInfo.td.insn_info_includes
extensions:
- artifacts:
  - append: false
    path: llvm/lib/Target/RISCV/$EXT.td
  # - append: false
  #   path: llvm/lib/Target/RISCV/${EXT}InstrFormat.td
END

cd $LLVM_DIR
python $SCRIPTS_DIR/inject_extensions_2.py $GEN_DIR/1/index.yaml --msg "Add $EXT patterns" --append --out-file $GEN_DIR/1/diff.patch
cd -
git -C $LLVM_DIR am $GEN_DIR/1/diff.patch

echo "[FLOW] llvm build after adding $EXT patterns"
cmake --build $LLVM_BUILD_DIR
# optional
# cmake --install $LLVM_BUILD_DIR

# Test if compilation works and patterns are used
export PATH=${LLVM_BUILD_DIR}/bin:$PATH
llvm-lit tests/cv_nand/

# TODO: simulation works (includes patching etiss)
