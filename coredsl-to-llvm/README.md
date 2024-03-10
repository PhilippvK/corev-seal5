# CoreDSL to LLVM
In very active development, very experimental!

Compiles (a subset of) CoreDSL2 to LLVM TableGen ISel Patterns. 

CoreDSL2 `behavior` is first parsed into LLVM IR, which is then optimized and compiled via LLVM's usual pipeline. Just before instruction selection, we read the ISelDAG, yielding LLVM-generated instruction selection patterns for the CoreDSL2 instruction definitions.

## Usage
- Compile via `make` (initially, this will fetch and compile LLVM as well)
- Run `./cdsl2llvm <CORE DESCRIPTION>`, e.g. `./cdsl2llvm core_descs/XCoreVSIMD.core_desc`  
This will output generated patterns in `core_descs/XCoreVSIMD.td` (and instruction format classes in `XCoreVSIMDInstrFormat.td`)
- To compile LLVM with XCoreVSIMD patterns, run `make flow`.  
This will generate patterns and build LLVM for instructions defined in `core_descs/XCoreVSIMD.td`. 
The LLVM build directory is accessible via `out/` (symlink).

## Example
### Source Code
```verilog
CV_SDOTUSP_B {
    encoding: 5'b10101 :: 1'b0 :: 1'b0 :: rs2[4:0] :: rs1[4:0] :: 3'b001 :: rd[4:0] :: 7'b1010111;
    assembly: "{name(rd)}, {name(rs1)}, {name(rs2)}";
    behavior: {
        if (rd != 0) X[rd] += (unsigned<32>)((
            ((unsigned)X[rs1][ 7: 0] * (signed)X[rs2][ 7: 0]) +
            ((unsigned)X[rs1][15: 8] * (signed)X[rs2][15: 8]) +
            ((unsigned)X[rs1][23:16] * (signed)X[rs2][23:16]) +
            ((unsigned)X[rs1][31:24] * (signed)X[rs2][31:24])));
    }
}
```

### LLVM-IR (post-opt)
```llvm
define void @implCV_SDOTUSP_B(ptr noalias nocapture %0, ptr noalias nocapture readonly %1, ptr noalias nocapture readonly %2, i32 %3) local_unnamed_addr #0 {
  %5 = load <4 x i8>, ptr %1, align 1
  %6 = load <4 x i8>, ptr %2, align 1
  %7 = zext <4 x i8> %5 to <4 x i32>
  %8 = sext <4 x i8> %6 to <4 x i32>
  %9 = mul nsw <4 x i32> %8, %7
  %10 = load i32, ptr %0, align 4
  %11 = call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> %9)
  %op.rdx = add i32 %11, %10
  store i32 %op.rdx, ptr %0, align 4
  ret void
}
```

### Instruction Selection Pattern
```tablgen
def : Pat<
	(add (i32 (vecreduce_add (mul (v4i32 (sign_extend PulpV4:$rs2)), (v4i32 (zero_extend PulpV4:$rs1))))), GPR:$rd),
	(!cast<RVInst>("CV_SDOTUSP_B__S_S_V4_V4") GPR:$rd, PulpV4:$rs1, PulpV4:$rs2)>;
```
