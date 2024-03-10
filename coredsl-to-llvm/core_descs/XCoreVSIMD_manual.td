
// Immediate index vector-extract
let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in 
    def CV_EXTRACT_H__S_V2 : RVInst_CV_EXTRACT_H<(outs GPR:$rd), (ins PulpV2:$rs1, simm6:$imm)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in 
    def CV_EXTRACTU_H__S_V2 : RVInst_CV_EXTRACTU_H<(outs GPR:$rd), (ins PulpV2:$rs1, simm6:$imm)>;

def : Pat<
	(vector_extract PulpV2:$rs1, simm6:$imm),
	(!cast<RVInst>("CV_EXTRACT_H__S_V2") PulpV2:$rs1, simm6:$imm)>;

def : Pat<
	(sra (shl (vector_extract PulpV2:$rs1, simm6:$imm), (i32 16)), (i32 16)),
	(!cast<RVInst>("CV_EXTRACT_H__S_V2") PulpV2:$rs1, simm6:$imm)>;

def : Pat<
	(and (vector_extract PulpV2:$rs1, simm6:$imm), (i32 0xffff)),
	(!cast<RVInst>("CV_EXTRACTU_H__S_V2") PulpV2:$rs1, simm6:$imm)>;


let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in 
    def CV_EXTRACT_B__S_V4 : RVInst_CV_EXTRACT_B<(outs GPR:$rd), (ins PulpV4:$rs1, simm6:$imm)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in 
    def CV_EXTRACTU_B__S_V4 : RVInst_CV_EXTRACTU_B<(outs GPR:$rd), (ins PulpV4:$rs1, simm6:$imm)>;

def : Pat<
	(vector_extract PulpV4:$rs1, simm6:$imm),
	(!cast<RVInst>("CV_EXTRACT_B__S_V4") PulpV4:$rs1, simm6:$imm)>;

def : Pat<
	(sra (shl (vector_extract PulpV4:$rs1, simm6:$imm), (i32 24)), (i32 24)),
	(!cast<RVInst>("CV_EXTRACT_B__S_V4") PulpV4:$rs1, simm6:$imm)>;

def : Pat<
	(and (vector_extract PulpV4:$rs1, simm6:$imm), (i32 0xff)),
	(!cast<RVInst>("CV_EXTRACTU_B__S_V4") PulpV4:$rs1, simm6:$imm)>;

// Register index vector-extract
let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in 
    def CV_SHUFFLE_B__S_V4_S : RVInst_CV_SHUFFLE_B<(outs GPR:$rd), (ins PulpV4:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in 
    def CV_SHUFFLE_H__S_V2_S : RVInst_CV_SHUFFLE_H<(outs GPR:$rd), (ins PulpV2:$rs1, GPR:$rs2)>;

def : Pat<
	(vector_extract PulpV4:$rs1, GPR:$rs2),
	(!cast<RVInst>("CV_SHUFFLE_B__S_V4_S") PulpV4:$rs1, GPR:$rs2)>;

def : Pat<
	(vector_extract PulpV2:$rs1, GPR:$rs2),
	(!cast<RVInst>("CV_SHUFFLE_H__S_V2_S") PulpV2:$rs1, GPR:$rs2)>;

// v4i8 BUILD_VECTOR
let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in 
    def CV_PACKLO_B__V4_V4_S_S : RVInst_CV_PACKLO_B<(outs PulpV4:$rd), (ins GPR:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1, Constraints = "$rd = $rd_wb" in 
    def CV_PACKHI_B__V4_V4_S_S : RVInst_CV_PACKHI_B<(outs PulpV4:$rd_wb), (ins PulpV4:$rd, GPR:$rs1, GPR:$rs2)>;

def : Pat<
	(v4i8 (build_vector GPR:$rs1, GPR:$rs2, GPR:$rs3, GPR:$rs4)),
	(!cast<RVInst>("CV_PACKHI_B__V4_V4_S_S") (!cast<RVInst>("CV_PACKLO_B__V4_V4_S_S") GPR:$rs2, GPR:$rs1), GPR:$rs4, GPR:$rs3)>;

// Immediate index vector-insert
let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1, Constraints = "$rd = $rd_wb" in 
    def CV_INSERT_B__V4_V4_S : RVInst_CV_INSERT_B<(outs PulpV4:$rd_wb), (ins PulpV4:$rd, GPR:$rs1, simm6:$imm)>;

def : Pat<
	(vector_insert PulpV4:$rd, GPR:$rs1, simm6:$imm),
	(!cast<RVInst>("CV_INSERT_B__V4_V4_S") PulpV4:$rd, GPR:$rs1, simm6:$imm)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1, Constraints = "$rd = $rd_wb" in 
    def CV_INSERT_H__V2_V2_S : RVInst_CV_INSERT_H<(outs PulpV2:$rd_wb), (ins PulpV2:$rd, GPR:$rs1, simm6:$imm)>;

def : Pat<
	(vector_insert PulpV2:$rd, GPR:$rs1, simm6:$imm),
	(!cast<RVInst>("CV_INSERT_H__V2_V2_S") PulpV2:$rd, GPR:$rs1, simm6:$imm)>;


def : Pat<
	(add
        (add 
            (mul 
                (sra (shl (i32 GPR:$rs2), (i32 16)), (i32 16)), 
                (sra (shl (i32 (vector_extract PulpV2:$rs1, 0)), (i32 16)), (i32 16))),
                
            GPR:$rd),

        (mul 
            (sra (shl (i32 GPR:$rs3), (i32 16)), (i32 16)), 
            (sra (shl (i32 (vector_extract PulpV2:$rs1, 1)), (i32 16)), (i32 16)))),

	(CV_SDOTSP_H__S_S_V2_V2 GPR:$rd, PulpV2:$rs1, (CV_PACK__V2_S_S GPR:$rs3, GPR:$rs2))>;


def : Pat<
	(add
        (add 
            (mul 
                (assertsext GPR:$rs2), 
                (sra (shl (i32 (vector_extract PulpV2:$rs1, 0)), (i32 16)), (i32 16))),
                
            GPR:$rd),

        (mul 
            (assertsext GPR:$rs3), 
            (sra (shl (i32 (vector_extract PulpV2:$rs1, 1)), (i32 16)), (i32 16)))),

	(CV_SDOTSP_H__S_S_V2_V2 GPR:$rd, PulpV2:$rs1, (CV_PACK__V2_S_S GPR:$rs3, GPR:$rs2))>;
