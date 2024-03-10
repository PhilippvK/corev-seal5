let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_ADD_H__V2_V2_V2 : RVInst_CV_ADD_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, PulpV2:$rs2)>;

def : Pat<
	(add PulpV2:$rs2, PulpV2:$rs1),
	(CV_ADD_H__V2_V2_V2 PulpV2:$rs1, PulpV2:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_ADD_SC_H__V2_V2_S : RVInst_CV_ADD_SC_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, GPR:$rs2)>;

def : Pat<
	(add PulpV2:$rs1, (v2i16 (splat_vector GPR:$rs2))),
	(CV_ADD_SC_H__V2_V2_S PulpV2:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_ADD_SCI_H__V2_V2 : RVInst_CV_ADD_SCI_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, simm6:$imm)>;

def : Pat<
	(add PulpV2:$rs1, (v2i16 (splat_vector (i32 simm6:$imm)))),
	(CV_ADD_SCI_H__V2_V2 PulpV2:$rs1, simm6:$imm)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_ADD_B__V4_V4_V4 : RVInst_CV_ADD_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, PulpV4:$rs2)>;

def : Pat<
	(add PulpV4:$rs2, PulpV4:$rs1),
	(CV_ADD_B__V4_V4_V4 PulpV4:$rs1, PulpV4:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_ADD_SC_B__V4_V4_S : RVInst_CV_ADD_SC_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, GPR:$rs2)>;

def : Pat<
	(add PulpV4:$rs1, (v4i8 (splat_vector GPR:$rs2))),
	(CV_ADD_SC_B__V4_V4_S PulpV4:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_ADD_SCI_B__V4_V4 : RVInst_CV_ADD_SCI_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, simm6:$imm)>;

def : Pat<
	(add PulpV4:$rs1, (v4i8 (splat_vector (i32 simm6:$imm)))),
	(CV_ADD_SCI_B__V4_V4 PulpV4:$rs1, simm6:$imm)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_SUB_H__V2_V2_V2 : RVInst_CV_SUB_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, PulpV2:$rs2)>;

def : Pat<
	(sub PulpV2:$rs1, PulpV2:$rs2),
	(CV_SUB_H__V2_V2_V2 PulpV2:$rs1, PulpV2:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_SUB_SC_H__V2_V2_S : RVInst_CV_SUB_SC_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, GPR:$rs2)>;

def : Pat<
	(sub PulpV2:$rs1, (v2i16 (splat_vector GPR:$rs2))),
	(CV_SUB_SC_H__V2_V2_S PulpV2:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_SUB_SCI_H__V2_V2 : RVInst_CV_SUB_SCI_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, simm6:$imm)>;

def : Pat<
	(sub PulpV2:$rs1, (v2i16 (splat_vector (i32 simm6:$imm)))),
	(CV_SUB_SCI_H__V2_V2 PulpV2:$rs1, simm6:$imm)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_SUB_B__V4_V4_V4 : RVInst_CV_SUB_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, PulpV4:$rs2)>;

def : Pat<
	(sub PulpV4:$rs1, PulpV4:$rs2),
	(CV_SUB_B__V4_V4_V4 PulpV4:$rs1, PulpV4:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_SUB_SC_B__V4_V4_S : RVInst_CV_SUB_SC_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, GPR:$rs2)>;

def : Pat<
	(sub PulpV4:$rs1, (v4i8 (splat_vector GPR:$rs2))),
	(CV_SUB_SC_B__V4_V4_S PulpV4:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_SUB_SCI_B__V4_V4 : RVInst_CV_SUB_SCI_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, simm6:$imm)>;

def : Pat<
	(sub PulpV4:$rs1, (v4i8 (splat_vector (i32 simm6:$imm)))),
	(CV_SUB_SCI_B__V4_V4 PulpV4:$rs1, simm6:$imm)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_AVG_H__V2_V2_V2 : RVInst_CV_AVG_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, PulpV2:$rs2)>;

def : Pat<
	(build_vector (i32 (srl (add (i32 (sra (i32 (shl (vector_extract PulpV2:$rs2, (i32 0)), (i32 16))), (i32 16))), (i32 (sra (i32 (shl (vector_extract PulpV2:$rs1, (i32 0)), (i32 16))), (i32 16)))), (i32 1))), (i32 (srl (add (i32 (sra (i32 (shl (vector_extract PulpV2:$rs2, (i32 1)), (i32 16))), (i32 16))), (i32 (sra (i32 (shl (vector_extract PulpV2:$rs1, (i32 1)), (i32 16))), (i32 16)))), (i32 1)))),
	(CV_AVG_H__V2_V2_V2 PulpV2:$rs1, PulpV2:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_AVG_SC_H__V2_V2_S : RVInst_CV_AVG_SC_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, GPR:$rs2)>;

def : Pat<
	(build_vector (i32 (srl (add (sra (shl GPR:$rs2, (i32 16)), (i32 16)), (i32 (sra (i32 (shl (vector_extract PulpV2:$rs1, (i32 0)), (i32 16))), (i32 16)))), (i32 1))), (i32 (srl (add (sra (shl GPR:$rs2, (i32 16)), (i32 16)), (i32 (sra (i32 (shl (vector_extract PulpV2:$rs1, (i32 1)), (i32 16))), (i32 16)))), (i32 1)))),
	(CV_AVG_SC_H__V2_V2_S PulpV2:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_AVG_SCI_H__V2_V2 : RVInst_CV_AVG_SCI_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, simm6:$imm)>;

def : Pat<
	(build_vector (i32 (srl (add (i32 (sra (i32 (shl (i32 simm6:$imm), (i32 16))), (i32 16))), (i32 (sra (i32 (shl (vector_extract PulpV2:$rs1, (i32 0)), (i32 16))), (i32 16)))), (i32 1))), (i32 (srl (add (i32 (sra (i32 (shl (i32 simm6:$imm), (i32 16))), (i32 16))), (i32 (sra (i32 (shl (vector_extract PulpV2:$rs1, (i32 1)), (i32 16))), (i32 16)))), (i32 1)))),
	(CV_AVG_SCI_H__V2_V2 PulpV2:$rs1, simm6:$imm)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_AVG_B__V4_V4_V4 : RVInst_CV_AVG_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, PulpV4:$rs2)>;

def : Pat<
	(build_vector (vector_extract (v2i16 (srl (add (v2i16 (sra (v2i16 (shl (build_vector (vector_extract PulpV4:$rs2, (i32 0)), (vector_extract PulpV4:$rs2, (i32 1))), (v2i16 (splat_vector (i32 8))))), (v2i16 (splat_vector (i32 8))))), (v2i16 (sra (v2i16 (shl (build_vector (vector_extract PulpV4:$rs1, (i32 0)), (vector_extract PulpV4:$rs1, (i32 1))), (v2i16 (splat_vector (i32 8))))), (v2i16 (splat_vector (i32 8)))))), (v2i16 (splat_vector (i32 1))))), (i32 0)), (vector_extract (v2i16 (srl (add (v2i16 (sra (v2i16 (shl (build_vector (vector_extract PulpV4:$rs2, (i32 0)), (vector_extract PulpV4:$rs2, (i32 1))), (v2i16 (splat_vector (i32 8))))), (v2i16 (splat_vector (i32 8))))), (v2i16 (sra (v2i16 (shl (build_vector (vector_extract PulpV4:$rs1, (i32 0)), (vector_extract PulpV4:$rs1, (i32 1))), (v2i16 (splat_vector (i32 8))))), (v2i16 (splat_vector (i32 8)))))), (v2i16 (splat_vector (i32 1))))), (i32 1)), (vector_extract (v2i16 (srl (add (v2i16 (sra (v2i16 (shl (build_vector (vector_extract PulpV4:$rs2, (i32 2)), (vector_extract PulpV4:$rs2, (i32 3))), (v2i16 (splat_vector (i32 8))))), (v2i16 (splat_vector (i32 8))))), (v2i16 (sra (v2i16 (shl (build_vector (vector_extract PulpV4:$rs1, (i32 2)), (vector_extract PulpV4:$rs1, (i32 3))), (v2i16 (splat_vector (i32 8))))), (v2i16 (splat_vector (i32 8)))))), (v2i16 (splat_vector (i32 1))))), (i32 0)), (vector_extract (v2i16 (srl (add (v2i16 (sra (v2i16 (shl (build_vector (vector_extract PulpV4:$rs2, (i32 2)), (vector_extract PulpV4:$rs2, (i32 3))), (v2i16 (splat_vector (i32 8))))), (v2i16 (splat_vector (i32 8))))), (v2i16 (sra (v2i16 (shl (build_vector (vector_extract PulpV4:$rs1, (i32 2)), (vector_extract PulpV4:$rs1, (i32 3))), (v2i16 (splat_vector (i32 8))))), (v2i16 (splat_vector (i32 8)))))), (v2i16 (splat_vector (i32 1))))), (i32 1))),
	(CV_AVG_B__V4_V4_V4 PulpV4:$rs1, PulpV4:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_AVG_SC_B__V4_V4_S : RVInst_CV_AVG_SC_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, GPR:$rs2)>;

def : Pat<
	(build_vector (vector_extract (v2i16 (srl (add (v2i16 (splat_vector (sra (shl GPR:$rs2, (i32 24)), (i32 24)))), (v2i16 (sra (v2i16 (shl (build_vector (vector_extract PulpV4:$rs1, (i32 0)), (vector_extract PulpV4:$rs1, (i32 1))), (v2i16 (splat_vector (i32 8))))), (v2i16 (splat_vector (i32 8)))))), (v2i16 (splat_vector (i32 1))))), (i32 0)), (vector_extract (v2i16 (srl (add (v2i16 (splat_vector (sra (shl GPR:$rs2, (i32 24)), (i32 24)))), (v2i16 (sra (v2i16 (shl (build_vector (vector_extract PulpV4:$rs1, (i32 0)), (vector_extract PulpV4:$rs1, (i32 1))), (v2i16 (splat_vector (i32 8))))), (v2i16 (splat_vector (i32 8)))))), (v2i16 (splat_vector (i32 1))))), (i32 1)), (vector_extract (v2i16 (srl (add (v2i16 (splat_vector (sra (shl GPR:$rs2, (i32 24)), (i32 24)))), (v2i16 (sra (v2i16 (shl (build_vector (vector_extract PulpV4:$rs1, (i32 2)), (vector_extract PulpV4:$rs1, (i32 3))), (v2i16 (splat_vector (i32 8))))), (v2i16 (splat_vector (i32 8)))))), (v2i16 (splat_vector (i32 1))))), (i32 0)), (vector_extract (v2i16 (srl (add (v2i16 (splat_vector (sra (shl GPR:$rs2, (i32 24)), (i32 24)))), (v2i16 (sra (v2i16 (shl (build_vector (vector_extract PulpV4:$rs1, (i32 2)), (vector_extract PulpV4:$rs1, (i32 3))), (v2i16 (splat_vector (i32 8))))), (v2i16 (splat_vector (i32 8)))))), (v2i16 (splat_vector (i32 1))))), (i32 1))),
	(CV_AVG_SC_B__V4_V4_S PulpV4:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_AVG_SCI_B__V4_V4 : RVInst_CV_AVG_SCI_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, simm6:$imm)>;

def : Pat<
	(build_vector (vector_extract (v2i16 (srl (add (v2i16 (splat_vector (i32 (sra (i32 (shl (i32 simm6:$imm), (i32 24))), (i32 24))))), (v2i16 (sra (v2i16 (shl (build_vector (vector_extract PulpV4:$rs1, (i32 0)), (vector_extract PulpV4:$rs1, (i32 1))), (v2i16 (splat_vector (i32 8))))), (v2i16 (splat_vector (i32 8)))))), (v2i16 (splat_vector (i32 1))))), (i32 0)), (vector_extract (v2i16 (srl (add (v2i16 (splat_vector (i32 (sra (i32 (shl (i32 simm6:$imm), (i32 24))), (i32 24))))), (v2i16 (sra (v2i16 (shl (build_vector (vector_extract PulpV4:$rs1, (i32 0)), (vector_extract PulpV4:$rs1, (i32 1))), (v2i16 (splat_vector (i32 8))))), (v2i16 (splat_vector (i32 8)))))), (v2i16 (splat_vector (i32 1))))), (i32 1)), (vector_extract (v2i16 (srl (add (v2i16 (splat_vector (i32 (sra (i32 (shl (i32 simm6:$imm), (i32 24))), (i32 24))))), (v2i16 (sra (v2i16 (shl (build_vector (vector_extract PulpV4:$rs1, (i32 2)), (vector_extract PulpV4:$rs1, (i32 3))), (v2i16 (splat_vector (i32 8))))), (v2i16 (splat_vector (i32 8)))))), (v2i16 (splat_vector (i32 1))))), (i32 0)), (vector_extract (v2i16 (srl (add (v2i16 (splat_vector (i32 (sra (i32 (shl (i32 simm6:$imm), (i32 24))), (i32 24))))), (v2i16 (sra (v2i16 (shl (build_vector (vector_extract PulpV4:$rs1, (i32 2)), (vector_extract PulpV4:$rs1, (i32 3))), (v2i16 (splat_vector (i32 8))))), (v2i16 (splat_vector (i32 8)))))), (v2i16 (splat_vector (i32 1))))), (i32 1))),
	(CV_AVG_SCI_B__V4_V4 PulpV4:$rs1, simm6:$imm)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_AVGU_H__V2_V2_V2 : RVInst_CV_AVGU_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, PulpV2:$rs2)>;

def : Pat<
	(build_vector (i32 (srl (add (and (vector_extract PulpV2:$rs2, (i32 0)), (i32 65535)), (and (vector_extract PulpV2:$rs1, (i32 0)), (i32 65535))), (i32 1))), (i32 (srl (add (and (vector_extract PulpV2:$rs2, (i32 1)), (i32 65535)), (and (vector_extract PulpV2:$rs1, (i32 1)), (i32 65535))), (i32 1)))),
	(CV_AVGU_H__V2_V2_V2 PulpV2:$rs1, PulpV2:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_AVGU_SC_H__V2_V2_S : RVInst_CV_AVGU_SC_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, GPR:$rs2)>;

def : Pat<
	(build_vector (i32 (srl (add (and GPR:$rs2, (i32 65535)), (and (vector_extract PulpV2:$rs1, (i32 0)), (i32 65535))), (i32 1))), (i32 (srl (add (and GPR:$rs2, (i32 65535)), (and (vector_extract PulpV2:$rs1, (i32 1)), (i32 65535))), (i32 1)))),
	(CV_AVGU_SC_H__V2_V2_S PulpV2:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_AVGU_SCI_H__V2_V2 : RVInst_CV_AVGU_SCI_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, uimm6:$imm)>;

def : Pat<
	(build_vector (i32 (srl (add (and (i32 uimm6:$imm), (i32 65535)), (and (vector_extract PulpV2:$rs1, (i32 0)), (i32 65535))), (i32 1))), (i32 (srl (add (and (i32 uimm6:$imm), (i32 65535)), (and (vector_extract PulpV2:$rs1, (i32 1)), (i32 65535))), (i32 1)))),
	(CV_AVGU_SCI_H__V2_V2 PulpV2:$rs1, uimm6:$imm)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_AVGU_B__V4_V4_V4 : RVInst_CV_AVGU_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, PulpV4:$rs2)>;

def : Pat<
	(build_vector (vector_extract (v2i16 (srl (add (and (build_vector (vector_extract PulpV4:$rs2, (i32 0)), (vector_extract PulpV4:$rs2, (i32 1))), (v2i16 (splat_vector (i32 255)))), (and (build_vector (vector_extract PulpV4:$rs1, (i32 0)), (vector_extract PulpV4:$rs1, (i32 1))), (v2i16 (splat_vector (i32 255))))), (v2i16 (splat_vector (i32 1))))), (i32 0)), (vector_extract (v2i16 (srl (add (and (build_vector (vector_extract PulpV4:$rs2, (i32 0)), (vector_extract PulpV4:$rs2, (i32 1))), (v2i16 (splat_vector (i32 255)))), (and (build_vector (vector_extract PulpV4:$rs1, (i32 0)), (vector_extract PulpV4:$rs1, (i32 1))), (v2i16 (splat_vector (i32 255))))), (v2i16 (splat_vector (i32 1))))), (i32 1)), (vector_extract (v2i16 (srl (add (and (build_vector (vector_extract PulpV4:$rs2, (i32 2)), (vector_extract PulpV4:$rs2, (i32 3))), (v2i16 (splat_vector (i32 255)))), (and (build_vector (vector_extract PulpV4:$rs1, (i32 2)), (vector_extract PulpV4:$rs1, (i32 3))), (v2i16 (splat_vector (i32 255))))), (v2i16 (splat_vector (i32 1))))), (i32 0)), (vector_extract (v2i16 (srl (add (and (build_vector (vector_extract PulpV4:$rs2, (i32 2)), (vector_extract PulpV4:$rs2, (i32 3))), (v2i16 (splat_vector (i32 255)))), (and (build_vector (vector_extract PulpV4:$rs1, (i32 2)), (vector_extract PulpV4:$rs1, (i32 3))), (v2i16 (splat_vector (i32 255))))), (v2i16 (splat_vector (i32 1))))), (i32 1))),
	(CV_AVGU_B__V4_V4_V4 PulpV4:$rs1, PulpV4:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_AVGU_SC_B__V4_V4_S : RVInst_CV_AVGU_SC_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, GPR:$rs2)>;

def : Pat<
	(build_vector (vector_extract (v2i16 (srl (add (v2i16 (splat_vector (and GPR:$rs2, (i32 255)))), (and (build_vector (vector_extract PulpV4:$rs1, (i32 0)), (vector_extract PulpV4:$rs1, (i32 1))), (v2i16 (splat_vector (i32 255))))), (v2i16 (splat_vector (i32 1))))), (i32 0)), (vector_extract (v2i16 (srl (add (v2i16 (splat_vector (and GPR:$rs2, (i32 255)))), (and (build_vector (vector_extract PulpV4:$rs1, (i32 0)), (vector_extract PulpV4:$rs1, (i32 1))), (v2i16 (splat_vector (i32 255))))), (v2i16 (splat_vector (i32 1))))), (i32 1)), (vector_extract (v2i16 (srl (add (v2i16 (splat_vector (and GPR:$rs2, (i32 255)))), (and (build_vector (vector_extract PulpV4:$rs1, (i32 2)), (vector_extract PulpV4:$rs1, (i32 3))), (v2i16 (splat_vector (i32 255))))), (v2i16 (splat_vector (i32 1))))), (i32 0)), (vector_extract (v2i16 (srl (add (v2i16 (splat_vector (and GPR:$rs2, (i32 255)))), (and (build_vector (vector_extract PulpV4:$rs1, (i32 2)), (vector_extract PulpV4:$rs1, (i32 3))), (v2i16 (splat_vector (i32 255))))), (v2i16 (splat_vector (i32 1))))), (i32 1))),
	(CV_AVGU_SC_B__V4_V4_S PulpV4:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_AVGU_SCI_B__V4_V4 : RVInst_CV_AVGU_SCI_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, uimm6:$imm)>;

def : Pat<
	(build_vector (vector_extract (v2i16 (srl (add (v2i16 (splat_vector (and (i32 uimm6:$imm), (i32 255)))), (and (build_vector (vector_extract PulpV4:$rs1, (i32 0)), (vector_extract PulpV4:$rs1, (i32 1))), (v2i16 (splat_vector (i32 255))))), (v2i16 (splat_vector (i32 1))))), (i32 0)), (vector_extract (v2i16 (srl (add (v2i16 (splat_vector (and (i32 uimm6:$imm), (i32 255)))), (and (build_vector (vector_extract PulpV4:$rs1, (i32 0)), (vector_extract PulpV4:$rs1, (i32 1))), (v2i16 (splat_vector (i32 255))))), (v2i16 (splat_vector (i32 1))))), (i32 1)), (vector_extract (v2i16 (srl (add (v2i16 (splat_vector (and (i32 uimm6:$imm), (i32 255)))), (and (build_vector (vector_extract PulpV4:$rs1, (i32 2)), (vector_extract PulpV4:$rs1, (i32 3))), (v2i16 (splat_vector (i32 255))))), (v2i16 (splat_vector (i32 1))))), (i32 0)), (vector_extract (v2i16 (srl (add (v2i16 (splat_vector (and (i32 uimm6:$imm), (i32 255)))), (and (build_vector (vector_extract PulpV4:$rs1, (i32 2)), (vector_extract PulpV4:$rs1, (i32 3))), (v2i16 (splat_vector (i32 255))))), (v2i16 (splat_vector (i32 1))))), (i32 1))),
	(CV_AVGU_SCI_B__V4_V4 PulpV4:$rs1, uimm6:$imm)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_MIN_H__V2_V2_V2 : RVInst_CV_MIN_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, PulpV2:$rs2)>;

def : Pat<
	(smin PulpV2:$rs1, PulpV2:$rs2),
	(CV_MIN_H__V2_V2_V2 PulpV2:$rs1, PulpV2:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_MIN_SC_H__V2_V2_S : RVInst_CV_MIN_SC_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, GPR:$rs2)>;

def : Pat<
	(smin PulpV2:$rs1, (v2i16 (splat_vector GPR:$rs2))),
	(CV_MIN_SC_H__V2_V2_S PulpV2:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_MIN_SCI_H__V2_V2 : RVInst_CV_MIN_SCI_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, simm6:$imm)>;

def : Pat<
	(smin PulpV2:$rs1, (v2i16 (splat_vector (i32 simm6:$imm)))),
	(CV_MIN_SCI_H__V2_V2 PulpV2:$rs1, simm6:$imm)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_MIN_B__V4_V4_V4 : RVInst_CV_MIN_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, PulpV4:$rs2)>;

def : Pat<
	(smin PulpV4:$rs1, PulpV4:$rs2),
	(CV_MIN_B__V4_V4_V4 PulpV4:$rs1, PulpV4:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_MIN_SC_B__V4_V4_S : RVInst_CV_MIN_SC_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, GPR:$rs2)>;

def : Pat<
	(smin PulpV4:$rs1, (v4i8 (splat_vector GPR:$rs2))),
	(CV_MIN_SC_B__V4_V4_S PulpV4:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_MIN_SCI_B__V4_V4 : RVInst_CV_MIN_SCI_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, simm6:$imm)>;

def : Pat<
	(smin PulpV4:$rs1, (v4i8 (splat_vector (i32 simm6:$imm)))),
	(CV_MIN_SCI_B__V4_V4 PulpV4:$rs1, simm6:$imm)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_MINU_H__V2_V2_V2 : RVInst_CV_MINU_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, PulpV2:$rs2)>;

def : Pat<
	(umin PulpV2:$rs1, PulpV2:$rs2),
	(CV_MINU_H__V2_V2_V2 PulpV2:$rs1, PulpV2:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_MINU_SC_H__V2_V2_S : RVInst_CV_MINU_SC_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, GPR:$rs2)>;

def : Pat<
	(umin PulpV2:$rs1, (v2i16 (splat_vector GPR:$rs2))),
	(CV_MINU_SC_H__V2_V2_S PulpV2:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_MINU_SCI_H__V2_V2 : RVInst_CV_MINU_SCI_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, uimm6:$imm)>;

def : Pat<
	(umin PulpV2:$rs1, (v2i16 (splat_vector (i32 uimm6:$imm)))),
	(CV_MINU_SCI_H__V2_V2 PulpV2:$rs1, uimm6:$imm)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_MINU_B__V4_V4_V4 : RVInst_CV_MINU_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, PulpV4:$rs2)>;

def : Pat<
	(umin PulpV4:$rs1, PulpV4:$rs2),
	(CV_MINU_B__V4_V4_V4 PulpV4:$rs1, PulpV4:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_MINU_SC_B__V4_V4_S : RVInst_CV_MINU_SC_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, GPR:$rs2)>;

def : Pat<
	(umin PulpV4:$rs1, (v4i8 (splat_vector GPR:$rs2))),
	(CV_MINU_SC_B__V4_V4_S PulpV4:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_MINU_SCI_B__V4_V4 : RVInst_CV_MINU_SCI_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, uimm6:$imm)>;

def : Pat<
	(umin PulpV4:$rs1, (v4i8 (splat_vector (i32 uimm6:$imm)))),
	(CV_MINU_SCI_B__V4_V4 PulpV4:$rs1, uimm6:$imm)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_MAX_H__V2_V2_V2 : RVInst_CV_MAX_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, PulpV2:$rs2)>;

def : Pat<
	(smax PulpV2:$rs1, PulpV2:$rs2),
	(CV_MAX_H__V2_V2_V2 PulpV2:$rs1, PulpV2:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_MAX_SC_H__V2_V2_S : RVInst_CV_MAX_SC_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, GPR:$rs2)>;

def : Pat<
	(smax PulpV2:$rs1, (v2i16 (splat_vector GPR:$rs2))),
	(CV_MAX_SC_H__V2_V2_S PulpV2:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_MAX_SCI_H__V2_V2 : RVInst_CV_MAX_SCI_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, simm6:$imm)>;

def : Pat<
	(smax PulpV2:$rs1, (v2i16 (splat_vector (i32 simm6:$imm)))),
	(CV_MAX_SCI_H__V2_V2 PulpV2:$rs1, simm6:$imm)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_MAX_B__V4_V4_V4 : RVInst_CV_MAX_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, PulpV4:$rs2)>;

def : Pat<
	(smax PulpV4:$rs1, PulpV4:$rs2),
	(CV_MAX_B__V4_V4_V4 PulpV4:$rs1, PulpV4:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_MAX_SC_B__V4_V4_S : RVInst_CV_MAX_SC_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, GPR:$rs2)>;

def : Pat<
	(smax PulpV4:$rs1, (v4i8 (splat_vector GPR:$rs2))),
	(CV_MAX_SC_B__V4_V4_S PulpV4:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_MAX_SCI_B__V4_V4 : RVInst_CV_MAX_SCI_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, simm6:$imm)>;

def : Pat<
	(smax PulpV4:$rs1, (v4i8 (splat_vector (i32 simm6:$imm)))),
	(CV_MAX_SCI_B__V4_V4 PulpV4:$rs1, simm6:$imm)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_MAXU_H__V2_V2_V2 : RVInst_CV_MAXU_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, PulpV2:$rs2)>;

def : Pat<
	(umax PulpV2:$rs1, PulpV2:$rs2),
	(CV_MAXU_H__V2_V2_V2 PulpV2:$rs1, PulpV2:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_MAXU_SC_H__V2_V2_S : RVInst_CV_MAXU_SC_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, GPR:$rs2)>;

def : Pat<
	(umax PulpV2:$rs1, (v2i16 (splat_vector GPR:$rs2))),
	(CV_MAXU_SC_H__V2_V2_S PulpV2:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_MAXU_SCI_H__V2_V2 : RVInst_CV_MAXU_SCI_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, uimm6:$imm)>;

def : Pat<
	(umax PulpV2:$rs1, (v2i16 (splat_vector (i32 uimm6:$imm)))),
	(CV_MAXU_SCI_H__V2_V2 PulpV2:$rs1, uimm6:$imm)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_MAXU_B__V4_V4_V4 : RVInst_CV_MAXU_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, PulpV4:$rs2)>;

def : Pat<
	(umax PulpV4:$rs1, PulpV4:$rs2),
	(CV_MAXU_B__V4_V4_V4 PulpV4:$rs1, PulpV4:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_MAXU_SC_B__V4_V4_S : RVInst_CV_MAXU_SC_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, GPR:$rs2)>;

def : Pat<
	(umax PulpV4:$rs1, (v4i8 (splat_vector GPR:$rs2))),
	(CV_MAXU_SC_B__V4_V4_S PulpV4:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_MAXU_SCI_B__V4_V4 : RVInst_CV_MAXU_SCI_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, uimm6:$imm)>;

def : Pat<
	(umax PulpV4:$rs1, (v4i8 (splat_vector (i32 uimm6:$imm)))),
	(CV_MAXU_SCI_B__V4_V4 PulpV4:$rs1, uimm6:$imm)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_SRL_H__V2_V2_V2 : RVInst_CV_SRL_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, PulpV2:$rs2)>;

def : Pat<
	(v2i16 (srl PulpV2:$rs1, PulpV2:$rs2)),
	(CV_SRL_H__V2_V2_V2 PulpV2:$rs1, PulpV2:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_SRL_SC_H__V2_V2_S : RVInst_CV_SRL_SC_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, GPR:$rs2)>;

def : Pat<
	(v2i16 (srl PulpV2:$rs1, (v2i16 (splat_vector GPR:$rs2)))),
	(CV_SRL_SC_H__V2_V2_S PulpV2:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_SRL_SCI_H__V2_V2 : RVInst_CV_SRL_SCI_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, uimm6:$imm)>;

def : Pat<
	(v2i16 (srl PulpV2:$rs1, (v2i16 (splat_vector (i32 uimm6:$imm))))),
	(CV_SRL_SCI_H__V2_V2 PulpV2:$rs1, uimm6:$imm)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_SRL_B__V4_V4_V4 : RVInst_CV_SRL_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, PulpV4:$rs2)>;

def : Pat<
	(v4i8 (srl PulpV4:$rs1, PulpV4:$rs2)),
	(CV_SRL_B__V4_V4_V4 PulpV4:$rs1, PulpV4:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_SRL_SC_B__V4_V4_S : RVInst_CV_SRL_SC_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, GPR:$rs2)>;

def : Pat<
	(v4i8 (srl PulpV4:$rs1, (v4i8 (splat_vector GPR:$rs2)))),
	(CV_SRL_SC_B__V4_V4_S PulpV4:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_SRL_SCI_B__V4_V4 : RVInst_CV_SRL_SCI_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, uimm6:$imm)>;

def : Pat<
	(v4i8 (srl PulpV4:$rs1, (v4i8 (splat_vector (i32 uimm6:$imm))))),
	(CV_SRL_SCI_B__V4_V4 PulpV4:$rs1, uimm6:$imm)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_SRA_H__V2_V2_V2 : RVInst_CV_SRA_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, PulpV2:$rs2)>;

def : Pat<
	(v2i16 (sra PulpV2:$rs1, PulpV2:$rs2)),
	(CV_SRA_H__V2_V2_V2 PulpV2:$rs1, PulpV2:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_SRA_SC_H__V2_V2_S : RVInst_CV_SRA_SC_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, GPR:$rs2)>;

def : Pat<
	(v2i16 (sra PulpV2:$rs1, (v2i16 (splat_vector GPR:$rs2)))),
	(CV_SRA_SC_H__V2_V2_S PulpV2:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_SRA_SCI_H__V2_V2 : RVInst_CV_SRA_SCI_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, uimm6:$imm)>;

def : Pat<
	(v2i16 (sra PulpV2:$rs1, (v2i16 (splat_vector (i32 uimm6:$imm))))),
	(CV_SRA_SCI_H__V2_V2 PulpV2:$rs1, uimm6:$imm)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_SRA_B__V4_V4_V4 : RVInst_CV_SRA_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, PulpV4:$rs2)>;

def : Pat<
	(v4i8 (sra PulpV4:$rs1, PulpV4:$rs2)),
	(CV_SRA_B__V4_V4_V4 PulpV4:$rs1, PulpV4:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_SRA_SC_B__V4_V4_S : RVInst_CV_SRA_SC_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, GPR:$rs2)>;

def : Pat<
	(v4i8 (sra PulpV4:$rs1, (v4i8 (splat_vector GPR:$rs2)))),
	(CV_SRA_SC_B__V4_V4_S PulpV4:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_SRA_SCI_B__V4_V4 : RVInst_CV_SRA_SCI_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, uimm6:$imm)>;

def : Pat<
	(v4i8 (sra PulpV4:$rs1, (v4i8 (splat_vector (i32 uimm6:$imm))))),
	(CV_SRA_SCI_B__V4_V4 PulpV4:$rs1, uimm6:$imm)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_SLL_H__V2_V2_V2 : RVInst_CV_SLL_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, PulpV2:$rs2)>;

def : Pat<
	(v2i16 (shl PulpV2:$rs1, PulpV2:$rs2)),
	(CV_SLL_H__V2_V2_V2 PulpV2:$rs1, PulpV2:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_SLL_SC_H__V2_V2_S : RVInst_CV_SLL_SC_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, GPR:$rs2)>;

def : Pat<
	(v2i16 (shl PulpV2:$rs1, (v2i16 (splat_vector GPR:$rs2)))),
	(CV_SLL_SC_H__V2_V2_S PulpV2:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_SLL_SCI_H__V2_V2 : RVInst_CV_SLL_SCI_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, uimm6:$imm)>;

def : Pat<
	(v2i16 (shl PulpV2:$rs1, (v2i16 (splat_vector (i32 uimm6:$imm))))),
	(CV_SLL_SCI_H__V2_V2 PulpV2:$rs1, uimm6:$imm)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_SLL_B__V4_V4_V4 : RVInst_CV_SLL_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, PulpV4:$rs2)>;

def : Pat<
	(v4i8 (shl PulpV4:$rs1, PulpV4:$rs2)),
	(CV_SLL_B__V4_V4_V4 PulpV4:$rs1, PulpV4:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_SLL_SC_B__V4_V4_S : RVInst_CV_SLL_SC_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, GPR:$rs2)>;

def : Pat<
	(v4i8 (shl PulpV4:$rs1, (v4i8 (splat_vector GPR:$rs2)))),
	(CV_SLL_SC_B__V4_V4_S PulpV4:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_SLL_SCI_B__V4_V4 : RVInst_CV_SLL_SCI_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, uimm6:$imm)>;

def : Pat<
	(v4i8 (shl PulpV4:$rs1, (v4i8 (splat_vector (i32 uimm6:$imm))))),
	(CV_SLL_SCI_B__V4_V4 PulpV4:$rs1, uimm6:$imm)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_OR_H__V2_V2_V2 : RVInst_CV_OR_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, PulpV2:$rs2)>;

def : Pat<
	(or PulpV2:$rs2, PulpV2:$rs1),
	(CV_OR_H__V2_V2_V2 PulpV2:$rs1, PulpV2:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_OR_SC_H__V2_V2_S : RVInst_CV_OR_SC_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, GPR:$rs2)>;

def : Pat<
	(or PulpV2:$rs1, (v2i16 (splat_vector GPR:$rs2))),
	(CV_OR_SC_H__V2_V2_S PulpV2:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_OR_SCI_H__V2_V2 : RVInst_CV_OR_SCI_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, simm6:$imm)>;

def : Pat<
	(or PulpV2:$rs1, (v2i16 (splat_vector (i32 simm6:$imm)))),
	(CV_OR_SCI_H__V2_V2 PulpV2:$rs1, simm6:$imm)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_OR_B__V4_V4_V4 : RVInst_CV_OR_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, PulpV4:$rs2)>;

def : Pat<
	(or PulpV4:$rs2, PulpV4:$rs1),
	(CV_OR_B__V4_V4_V4 PulpV4:$rs1, PulpV4:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_OR_SC_B__V4_V4_S : RVInst_CV_OR_SC_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, GPR:$rs2)>;

def : Pat<
	(or PulpV4:$rs1, (v4i8 (splat_vector GPR:$rs2))),
	(CV_OR_SC_B__V4_V4_S PulpV4:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_OR_SCI_B__V4_V4 : RVInst_CV_OR_SCI_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, simm6:$imm)>;

def : Pat<
	(or PulpV4:$rs1, (v4i8 (splat_vector (i32 simm6:$imm)))),
	(CV_OR_SCI_B__V4_V4 PulpV4:$rs1, simm6:$imm)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_XOR_H__V2_V2_V2 : RVInst_CV_XOR_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, PulpV2:$rs2)>;

def : Pat<
	(xor PulpV2:$rs2, PulpV2:$rs1),
	(CV_XOR_H__V2_V2_V2 PulpV2:$rs1, PulpV2:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_XOR_SC_H__V2_V2_S : RVInst_CV_XOR_SC_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, GPR:$rs2)>;

def : Pat<
	(xor PulpV2:$rs1, (v2i16 (splat_vector GPR:$rs2))),
	(CV_XOR_SC_H__V2_V2_S PulpV2:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_XOR_SCI_H__V2_V2 : RVInst_CV_XOR_SCI_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, simm6:$imm)>;

def : Pat<
	(xor PulpV2:$rs1, (v2i16 (splat_vector (i32 simm6:$imm)))),
	(CV_XOR_SCI_H__V2_V2 PulpV2:$rs1, simm6:$imm)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_XOR_B__V4_V4_V4 : RVInst_CV_XOR_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, PulpV4:$rs2)>;

def : Pat<
	(xor PulpV4:$rs2, PulpV4:$rs1),
	(CV_XOR_B__V4_V4_V4 PulpV4:$rs1, PulpV4:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_XOR_SC_B__V4_V4_S : RVInst_CV_XOR_SC_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, GPR:$rs2)>;

def : Pat<
	(xor PulpV4:$rs1, (v4i8 (splat_vector GPR:$rs2))),
	(CV_XOR_SC_B__V4_V4_S PulpV4:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_XOR_SCI_B__V4_V4 : RVInst_CV_XOR_SCI_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, simm6:$imm)>;

def : Pat<
	(xor PulpV4:$rs1, (v4i8 (splat_vector (i32 simm6:$imm)))),
	(CV_XOR_SCI_B__V4_V4 PulpV4:$rs1, simm6:$imm)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_AND_H__V2_V2_V2 : RVInst_CV_AND_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, PulpV2:$rs2)>;

def : Pat<
	(and PulpV2:$rs2, PulpV2:$rs1),
	(CV_AND_H__V2_V2_V2 PulpV2:$rs1, PulpV2:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_AND_SC_H__V2_V2_S : RVInst_CV_AND_SC_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, GPR:$rs2)>;

def : Pat<
	(and PulpV2:$rs1, (v2i16 (splat_vector GPR:$rs2))),
	(CV_AND_SC_H__V2_V2_S PulpV2:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_AND_SCI_H__V2_V2 : RVInst_CV_AND_SCI_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, simm6:$imm)>;

def : Pat<
	(and PulpV2:$rs1, (v2i16 (splat_vector (i32 simm6:$imm)))),
	(CV_AND_SCI_H__V2_V2 PulpV2:$rs1, simm6:$imm)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_AND_B__V4_V4_V4 : RVInst_CV_AND_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, PulpV4:$rs2)>;

def : Pat<
	(and PulpV4:$rs2, PulpV4:$rs1),
	(CV_AND_B__V4_V4_V4 PulpV4:$rs1, PulpV4:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_AND_SC_B__V4_V4_S : RVInst_CV_AND_SC_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, GPR:$rs2)>;

def : Pat<
	(and PulpV4:$rs1, (v4i8 (splat_vector GPR:$rs2))),
	(CV_AND_SC_B__V4_V4_S PulpV4:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_AND_SCI_B__V4_V4 : RVInst_CV_AND_SCI_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, simm6:$imm)>;

def : Pat<
	(and PulpV4:$rs1, (v4i8 (splat_vector (i32 simm6:$imm)))),
	(CV_AND_SCI_B__V4_V4 PulpV4:$rs1, simm6:$imm)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_ABS_H__V2_V2 : RVInst_CV_ABS_H<(outs PulpV2:$rd), (ins PulpV2:$rs1)>;

def : Pat<
	(v2i16 (abs PulpV2:$rs1)),
	(CV_ABS_H__V2_V2 PulpV2:$rs1)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_ABS_B__V4_V4 : RVInst_CV_ABS_B<(outs PulpV4:$rd), (ins PulpV4:$rs1)>;

def : Pat<
	(v4i8 (abs PulpV4:$rs1)),
	(CV_ABS_B__V4_V4 PulpV4:$rs1)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_DOTUP_H__S_V2_V2 : RVInst_CV_DOTUP_H<(outs GPR:$rd), (ins PulpV2:$rs1, PulpV2:$rs2)>;

def : Pat<
	(add (mul (and (i32 (vector_extract PulpV2:$rs2, 1)), (i32 65535)), (and (i32 (vector_extract PulpV2:$rs1, 1)), (i32 65535))), (mul (and (i32 (vector_extract PulpV2:$rs2, 0)), (i32 65535)), (and (i32 (vector_extract PulpV2:$rs1, 0)), (i32 65535)))),
	(CV_DOTUP_H__S_V2_V2 PulpV2:$rs1, PulpV2:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_DOTUP_SC_H__S_V2_S : RVInst_CV_DOTUP_SC_H<(outs GPR:$rd), (ins PulpV2:$rs1, GPR:$rs2)>;

def : Pat<
	(mul (add (and (i32 (vector_extract PulpV2:$rs1, 1)), (i32 65535)), (and (i32 (vector_extract PulpV2:$rs1, 0)), (i32 65535))), (and GPR:$rs2, (i32 65535))),
	(CV_DOTUP_SC_H__S_V2_S PulpV2:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_DOTUP_SCI_H__S_V2 : RVInst_CV_DOTUP_SCI_H<(outs GPR:$rd), (ins PulpV2:$rs1, simm6:$imm)>;

def : Pat<
	(mul (add (and (i32 (vector_extract PulpV2:$rs1, 1)), (i32 65535)), (and (i32 (vector_extract PulpV2:$rs1, 0)), (i32 65535))), (and (i32 simm6:$imm), (i32 65535))),
	(CV_DOTUP_SCI_H__S_V2 PulpV2:$rs1, simm6:$imm)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_DOTUP_B__S_V4_V4 : RVInst_CV_DOTUP_B<(outs GPR:$rd), (ins PulpV4:$rs1, PulpV4:$rs2)>;

def : Pat<
	(add (add (mul (and (vector_extract PulpV4:$rs2, (i32 0)), (i32 255)), (and (vector_extract PulpV4:$rs1, (i32 0)), (i32 255))), (mul (and (vector_extract PulpV4:$rs2, (i32 2)), (i32 255)), (and (vector_extract PulpV4:$rs1, (i32 2)), (i32 255)))), (add (mul (and (vector_extract PulpV4:$rs2, (i32 1)), (i32 255)), (and (vector_extract PulpV4:$rs1, (i32 1)), (i32 255))), (mul (and (vector_extract PulpV4:$rs2, (i32 3)), (i32 255)), (and (vector_extract PulpV4:$rs1, (i32 3)), (i32 255))))),
	(CV_DOTUP_B__S_V4_V4 PulpV4:$rs1, PulpV4:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_DOTUP_SC_B__S_V4_S : RVInst_CV_DOTUP_SC_B<(outs GPR:$rd), (ins PulpV4:$rs1, GPR:$rs2)>;

def : Pat<
	(add (add (add (mul (and (i32 (vector_extract PulpV4:$rs1, 1)), (i32 255)), (and GPR:$rs2, (i32 255))), (mul (and GPR:$rs2, (i32 255)), (and (i32 (vector_extract PulpV4:$rs1, 0)), (i32 255)))), (mul (and (i32 (vector_extract PulpV4:$rs1, 2)), (i32 255)), (and GPR:$rs2, (i32 255)))), (mul (and (i32 (vector_extract PulpV4:$rs1, 3)), (i32 255)), (and GPR:$rs2, (i32 255)))),
	(CV_DOTUP_SC_B__S_V4_S PulpV4:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_DOTUP_SCI_B__S_V4 : RVInst_CV_DOTUP_SCI_B<(outs GPR:$rd), (ins PulpV4:$rs1, simm6:$imm)>;

def : Pat<
	(add (add (add (mul (and (i32 simm6:$imm), (i32 255)), (and (i32 (vector_extract PulpV4:$rs1, 1)), (i32 255))), (mul (and (i32 simm6:$imm), (i32 255)), (and (i32 (vector_extract PulpV4:$rs1, 0)), (i32 255)))), (mul (and (i32 simm6:$imm), (i32 255)), (and (i32 (vector_extract PulpV4:$rs1, 2)), (i32 255)))), (mul (and (i32 simm6:$imm), (i32 255)), (and (i32 (vector_extract PulpV4:$rs1, 3)), (i32 255)))),
	(CV_DOTUP_SCI_B__S_V4 PulpV4:$rs1, simm6:$imm)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_DOTUSP_H__S_V2_V2 : RVInst_CV_DOTUSP_H<(outs GPR:$rd), (ins PulpV2:$rs1, PulpV2:$rs2)>;

def : Pat<
	(add (mul (sra (shl (i32 (vector_extract PulpV2:$rs2, 1)), (i32 16)), (i32 16)), (and (i32 (vector_extract PulpV2:$rs1, 1)), (i32 65535))), (mul (sra (shl (i32 (vector_extract PulpV2:$rs2, 0)), (i32 16)), (i32 16)), (and (i32 (vector_extract PulpV2:$rs1, 0)), (i32 65535)))),
	(CV_DOTUSP_H__S_V2_V2 PulpV2:$rs1, PulpV2:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_DOTUSP_SC_H__S_V2_S : RVInst_CV_DOTUSP_SC_H<(outs GPR:$rd), (ins PulpV2:$rs1, GPR:$rs2)>;

def : Pat<
	(mul (add (and (i32 (vector_extract PulpV2:$rs1, 1)), (i32 65535)), (and (i32 (vector_extract PulpV2:$rs1, 0)), (i32 65535))), (sra (shl GPR:$rs2, (i32 16)), (i32 16))),
	(CV_DOTUSP_SC_H__S_V2_S PulpV2:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_DOTUSP_SCI_H__S_V2 : RVInst_CV_DOTUSP_SCI_H<(outs GPR:$rd), (ins PulpV2:$rs1, simm6:$imm)>;

def : Pat<
	(mul (add (and (i32 (vector_extract PulpV2:$rs1, 1)), (i32 65535)), (and (i32 (vector_extract PulpV2:$rs1, 0)), (i32 65535))), (i32 (sra (i32 (shl (i32 simm6:$imm), (i32 16))), (i32 16)))),
	(CV_DOTUSP_SCI_H__S_V2 PulpV2:$rs1, simm6:$imm)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_DOTUSP_B__S_V4_V4 : RVInst_CV_DOTUSP_B<(outs GPR:$rd), (ins PulpV4:$rs1, PulpV4:$rs2)>;

def : Pat<
	(add (add (mul (i32 (sra (i32 (shl (vector_extract PulpV4:$rs2, (i32 0)), (i32 24))), (i32 24))), (and (vector_extract PulpV4:$rs1, (i32 0)), (i32 255))), (mul (i32 (sra (i32 (shl (vector_extract PulpV4:$rs2, (i32 2)), (i32 24))), (i32 24))), (and (vector_extract PulpV4:$rs1, (i32 2)), (i32 255)))), (add (mul (i32 (sra (i32 (shl (vector_extract PulpV4:$rs2, (i32 1)), (i32 24))), (i32 24))), (and (vector_extract PulpV4:$rs1, (i32 1)), (i32 255))), (mul (i32 (sra (i32 (shl (vector_extract PulpV4:$rs2, (i32 3)), (i32 24))), (i32 24))), (and (vector_extract PulpV4:$rs1, (i32 3)), (i32 255))))),
	(CV_DOTUSP_B__S_V4_V4 PulpV4:$rs1, PulpV4:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_DOTUSP_SC_B__S_V4_S : RVInst_CV_DOTUSP_SC_B<(outs GPR:$rd), (ins PulpV4:$rs1, GPR:$rs2)>;

def : Pat<
	(add (add (add (i32 (sra (i32 (shl (mul (and (i32 (vector_extract PulpV4:$rs1, 1)), (i32 255)), (sra (shl GPR:$rs2, (i32 24)), (i32 24))), (i32 16))), (i32 16))), (i32 (sra (i32 (shl (mul (sra (shl GPR:$rs2, (i32 24)), (i32 24)), (and (i32 (vector_extract PulpV4:$rs1, 0)), (i32 255))), (i32 16))), (i32 16)))), (i32 (sra (i32 (shl (mul (and (i32 (vector_extract PulpV4:$rs1, 2)), (i32 255)), (sra (shl GPR:$rs2, (i32 24)), (i32 24))), (i32 16))), (i32 16)))), (i32 (sra (i32 (shl (mul (and (i32 (vector_extract PulpV4:$rs1, 3)), (i32 255)), (sra (shl GPR:$rs2, (i32 24)), (i32 24))), (i32 16))), (i32 16)))),
	(CV_DOTUSP_SC_B__S_V4_S PulpV4:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_DOTUSP_SCI_B__S_V4 : RVInst_CV_DOTUSP_SCI_B<(outs GPR:$rd), (ins PulpV4:$rs1, simm6:$imm)>;

def : Pat<
	(add (add (add (i32 (sra (i32 (shl (mul (and (i32 (vector_extract PulpV4:$rs1, 1)), (i32 255)), (i32 (sra (i32 (shl (i32 simm6:$imm), (i32 24))), (i32 24)))), (i32 16))), (i32 16))), (i32 (sra (i32 (shl (mul (and (i32 (vector_extract PulpV4:$rs1, 0)), (i32 255)), (i32 (sra (i32 (shl (i32 simm6:$imm), (i32 24))), (i32 24)))), (i32 16))), (i32 16)))), (i32 (sra (i32 (shl (mul (and (i32 (vector_extract PulpV4:$rs1, 2)), (i32 255)), (i32 (sra (i32 (shl (i32 simm6:$imm), (i32 24))), (i32 24)))), (i32 16))), (i32 16)))), (i32 (sra (i32 (shl (mul (and (i32 (vector_extract PulpV4:$rs1, 3)), (i32 255)), (i32 (sra (i32 (shl (i32 simm6:$imm), (i32 24))), (i32 24)))), (i32 16))), (i32 16)))),
	(CV_DOTUSP_SCI_B__S_V4 PulpV4:$rs1, simm6:$imm)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_DOTSP_H__S_V2_V2 : RVInst_CV_DOTSP_H<(outs GPR:$rd), (ins PulpV2:$rs1, PulpV2:$rs2)>;

def : Pat<
	(add (mul (sra (shl (i32 (vector_extract PulpV2:$rs2, 1)), (i32 16)), (i32 16)), (sra (shl (i32 (vector_extract PulpV2:$rs1, 1)), (i32 16)), (i32 16))), (mul (sra (shl (i32 (vector_extract PulpV2:$rs2, 0)), (i32 16)), (i32 16)), (sra (shl (i32 (vector_extract PulpV2:$rs1, 0)), (i32 16)), (i32 16)))),
	(CV_DOTSP_H__S_V2_V2 PulpV2:$rs1, PulpV2:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_DOTSP_SC_H__S_V2_S : RVInst_CV_DOTSP_SC_H<(outs GPR:$rd), (ins PulpV2:$rs1, GPR:$rs2)>;

def : Pat<
	(mul (add (sra (shl (i32 (vector_extract PulpV2:$rs1, 1)), (i32 16)), (i32 16)), (sra (shl (i32 (vector_extract PulpV2:$rs1, 0)), (i32 16)), (i32 16))), (sra (shl GPR:$rs2, (i32 16)), (i32 16))),
	(CV_DOTSP_SC_H__S_V2_S PulpV2:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_DOTSP_SCI_H__S_V2 : RVInst_CV_DOTSP_SCI_H<(outs GPR:$rd), (ins PulpV2:$rs1, simm6:$imm)>;

def : Pat<
	(mul (add (sra (shl (i32 (vector_extract PulpV2:$rs1, 1)), (i32 16)), (i32 16)), (sra (shl (i32 (vector_extract PulpV2:$rs1, 0)), (i32 16)), (i32 16))), (i32 (sra (i32 (shl (i32 simm6:$imm), (i32 16))), (i32 16)))),
	(CV_DOTSP_SCI_H__S_V2 PulpV2:$rs1, simm6:$imm)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_DOTSP_B__S_V4_V4 : RVInst_CV_DOTSP_B<(outs GPR:$rd), (ins PulpV4:$rs1, PulpV4:$rs2)>;

def : Pat<
	(add (add (mul (i32 (sra (i32 (shl (vector_extract PulpV4:$rs2, (i32 0)), (i32 24))), (i32 24))), (i32 (sra (i32 (shl (vector_extract PulpV4:$rs1, (i32 0)), (i32 24))), (i32 24)))), (mul (i32 (sra (i32 (shl (vector_extract PulpV4:$rs2, (i32 2)), (i32 24))), (i32 24))), (i32 (sra (i32 (shl (vector_extract PulpV4:$rs1, (i32 2)), (i32 24))), (i32 24))))), (add (mul (i32 (sra (i32 (shl (vector_extract PulpV4:$rs2, (i32 1)), (i32 24))), (i32 24))), (i32 (sra (i32 (shl (vector_extract PulpV4:$rs1, (i32 1)), (i32 24))), (i32 24)))), (mul (i32 (sra (i32 (shl (vector_extract PulpV4:$rs2, (i32 3)), (i32 24))), (i32 24))), (i32 (sra (i32 (shl (vector_extract PulpV4:$rs1, (i32 3)), (i32 24))), (i32 24)))))),
	(CV_DOTSP_B__S_V4_V4 PulpV4:$rs1, PulpV4:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_DOTSP_SC_B__S_V4_S : RVInst_CV_DOTSP_SC_B<(outs GPR:$rd), (ins PulpV4:$rs1, GPR:$rs2)>;

def : Pat<
	(add (add (add (mul (sra (shl (i32 (vector_extract PulpV4:$rs1, 1)), (i32 24)), (i32 24)), (sra (shl GPR:$rs2, (i32 24)), (i32 24))), (mul (sra (shl GPR:$rs2, (i32 24)), (i32 24)), (sra (shl (i32 (vector_extract PulpV4:$rs1, 0)), (i32 24)), (i32 24)))), (mul (sra (shl (i32 (vector_extract PulpV4:$rs1, 2)), (i32 24)), (i32 24)), (sra (shl GPR:$rs2, (i32 24)), (i32 24)))), (mul (sra (shl (i32 (vector_extract PulpV4:$rs1, 3)), (i32 24)), (i32 24)), (sra (shl GPR:$rs2, (i32 24)), (i32 24)))),
	(CV_DOTSP_SC_B__S_V4_S PulpV4:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_DOTSP_SCI_B__S_V4 : RVInst_CV_DOTSP_SCI_B<(outs GPR:$rd), (ins PulpV4:$rs1, simm6:$imm)>;

def : Pat<
	(add (add (add (mul (sra (shl (i32 (vector_extract PulpV4:$rs1, 1)), (i32 24)), (i32 24)), (i32 (sra (i32 (shl (i32 simm6:$imm), (i32 24))), (i32 24)))), (mul (sra (shl (i32 (vector_extract PulpV4:$rs1, 0)), (i32 24)), (i32 24)), (i32 (sra (i32 (shl (i32 simm6:$imm), (i32 24))), (i32 24))))), (mul (sra (shl (i32 (vector_extract PulpV4:$rs1, 2)), (i32 24)), (i32 24)), (i32 (sra (i32 (shl (i32 simm6:$imm), (i32 24))), (i32 24))))), (mul (sra (shl (i32 (vector_extract PulpV4:$rs1, 3)), (i32 24)), (i32 24)), (i32 (sra (i32 (shl (i32 simm6:$imm), (i32 24))), (i32 24))))),
	(CV_DOTSP_SCI_B__S_V4 PulpV4:$rs1, simm6:$imm)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1, Constraints = "$rd = $rd_wb" in def CV_SDOTUP_H__S_S_V2_V2 : RVInst_CV_SDOTUP_H<(outs GPR:$rd_wb), (ins GPR:$rd, PulpV2:$rs1, PulpV2:$rs2)>;

def : Pat<
	(add (add (mul (and (i32 (vector_extract PulpV2:$rs2, 0)), (i32 65535)), (and (i32 (vector_extract PulpV2:$rs1, 0)), (i32 65535))), GPR:$rd), (mul (and (i32 (vector_extract PulpV2:$rs2, 1)), (i32 65535)), (and (i32 (vector_extract PulpV2:$rs1, 1)), (i32 65535)))),
	(CV_SDOTUP_H__S_S_V2_V2 GPR:$rd, PulpV2:$rs1, PulpV2:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1, Constraints = "$rd = $rd_wb" in def CV_SDOTUP_SC_H__S_S_V2_S : RVInst_CV_SDOTUP_SC_H<(outs GPR:$rd_wb), (ins GPR:$rd, PulpV2:$rs1, GPR:$rs2)>;

def : Pat<
	(add (mul (add (and (i32 (vector_extract PulpV2:$rs1, 1)), (i32 65535)), (and (i32 (vector_extract PulpV2:$rs1, 0)), (i32 65535))), (and GPR:$rs2, (i32 65535))), GPR:$rd),
	(CV_SDOTUP_SC_H__S_S_V2_S GPR:$rd, PulpV2:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1, Constraints = "$rd = $rd_wb" in def CV_SDOTUP_SCI_H__S_S_V2 : RVInst_CV_SDOTUP_SCI_H<(outs GPR:$rd_wb), (ins GPR:$rd, PulpV2:$rs1, simm6:$imm)>;

def : Pat<
	(add (mul (add (and (i32 (vector_extract PulpV2:$rs1, 1)), (i32 65535)), (and (i32 (vector_extract PulpV2:$rs1, 0)), (i32 65535))), (and (i32 simm6:$imm), (i32 65535))), GPR:$rd),
	(CV_SDOTUP_SCI_H__S_S_V2 GPR:$rd, PulpV2:$rs1, simm6:$imm)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1, Constraints = "$rd = $rd_wb" in def CV_SDOTUP_B__S_S_V4_V4 : RVInst_CV_SDOTUP_B<(outs GPR:$rd_wb), (ins GPR:$rd, PulpV4:$rs1, PulpV4:$rs2)>;

def : Pat<
	(add (add (add (mul (and (vector_extract PulpV4:$rs2, (i32 0)), (i32 255)), (and (vector_extract PulpV4:$rs1, (i32 0)), (i32 255))), (mul (and (vector_extract PulpV4:$rs2, (i32 2)), (i32 255)), (and (vector_extract PulpV4:$rs1, (i32 2)), (i32 255)))), (add (mul (and (vector_extract PulpV4:$rs2, (i32 1)), (i32 255)), (and (vector_extract PulpV4:$rs1, (i32 1)), (i32 255))), (mul (and (vector_extract PulpV4:$rs2, (i32 3)), (i32 255)), (and (vector_extract PulpV4:$rs1, (i32 3)), (i32 255))))), GPR:$rd),
	(CV_SDOTUP_B__S_S_V4_V4 GPR:$rd, PulpV4:$rs1, PulpV4:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1, Constraints = "$rd = $rd_wb" in def CV_SDOTUP_SC_B__S_S_V4_S : RVInst_CV_SDOTUP_SC_B<(outs GPR:$rd_wb), (ins GPR:$rd, PulpV4:$rs1, GPR:$rs2)>;

def : Pat<
	(add (add (add (add GPR:$rd, (mul (and GPR:$rs2, (i32 255)), (and (i32 (vector_extract PulpV4:$rs1, 0)), (i32 255)))), (mul (and (i32 (vector_extract PulpV4:$rs1, 1)), (i32 255)), (and GPR:$rs2, (i32 255)))), (mul (and (i32 (vector_extract PulpV4:$rs1, 2)), (i32 255)), (and GPR:$rs2, (i32 255)))), (mul (and (i32 (vector_extract PulpV4:$rs1, 3)), (i32 255)), (and GPR:$rs2, (i32 255)))),
	(CV_SDOTUP_SC_B__S_S_V4_S GPR:$rd, PulpV4:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1, Constraints = "$rd = $rd_wb" in def CV_SDOTUP_SCI_B__S_S_V4 : RVInst_CV_SDOTUP_SCI_B<(outs GPR:$rd_wb), (ins GPR:$rd, PulpV4:$rs1, simm6:$imm)>;

def : Pat<
	(add (add (add (add GPR:$rd, (mul (and (i32 simm6:$imm), (i32 255)), (and (i32 (vector_extract PulpV4:$rs1, 0)), (i32 255)))), (mul (and (i32 simm6:$imm), (i32 255)), (and (i32 (vector_extract PulpV4:$rs1, 1)), (i32 255)))), (mul (and (i32 simm6:$imm), (i32 255)), (and (i32 (vector_extract PulpV4:$rs1, 2)), (i32 255)))), (mul (and (i32 simm6:$imm), (i32 255)), (and (i32 (vector_extract PulpV4:$rs1, 3)), (i32 255)))),
	(CV_SDOTUP_SCI_B__S_S_V4 GPR:$rd, PulpV4:$rs1, simm6:$imm)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1, Constraints = "$rd = $rd_wb" in def CV_SDOTUSP_H__S_S_V2_V2 : RVInst_CV_SDOTUSP_H<(outs GPR:$rd_wb), (ins GPR:$rd, PulpV2:$rs1, PulpV2:$rs2)>;

def : Pat<
	(add (add (mul (sra (shl (i32 (vector_extract PulpV2:$rs2, 0)), (i32 16)), (i32 16)), (and (i32 (vector_extract PulpV2:$rs1, 0)), (i32 65535))), GPR:$rd), (mul (sra (shl (i32 (vector_extract PulpV2:$rs2, 1)), (i32 16)), (i32 16)), (and (i32 (vector_extract PulpV2:$rs1, 1)), (i32 65535)))),
	(CV_SDOTUSP_H__S_S_V2_V2 GPR:$rd, PulpV2:$rs1, PulpV2:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1, Constraints = "$rd = $rd_wb" in def CV_SDOTUSP_SC_H__S_S_V2_S : RVInst_CV_SDOTUSP_SC_H<(outs GPR:$rd_wb), (ins GPR:$rd, PulpV2:$rs1, GPR:$rs2)>;

def : Pat<
	(add (mul (add (and (i32 (vector_extract PulpV2:$rs1, 1)), (i32 65535)), (and (i32 (vector_extract PulpV2:$rs1, 0)), (i32 65535))), (sra (shl GPR:$rs2, (i32 16)), (i32 16))), GPR:$rd),
	(CV_SDOTUSP_SC_H__S_S_V2_S GPR:$rd, PulpV2:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1, Constraints = "$rd = $rd_wb" in def CV_SDOTUSP_SCI_H__S_S_V2 : RVInst_CV_SDOTUSP_SCI_H<(outs GPR:$rd_wb), (ins GPR:$rd, PulpV2:$rs1, simm6:$imm)>;

def : Pat<
	(add (mul (add (and (i32 (vector_extract PulpV2:$rs1, 1)), (i32 65535)), (and (i32 (vector_extract PulpV2:$rs1, 0)), (i32 65535))), (i32 (sra (i32 (shl (i32 simm6:$imm), (i32 16))), (i32 16)))), GPR:$rd),
	(CV_SDOTUSP_SCI_H__S_S_V2 GPR:$rd, PulpV2:$rs1, simm6:$imm)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1, Constraints = "$rd = $rd_wb" in def CV_SDOTUSP_B__S_S_V4_V4 : RVInst_CV_SDOTUSP_B<(outs GPR:$rd_wb), (ins GPR:$rd, PulpV4:$rs1, PulpV4:$rs2)>;

def : Pat<
	(add (add (add (mul (i32 (sra (i32 (shl (vector_extract PulpV4:$rs2, (i32 0)), (i32 24))), (i32 24))), (and (vector_extract PulpV4:$rs1, (i32 0)), (i32 255))), (mul (i32 (sra (i32 (shl (vector_extract PulpV4:$rs2, (i32 2)), (i32 24))), (i32 24))), (and (vector_extract PulpV4:$rs1, (i32 2)), (i32 255)))), (add (mul (i32 (sra (i32 (shl (vector_extract PulpV4:$rs2, (i32 1)), (i32 24))), (i32 24))), (and (vector_extract PulpV4:$rs1, (i32 1)), (i32 255))), (mul (i32 (sra (i32 (shl (vector_extract PulpV4:$rs2, (i32 3)), (i32 24))), (i32 24))), (and (vector_extract PulpV4:$rs1, (i32 3)), (i32 255))))), GPR:$rd),
	(CV_SDOTUSP_B__S_S_V4_V4 GPR:$rd, PulpV4:$rs1, PulpV4:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1, Constraints = "$rd = $rd_wb" in def CV_SDOTUSP_SC_B__S_S_V4_S : RVInst_CV_SDOTUSP_SC_B<(outs GPR:$rd_wb), (ins GPR:$rd, PulpV4:$rs1, GPR:$rs2)>;

def : Pat<
	(add (add (add (add GPR:$rd, (i32 (sra (i32 (shl (mul (sra (shl GPR:$rs2, (i32 24)), (i32 24)), (and (i32 (vector_extract PulpV4:$rs1, 0)), (i32 255))), (i32 16))), (i32 16)))), (i32 (sra (i32 (shl (mul (and (i32 (vector_extract PulpV4:$rs1, 1)), (i32 255)), (sra (shl GPR:$rs2, (i32 24)), (i32 24))), (i32 16))), (i32 16)))), (i32 (sra (i32 (shl (mul (and (i32 (vector_extract PulpV4:$rs1, 2)), (i32 255)), (sra (shl GPR:$rs2, (i32 24)), (i32 24))), (i32 16))), (i32 16)))), (i32 (sra (i32 (shl (mul (and (i32 (vector_extract PulpV4:$rs1, 3)), (i32 255)), (sra (shl GPR:$rs2, (i32 24)), (i32 24))), (i32 16))), (i32 16)))),
	(CV_SDOTUSP_SC_B__S_S_V4_S GPR:$rd, PulpV4:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1, Constraints = "$rd = $rd_wb" in def CV_SDOTUSP_SCI_B__S_S_V4 : RVInst_CV_SDOTUSP_SCI_B<(outs GPR:$rd_wb), (ins GPR:$rd, PulpV4:$rs1, simm6:$imm)>;

def : Pat<
	(add (add (add (add GPR:$rd, (i32 (sra (i32 (shl (mul (and (i32 (vector_extract PulpV4:$rs1, 0)), (i32 255)), (i32 (sra (i32 (shl (i32 simm6:$imm), (i32 24))), (i32 24)))), (i32 16))), (i32 16)))), (i32 (sra (i32 (shl (mul (and (i32 (vector_extract PulpV4:$rs1, 1)), (i32 255)), (i32 (sra (i32 (shl (i32 simm6:$imm), (i32 24))), (i32 24)))), (i32 16))), (i32 16)))), (i32 (sra (i32 (shl (mul (and (i32 (vector_extract PulpV4:$rs1, 2)), (i32 255)), (i32 (sra (i32 (shl (i32 simm6:$imm), (i32 24))), (i32 24)))), (i32 16))), (i32 16)))), (i32 (sra (i32 (shl (mul (and (i32 (vector_extract PulpV4:$rs1, 3)), (i32 255)), (i32 (sra (i32 (shl (i32 simm6:$imm), (i32 24))), (i32 24)))), (i32 16))), (i32 16)))),
	(CV_SDOTUSP_SCI_B__S_S_V4 GPR:$rd, PulpV4:$rs1, simm6:$imm)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1, Constraints = "$rd = $rd_wb" in def CV_SDOTSP_H__S_S_V2_V2 : RVInst_CV_SDOTSP_H<(outs GPR:$rd_wb), (ins GPR:$rd, PulpV2:$rs1, PulpV2:$rs2)>;

def : Pat<
	(add (add (mul (sra (shl (i32 (vector_extract PulpV2:$rs2, 0)), (i32 16)), (i32 16)), (sra (shl (i32 (vector_extract PulpV2:$rs1, 0)), (i32 16)), (i32 16))), GPR:$rd), (mul (sra (shl (i32 (vector_extract PulpV2:$rs2, 1)), (i32 16)), (i32 16)), (sra (shl (i32 (vector_extract PulpV2:$rs1, 1)), (i32 16)), (i32 16)))),
	(CV_SDOTSP_H__S_S_V2_V2 GPR:$rd, PulpV2:$rs1, PulpV2:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1, Constraints = "$rd = $rd_wb" in def CV_SDOTSP_SC_H__S_S_V2_S : RVInst_CV_SDOTSP_SC_H<(outs GPR:$rd_wb), (ins GPR:$rd, PulpV2:$rs1, GPR:$rs2)>;

def : Pat<
	(add (mul (add (sra (shl (i32 (vector_extract PulpV2:$rs1, 1)), (i32 16)), (i32 16)), (sra (shl (i32 (vector_extract PulpV2:$rs1, 0)), (i32 16)), (i32 16))), (sra (shl GPR:$rs2, (i32 16)), (i32 16))), GPR:$rd),
	(CV_SDOTSP_SC_H__S_S_V2_S GPR:$rd, PulpV2:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1, Constraints = "$rd = $rd_wb" in def CV_SDOTSP_SCI_H__S_S_V2 : RVInst_CV_SDOTSP_SCI_H<(outs GPR:$rd_wb), (ins GPR:$rd, PulpV2:$rs1, simm6:$imm)>;

def : Pat<
	(add (mul (add (sra (shl (i32 (vector_extract PulpV2:$rs1, 1)), (i32 16)), (i32 16)), (sra (shl (i32 (vector_extract PulpV2:$rs1, 0)), (i32 16)), (i32 16))), (i32 (sra (i32 (shl (i32 simm6:$imm), (i32 16))), (i32 16)))), GPR:$rd),
	(CV_SDOTSP_SCI_H__S_S_V2 GPR:$rd, PulpV2:$rs1, simm6:$imm)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1, Constraints = "$rd = $rd_wb" in def CV_SDOTSP_B__S_S_V4_V4 : RVInst_CV_SDOTSP_B<(outs GPR:$rd_wb), (ins GPR:$rd, PulpV4:$rs1, PulpV4:$rs2)>;

def : Pat<
	(add (add (add (mul (i32 (sra (i32 (shl (vector_extract PulpV4:$rs2, (i32 0)), (i32 24))), (i32 24))), (i32 (sra (i32 (shl (vector_extract PulpV4:$rs1, (i32 0)), (i32 24))), (i32 24)))), (mul (i32 (sra (i32 (shl (vector_extract PulpV4:$rs2, (i32 2)), (i32 24))), (i32 24))), (i32 (sra (i32 (shl (vector_extract PulpV4:$rs1, (i32 2)), (i32 24))), (i32 24))))), (add (mul (i32 (sra (i32 (shl (vector_extract PulpV4:$rs2, (i32 1)), (i32 24))), (i32 24))), (i32 (sra (i32 (shl (vector_extract PulpV4:$rs1, (i32 1)), (i32 24))), (i32 24)))), (mul (i32 (sra (i32 (shl (vector_extract PulpV4:$rs2, (i32 3)), (i32 24))), (i32 24))), (i32 (sra (i32 (shl (vector_extract PulpV4:$rs1, (i32 3)), (i32 24))), (i32 24)))))), GPR:$rd),
	(CV_SDOTSP_B__S_S_V4_V4 GPR:$rd, PulpV4:$rs1, PulpV4:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1, Constraints = "$rd = $rd_wb" in def CV_SDOTSP_SC_B__S_S_V4_S : RVInst_CV_SDOTSP_SC_B<(outs GPR:$rd_wb), (ins GPR:$rd, PulpV4:$rs1, GPR:$rs2)>;

def : Pat<
	(add (add (add (add GPR:$rd, (mul (sra (shl GPR:$rs2, (i32 24)), (i32 24)), (sra (shl (i32 (vector_extract PulpV4:$rs1, 0)), (i32 24)), (i32 24)))), (mul (sra (shl (i32 (vector_extract PulpV4:$rs1, 1)), (i32 24)), (i32 24)), (sra (shl GPR:$rs2, (i32 24)), (i32 24)))), (mul (sra (shl (i32 (vector_extract PulpV4:$rs1, 2)), (i32 24)), (i32 24)), (sra (shl GPR:$rs2, (i32 24)), (i32 24)))), (mul (sra (shl (i32 (vector_extract PulpV4:$rs1, 3)), (i32 24)), (i32 24)), (sra (shl GPR:$rs2, (i32 24)), (i32 24)))),
	(CV_SDOTSP_SC_B__S_S_V4_S GPR:$rd, PulpV4:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1, Constraints = "$rd = $rd_wb" in def CV_SDOTSP_SCI_B__S_S_V4 : RVInst_CV_SDOTSP_SCI_B<(outs GPR:$rd_wb), (ins GPR:$rd, PulpV4:$rs1, simm6:$imm)>;

def : Pat<
	(add (add (add (add GPR:$rd, (mul (sra (shl (i32 (vector_extract PulpV4:$rs1, 0)), (i32 24)), (i32 24)), (i32 (sra (i32 (shl (i32 simm6:$imm), (i32 24))), (i32 24))))), (mul (sra (shl (i32 (vector_extract PulpV4:$rs1, 1)), (i32 24)), (i32 24)), (i32 (sra (i32 (shl (i32 simm6:$imm), (i32 24))), (i32 24))))), (mul (sra (shl (i32 (vector_extract PulpV4:$rs1, 2)), (i32 24)), (i32 24)), (i32 (sra (i32 (shl (i32 simm6:$imm), (i32 24))), (i32 24))))), (mul (sra (shl (i32 (vector_extract PulpV4:$rs1, 3)), (i32 24)), (i32 24)), (i32 (sra (i32 (shl (i32 simm6:$imm), (i32 24))), (i32 24))))),
	(CV_SDOTSP_SCI_B__S_S_V4 GPR:$rd, PulpV4:$rs1, simm6:$imm)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_PACK__V2_S_S : RVInst_CV_PACK<(outs PulpV2:$rd), (ins GPR:$rs1, GPR:$rs2)>;

def : Pat<
	(build_vector GPR:$rs2, GPR:$rs1),
	(CV_PACK__V2_S_S GPR:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_PACK_H__V2_V2_V2 : RVInst_CV_PACK_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, PulpV2:$rs2)>;

def : Pat<
	(build_vector (i32 (vector_extract PulpV2:$rs2, 1)), (i32 (vector_extract PulpV2:$rs1, 1))),
	(CV_PACK_H__V2_V2_V2 PulpV2:$rs1, PulpV2:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_CMPEQ_H__V2_V2_V2 : RVInst_CV_CMPEQ_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, PulpV2:$rs2)>;

def : Pat<
	(v2i16 (setcc PulpV2:$rs1, PulpV2:$rs2, SETEQ)),
	(CV_CMPEQ_H__V2_V2_V2 PulpV2:$rs1, PulpV2:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_CMPEQ_SC_H__V2_V2_S : RVInst_CV_CMPEQ_SC_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, GPR:$rs2)>;

def : Pat<
	(v2i16 (setcc PulpV2:$rs1, (v2i16 (splat_vector GPR:$rs2)), SETEQ)),
	(CV_CMPEQ_SC_H__V2_V2_S PulpV2:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_CMPEQ_SCI_H__V2_V2 : RVInst_CV_CMPEQ_SCI_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, simm6:$imm)>;

def : Pat<
	(v2i16 (setcc PulpV2:$rs1, (v2i16 (splat_vector (i32 simm6:$imm))), SETEQ)),
	(CV_CMPEQ_SCI_H__V2_V2 PulpV2:$rs1, simm6:$imm)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_CMPEQ_B__V4_V4_V4 : RVInst_CV_CMPEQ_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, PulpV4:$rs2)>;

def : Pat<
	(v4i8 (setcc PulpV4:$rs1, PulpV4:$rs2, SETEQ)),
	(CV_CMPEQ_B__V4_V4_V4 PulpV4:$rs1, PulpV4:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_CMPEQ_SC_B__V4_V4_S : RVInst_CV_CMPEQ_SC_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, GPR:$rs2)>;

def : Pat<
	(v4i8 (setcc PulpV4:$rs1, (v4i8 (splat_vector GPR:$rs2)), SETEQ)),
	(CV_CMPEQ_SC_B__V4_V4_S PulpV4:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_CMPEQ_SCI_B__V4_V4 : RVInst_CV_CMPEQ_SCI_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, simm6:$imm)>;

def : Pat<
	(v4i8 (setcc PulpV4:$rs1, (v4i8 (splat_vector (i32 simm6:$imm))), SETEQ)),
	(CV_CMPEQ_SCI_B__V4_V4 PulpV4:$rs1, simm6:$imm)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_CMPNE_H__V2_V2_V2 : RVInst_CV_CMPNE_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, PulpV2:$rs2)>;

def : Pat<
	(v2i16 (setcc PulpV2:$rs1, PulpV2:$rs2, SETNE)),
	(CV_CMPNE_H__V2_V2_V2 PulpV2:$rs1, PulpV2:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_CMPNE_SC_H__V2_V2_S : RVInst_CV_CMPNE_SC_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, GPR:$rs2)>;

def : Pat<
	(v2i16 (setcc PulpV2:$rs1, (v2i16 (splat_vector GPR:$rs2)), SETNE)),
	(CV_CMPNE_SC_H__V2_V2_S PulpV2:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_CMPNE_SCI_H__V2_V2 : RVInst_CV_CMPNE_SCI_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, simm6:$imm)>;

def : Pat<
	(v2i16 (setcc PulpV2:$rs1, (v2i16 (splat_vector (i32 simm6:$imm))), SETNE)),
	(CV_CMPNE_SCI_H__V2_V2 PulpV2:$rs1, simm6:$imm)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_CMPNE_B__V4_V4_V4 : RVInst_CV_CMPNE_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, PulpV4:$rs2)>;

def : Pat<
	(v4i8 (setcc PulpV4:$rs1, PulpV4:$rs2, SETNE)),
	(CV_CMPNE_B__V4_V4_V4 PulpV4:$rs1, PulpV4:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_CMPNE_SC_B__V4_V4_S : RVInst_CV_CMPNE_SC_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, GPR:$rs2)>;

def : Pat<
	(v4i8 (setcc PulpV4:$rs1, (v4i8 (splat_vector GPR:$rs2)), SETNE)),
	(CV_CMPNE_SC_B__V4_V4_S PulpV4:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_CMPNE_SCI_B__V4_V4 : RVInst_CV_CMPNE_SCI_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, simm6:$imm)>;

def : Pat<
	(v4i8 (setcc PulpV4:$rs1, (v4i8 (splat_vector (i32 simm6:$imm))), SETNE)),
	(CV_CMPNE_SCI_B__V4_V4 PulpV4:$rs1, simm6:$imm)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_CMPGT_H__V2_V2_V2 : RVInst_CV_CMPGT_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, PulpV2:$rs2)>;

def : Pat<
	(v2i16 (setcc PulpV2:$rs1, PulpV2:$rs2, SETGT)),
	(CV_CMPGT_H__V2_V2_V2 PulpV2:$rs1, PulpV2:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_CMPGT_SC_H__V2_V2_S : RVInst_CV_CMPGT_SC_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, GPR:$rs2)>;

def : Pat<
	(v2i16 (setcc PulpV2:$rs1, (v2i16 (splat_vector GPR:$rs2)), SETGT)),
	(CV_CMPGT_SC_H__V2_V2_S PulpV2:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_CMPGT_SCI_H__V2_V2 : RVInst_CV_CMPGT_SCI_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, simm6:$imm)>;

def : Pat<
	(v2i16 (setcc PulpV2:$rs1, (v2i16 (splat_vector (i32 simm6:$imm))), SETGT)),
	(CV_CMPGT_SCI_H__V2_V2 PulpV2:$rs1, simm6:$imm)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_CMPGT_B__V4_V4_V4 : RVInst_CV_CMPGT_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, PulpV4:$rs2)>;

def : Pat<
	(v4i8 (setcc PulpV4:$rs1, PulpV4:$rs2, SETGT)),
	(CV_CMPGT_B__V4_V4_V4 PulpV4:$rs1, PulpV4:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_CMPGT_SC_B__V4_V4_S : RVInst_CV_CMPGT_SC_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, GPR:$rs2)>;

def : Pat<
	(v4i8 (setcc PulpV4:$rs1, (v4i8 (splat_vector GPR:$rs2)), SETGT)),
	(CV_CMPGT_SC_B__V4_V4_S PulpV4:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_CMPGT_SCI_B__V4_V4 : RVInst_CV_CMPGT_SCI_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, simm6:$imm)>;

def : Pat<
	(v4i8 (setcc PulpV4:$rs1, (v4i8 (splat_vector (i32 simm6:$imm))), SETGT)),
	(CV_CMPGT_SCI_B__V4_V4 PulpV4:$rs1, simm6:$imm)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_CMPGE_H__V2_V2_V2 : RVInst_CV_CMPGE_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, PulpV2:$rs2)>;

def : Pat<
	(v2i16 (setcc PulpV2:$rs1, PulpV2:$rs2, SETGE)),
	(CV_CMPGE_H__V2_V2_V2 PulpV2:$rs1, PulpV2:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_CMPGE_SC_H__V2_V2_S : RVInst_CV_CMPGE_SC_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, GPR:$rs2)>;

def : Pat<
	(v2i16 (setcc PulpV2:$rs1, (v2i16 (splat_vector GPR:$rs2)), SETGE)),
	(CV_CMPGE_SC_H__V2_V2_S PulpV2:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_CMPGE_SCI_H__V2_V2 : RVInst_CV_CMPGE_SCI_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, simm6:$imm)>;

def : Pat<
	(v2i16 (setcc PulpV2:$rs1, (v2i16 (splat_vector (i32 simm6:$imm))), SETGE)),
	(CV_CMPGE_SCI_H__V2_V2 PulpV2:$rs1, simm6:$imm)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_CMPGE_B__V4_V4_V4 : RVInst_CV_CMPGE_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, PulpV4:$rs2)>;

def : Pat<
	(v4i8 (setcc PulpV4:$rs1, PulpV4:$rs2, SETGE)),
	(CV_CMPGE_B__V4_V4_V4 PulpV4:$rs1, PulpV4:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_CMPGE_SC_B__V4_V4_S : RVInst_CV_CMPGE_SC_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, GPR:$rs2)>;

def : Pat<
	(v4i8 (setcc PulpV4:$rs1, (v4i8 (splat_vector GPR:$rs2)), SETGE)),
	(CV_CMPGE_SC_B__V4_V4_S PulpV4:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_CMPGE_SCI_B__V4_V4 : RVInst_CV_CMPGE_SCI_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, simm6:$imm)>;

def : Pat<
	(v4i8 (setcc PulpV4:$rs1, (v4i8 (splat_vector (i32 simm6:$imm))), SETGE)),
	(CV_CMPGE_SCI_B__V4_V4 PulpV4:$rs1, simm6:$imm)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_CMPLT_H__V2_V2_V2 : RVInst_CV_CMPLT_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, PulpV2:$rs2)>;

def : Pat<
	(v2i16 (setcc PulpV2:$rs1, PulpV2:$rs2, SETLT)),
	(CV_CMPLT_H__V2_V2_V2 PulpV2:$rs1, PulpV2:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_CMPLT_SC_H__V2_V2_S : RVInst_CV_CMPLT_SC_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, GPR:$rs2)>;

def : Pat<
	(v2i16 (setcc PulpV2:$rs1, (v2i16 (splat_vector GPR:$rs2)), SETLT)),
	(CV_CMPLT_SC_H__V2_V2_S PulpV2:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_CMPLT_SCI_H__V2_V2 : RVInst_CV_CMPLT_SCI_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, simm6:$imm)>;

def : Pat<
	(v2i16 (setcc PulpV2:$rs1, (v2i16 (splat_vector (i32 simm6:$imm))), SETLT)),
	(CV_CMPLT_SCI_H__V2_V2 PulpV2:$rs1, simm6:$imm)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_CMPLT_B__V4_V4_V4 : RVInst_CV_CMPLT_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, PulpV4:$rs2)>;

def : Pat<
	(v4i8 (setcc PulpV4:$rs1, PulpV4:$rs2, SETLT)),
	(CV_CMPLT_B__V4_V4_V4 PulpV4:$rs1, PulpV4:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_CMPLT_SC_B__V4_V4_S : RVInst_CV_CMPLT_SC_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, GPR:$rs2)>;

def : Pat<
	(v4i8 (setcc PulpV4:$rs1, (v4i8 (splat_vector GPR:$rs2)), SETLT)),
	(CV_CMPLT_SC_B__V4_V4_S PulpV4:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_CMPLT_SCI_B__V4_V4 : RVInst_CV_CMPLT_SCI_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, simm6:$imm)>;

def : Pat<
	(v4i8 (setcc PulpV4:$rs1, (v4i8 (splat_vector (i32 simm6:$imm))), SETLT)),
	(CV_CMPLT_SCI_B__V4_V4 PulpV4:$rs1, simm6:$imm)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_CMPLE_H__V2_V2_V2 : RVInst_CV_CMPLE_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, PulpV2:$rs2)>;

def : Pat<
	(v2i16 (setcc PulpV2:$rs1, PulpV2:$rs2, SETLE)),
	(CV_CMPLE_H__V2_V2_V2 PulpV2:$rs1, PulpV2:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_CMPLE_SC_H__V2_V2_S : RVInst_CV_CMPLE_SC_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, GPR:$rs2)>;

def : Pat<
	(v2i16 (setcc PulpV2:$rs1, (v2i16 (splat_vector GPR:$rs2)), SETLE)),
	(CV_CMPLE_SC_H__V2_V2_S PulpV2:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_CMPLE_SCI_H__V2_V2 : RVInst_CV_CMPLE_SCI_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, simm6:$imm)>;

def : Pat<
	(v2i16 (setcc PulpV2:$rs1, (v2i16 (splat_vector (i32 simm6:$imm))), SETLE)),
	(CV_CMPLE_SCI_H__V2_V2 PulpV2:$rs1, simm6:$imm)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_CMPLE_B__V4_V4_V4 : RVInst_CV_CMPLE_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, PulpV4:$rs2)>;

def : Pat<
	(v4i8 (setcc PulpV4:$rs1, PulpV4:$rs2, SETLE)),
	(CV_CMPLE_B__V4_V4_V4 PulpV4:$rs1, PulpV4:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_CMPLE_SC_B__V4_V4_S : RVInst_CV_CMPLE_SC_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, GPR:$rs2)>;

def : Pat<
	(v4i8 (setcc PulpV4:$rs1, (v4i8 (splat_vector GPR:$rs2)), SETLE)),
	(CV_CMPLE_SC_B__V4_V4_S PulpV4:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_CMPLE_SCI_B__V4_V4 : RVInst_CV_CMPLE_SCI_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, simm6:$imm)>;

def : Pat<
	(v4i8 (setcc PulpV4:$rs1, (v4i8 (splat_vector (i32 simm6:$imm))), SETLE)),
	(CV_CMPLE_SCI_B__V4_V4 PulpV4:$rs1, simm6:$imm)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_CMPGTU_H__V2_V2_V2 : RVInst_CV_CMPGTU_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, PulpV2:$rs2)>;

def : Pat<
	(v2i16 (setcc PulpV2:$rs1, PulpV2:$rs2, SETUGT)),
	(CV_CMPGTU_H__V2_V2_V2 PulpV2:$rs1, PulpV2:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_CMPGTU_SC_H__V2_V2_S : RVInst_CV_CMPGTU_SC_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, GPR:$rs2)>;

def : Pat<
	(v2i16 (setcc PulpV2:$rs1, (v2i16 (splat_vector GPR:$rs2)), SETUGT)),
	(CV_CMPGTU_SC_H__V2_V2_S PulpV2:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_CMPGTU_SCI_H__V2_V2 : RVInst_CV_CMPGTU_SCI_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, uimm6:$imm)>;

def : Pat<
	(v2i16 (setcc PulpV2:$rs1, (v2i16 (splat_vector (i32 uimm6:$imm))), SETUGT)),
	(CV_CMPGTU_SCI_H__V2_V2 PulpV2:$rs1, uimm6:$imm)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_CMPGTU_B__V4_V4_V4 : RVInst_CV_CMPGTU_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, PulpV4:$rs2)>;

def : Pat<
	(v4i8 (setcc PulpV4:$rs1, PulpV4:$rs2, SETUGT)),
	(CV_CMPGTU_B__V4_V4_V4 PulpV4:$rs1, PulpV4:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_CMPGTU_SC_B__V4_V4_S : RVInst_CV_CMPGTU_SC_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, GPR:$rs2)>;

def : Pat<
	(v4i8 (setcc PulpV4:$rs1, (v4i8 (splat_vector GPR:$rs2)), SETUGT)),
	(CV_CMPGTU_SC_B__V4_V4_S PulpV4:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_CMPGTU_SCI_B__V4_V4 : RVInst_CV_CMPGTU_SCI_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, uimm6:$imm)>;

def : Pat<
	(v4i8 (setcc PulpV4:$rs1, (v4i8 (splat_vector (i32 uimm6:$imm))), SETUGT)),
	(CV_CMPGTU_SCI_B__V4_V4 PulpV4:$rs1, uimm6:$imm)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_CMPGEU_H__V2_V2_V2 : RVInst_CV_CMPGEU_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, PulpV2:$rs2)>;

def : Pat<
	(v2i16 (setcc PulpV2:$rs1, PulpV2:$rs2, SETUGE)),
	(CV_CMPGEU_H__V2_V2_V2 PulpV2:$rs1, PulpV2:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_CMPGEU_SC_H__V2_V2_S : RVInst_CV_CMPGEU_SC_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, GPR:$rs2)>;

def : Pat<
	(v2i16 (setcc PulpV2:$rs1, (v2i16 (splat_vector GPR:$rs2)), SETUGE)),
	(CV_CMPGEU_SC_H__V2_V2_S PulpV2:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_CMPGEU_SCI_H__V2_V2 : RVInst_CV_CMPGEU_SCI_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, uimm6:$imm)>;

def : Pat<
	(v2i16 (setcc PulpV2:$rs1, (v2i16 (splat_vector (i32 uimm6:$imm))), SETUGE)),
	(CV_CMPGEU_SCI_H__V2_V2 PulpV2:$rs1, uimm6:$imm)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_CMPGEU_B__V4_V4_V4 : RVInst_CV_CMPGEU_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, PulpV4:$rs2)>;

def : Pat<
	(v4i8 (setcc PulpV4:$rs1, PulpV4:$rs2, SETUGE)),
	(CV_CMPGEU_B__V4_V4_V4 PulpV4:$rs1, PulpV4:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_CMPGEU_SC_B__V4_V4_S : RVInst_CV_CMPGEU_SC_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, GPR:$rs2)>;

def : Pat<
	(v4i8 (setcc PulpV4:$rs1, (v4i8 (splat_vector GPR:$rs2)), SETUGE)),
	(CV_CMPGEU_SC_B__V4_V4_S PulpV4:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_CMPGEU_SCI_B__V4_V4 : RVInst_CV_CMPGEU_SCI_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, uimm6:$imm)>;

def : Pat<
	(v4i8 (setcc PulpV4:$rs1, (v4i8 (splat_vector (i32 uimm6:$imm))), SETUGE)),
	(CV_CMPGEU_SCI_B__V4_V4 PulpV4:$rs1, uimm6:$imm)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_CMPLTU_H__V2_V2_V2 : RVInst_CV_CMPLTU_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, PulpV2:$rs2)>;

def : Pat<
	(v2i16 (setcc PulpV2:$rs1, PulpV2:$rs2, SETULT)),
	(CV_CMPLTU_H__V2_V2_V2 PulpV2:$rs1, PulpV2:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_CMPLTU_SC_H__V2_V2_S : RVInst_CV_CMPLTU_SC_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, GPR:$rs2)>;

def : Pat<
	(v2i16 (setcc PulpV2:$rs1, (v2i16 (splat_vector GPR:$rs2)), SETULT)),
	(CV_CMPLTU_SC_H__V2_V2_S PulpV2:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_CMPLTU_SCI_H__V2_V2 : RVInst_CV_CMPLTU_SCI_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, uimm6:$imm)>;

def : Pat<
	(v2i16 (setcc PulpV2:$rs1, (v2i16 (splat_vector (i32 uimm6:$imm))), SETULT)),
	(CV_CMPLTU_SCI_H__V2_V2 PulpV2:$rs1, uimm6:$imm)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_CMPLTU_B__V4_V4_V4 : RVInst_CV_CMPLTU_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, PulpV4:$rs2)>;

def : Pat<
	(v4i8 (setcc PulpV4:$rs1, PulpV4:$rs2, SETULT)),
	(CV_CMPLTU_B__V4_V4_V4 PulpV4:$rs1, PulpV4:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_CMPLTU_SC_B__V4_V4_S : RVInst_CV_CMPLTU_SC_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, GPR:$rs2)>;

def : Pat<
	(v4i8 (setcc PulpV4:$rs1, (v4i8 (splat_vector GPR:$rs2)), SETULT)),
	(CV_CMPLTU_SC_B__V4_V4_S PulpV4:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_CMPLTU_SCI_B__V4_V4 : RVInst_CV_CMPLTU_SCI_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, uimm6:$imm)>;

def : Pat<
	(v4i8 (setcc PulpV4:$rs1, (v4i8 (splat_vector (i32 uimm6:$imm))), SETULT)),
	(CV_CMPLTU_SCI_B__V4_V4 PulpV4:$rs1, uimm6:$imm)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_CMPLEU_H__V2_V2_V2 : RVInst_CV_CMPLEU_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, PulpV2:$rs2)>;

def : Pat<
	(v2i16 (setcc PulpV2:$rs1, PulpV2:$rs2, SETULE)),
	(CV_CMPLEU_H__V2_V2_V2 PulpV2:$rs1, PulpV2:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_CMPLEU_SC_H__V2_V2_S : RVInst_CV_CMPLEU_SC_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, GPR:$rs2)>;

def : Pat<
	(v2i16 (setcc PulpV2:$rs1, (v2i16 (splat_vector GPR:$rs2)), SETULE)),
	(CV_CMPLEU_SC_H__V2_V2_S PulpV2:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_CMPLEU_SCI_H__V2_V2 : RVInst_CV_CMPLEU_SCI_H<(outs PulpV2:$rd), (ins PulpV2:$rs1, uimm6:$imm)>;

def : Pat<
	(v2i16 (setcc PulpV2:$rs1, (v2i16 (splat_vector (i32 uimm6:$imm))), SETULE)),
	(CV_CMPLEU_SCI_H__V2_V2 PulpV2:$rs1, uimm6:$imm)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_CMPLEU_B__V4_V4_V4 : RVInst_CV_CMPLEU_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, PulpV4:$rs2)>;

def : Pat<
	(v4i8 (setcc PulpV4:$rs1, PulpV4:$rs2, SETULE)),
	(CV_CMPLEU_B__V4_V4_V4 PulpV4:$rs1, PulpV4:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_CMPLEU_SC_B__V4_V4_S : RVInst_CV_CMPLEU_SC_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, GPR:$rs2)>;

def : Pat<
	(v4i8 (setcc PulpV4:$rs1, (v4i8 (splat_vector GPR:$rs2)), SETULE)),
	(CV_CMPLEU_SC_B__V4_V4_S PulpV4:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_CMPLEU_SCI_B__V4_V4 : RVInst_CV_CMPLEU_SCI_B<(outs PulpV4:$rd), (ins PulpV4:$rs1, uimm6:$imm)>;

def : Pat<
	(v4i8 (setcc PulpV4:$rs1, (v4i8 (splat_vector (i32 uimm6:$imm))), SETULE)),
	(CV_CMPLEU_SCI_B__V4_V4 PulpV4:$rs1, uimm6:$imm)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_ADD_DIV2__V2_V2_V2 : RVInst_CV_ADD_DIV2<(outs PulpV2:$rd), (ins PulpV2:$rs1, PulpV2:$rs2)>;

def : Pat<
	(build_vector (i32 (srl (add (i32 (sra (i32 (shl (vector_extract PulpV2:$rs2, (i32 0)), (i32 16))), (i32 16))), (i32 (sra (i32 (shl (vector_extract PulpV2:$rs1, (i32 0)), (i32 16))), (i32 16)))), (i32 1))), (i32 (srl (add (i32 (sra (i32 (shl (vector_extract PulpV2:$rs2, (i32 1)), (i32 16))), (i32 16))), (i32 (sra (i32 (shl (vector_extract PulpV2:$rs1, (i32 1)), (i32 16))), (i32 16)))), (i32 1)))),
	(CV_ADD_DIV2__V2_V2_V2 PulpV2:$rs1, PulpV2:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_ADD_DIV4__V2_V2_V2 : RVInst_CV_ADD_DIV4<(outs PulpV2:$rd), (ins PulpV2:$rs1, PulpV2:$rs2)>;

def : Pat<
	(build_vector (i32 (srl (add (i32 (sra (i32 (shl (vector_extract PulpV2:$rs2, (i32 0)), (i32 16))), (i32 16))), (i32 (sra (i32 (shl (vector_extract PulpV2:$rs1, (i32 0)), (i32 16))), (i32 16)))), (i32 2))), (i32 (srl (add (i32 (sra (i32 (shl (vector_extract PulpV2:$rs2, (i32 1)), (i32 16))), (i32 16))), (i32 (sra (i32 (shl (vector_extract PulpV2:$rs1, (i32 1)), (i32 16))), (i32 16)))), (i32 2)))),
	(CV_ADD_DIV4__V2_V2_V2 PulpV2:$rs1, PulpV2:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_ADD_DIV8__V2_V2_V2 : RVInst_CV_ADD_DIV8<(outs PulpV2:$rd), (ins PulpV2:$rs1, PulpV2:$rs2)>;

def : Pat<
	(build_vector (i32 (srl (add (i32 (sra (i32 (shl (vector_extract PulpV2:$rs2, (i32 0)), (i32 16))), (i32 16))), (i32 (sra (i32 (shl (vector_extract PulpV2:$rs1, (i32 0)), (i32 16))), (i32 16)))), (i32 3))), (i32 (srl (add (i32 (sra (i32 (shl (vector_extract PulpV2:$rs2, (i32 1)), (i32 16))), (i32 16))), (i32 (sra (i32 (shl (vector_extract PulpV2:$rs1, (i32 1)), (i32 16))), (i32 16)))), (i32 3)))),
	(CV_ADD_DIV8__V2_V2_V2 PulpV2:$rs1, PulpV2:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_SUB_DIV2__V2_V2_V2 : RVInst_CV_SUB_DIV2<(outs PulpV2:$rd), (ins PulpV2:$rs1, PulpV2:$rs2)>;

def : Pat<
	(build_vector (i32 (srl (sub (i32 (sra (i32 (shl (vector_extract PulpV2:$rs1, (i32 0)), (i32 16))), (i32 16))), (i32 (sra (i32 (shl (vector_extract PulpV2:$rs2, (i32 0)), (i32 16))), (i32 16)))), (i32 1))), (i32 (srl (sub (i32 (sra (i32 (shl (vector_extract PulpV2:$rs1, (i32 1)), (i32 16))), (i32 16))), (i32 (sra (i32 (shl (vector_extract PulpV2:$rs2, (i32 1)), (i32 16))), (i32 16)))), (i32 1)))),
	(CV_SUB_DIV2__V2_V2_V2 PulpV2:$rs1, PulpV2:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_SUB_DIV4__V2_V2_V2 : RVInst_CV_SUB_DIV4<(outs PulpV2:$rd), (ins PulpV2:$rs1, PulpV2:$rs2)>;

def : Pat<
	(build_vector (i32 (srl (sub (i32 (sra (i32 (shl (vector_extract PulpV2:$rs1, (i32 0)), (i32 16))), (i32 16))), (i32 (sra (i32 (shl (vector_extract PulpV2:$rs2, (i32 0)), (i32 16))), (i32 16)))), (i32 2))), (i32 (srl (sub (i32 (sra (i32 (shl (vector_extract PulpV2:$rs1, (i32 1)), (i32 16))), (i32 16))), (i32 (sra (i32 (shl (vector_extract PulpV2:$rs2, (i32 1)), (i32 16))), (i32 16)))), (i32 2)))),
	(CV_SUB_DIV4__V2_V2_V2 PulpV2:$rs1, PulpV2:$rs2)>;

let Predicates = [HasExtXcvsimd], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_SUB_DIV8__V2_V2_V2 : RVInst_CV_SUB_DIV8<(outs PulpV2:$rd), (ins PulpV2:$rs1, PulpV2:$rs2)>;

def : Pat<
	(build_vector (i32 (srl (sub (i32 (sra (i32 (shl (vector_extract PulpV2:$rs1, (i32 0)), (i32 16))), (i32 16))), (i32 (sra (i32 (shl (vector_extract PulpV2:$rs2, (i32 0)), (i32 16))), (i32 16)))), (i32 3))), (i32 (srl (sub (i32 (sra (i32 (shl (vector_extract PulpV2:$rs1, (i32 1)), (i32 16))), (i32 16))), (i32 (sra (i32 (shl (vector_extract PulpV2:$rs2, (i32 1)), (i32 16))), (i32 16)))), (i32 3)))),
	(CV_SUB_DIV8__V2_V2_V2 PulpV2:$rs1, PulpV2:$rs2)>;

