let Predicates = [HasExtXcvalu], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_ABS__S_S : RVInst_CV_ABS<(outs GPR:$rd), (ins GPR:$rs1)>;

def : Pat<
	(i32 (abs GPR:$rs1)),
	(CV_ABS__S_S GPR:$rs1)>;

let Predicates = [HasExtXcvalu], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_SLET__S_S_S : RVInst_CV_SLET<(outs GPR:$rd), (ins GPR:$rs1, GPR:$rs2)>;

def : Pat<
	(i32 (setcc GPR:$rs1, GPR:$rs2, SETLE)),
	(CV_SLET__S_S_S GPR:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvalu], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_SLETU__S_S_S : RVInst_CV_SLETU<(outs GPR:$rd), (ins GPR:$rs1, GPR:$rs2)>;

def : Pat<
	(i32 (setcc GPR:$rs1, GPR:$rs2, SETULE)),
	(CV_SLETU__S_S_S GPR:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvalu], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_MIN__S_S_S : RVInst_CV_MIN<(outs GPR:$rd), (ins GPR:$rs1, GPR:$rs2)>;

def : Pat<
	(smin GPR:$rs1, GPR:$rs2),
	(CV_MIN__S_S_S GPR:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvalu], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_MINU__S_S_S : RVInst_CV_MINU<(outs GPR:$rd), (ins GPR:$rs1, GPR:$rs2)>;

def : Pat<
	(umin GPR:$rs1, GPR:$rs2),
	(CV_MINU__S_S_S GPR:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvalu], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_MAX__S_S_S : RVInst_CV_MAX<(outs GPR:$rd), (ins GPR:$rs1, GPR:$rs2)>;

def : Pat<
	(smax GPR:$rs1, GPR:$rs2),
	(CV_MAX__S_S_S GPR:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvalu], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_MAXU__S_S_S : RVInst_CV_MAXU<(outs GPR:$rd), (ins GPR:$rs1, GPR:$rs2)>;

def : Pat<
	(umax GPR:$rs1, GPR:$rs2),
	(CV_MAXU__S_S_S GPR:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvalu], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_EXTHS__S_S : RVInst_CV_EXTHS<(outs GPR:$rd), (ins GPR:$rs1)>;

def : Pat<
	(sra (shl GPR:$rs1, (i32 16)), (i32 16)),
	(CV_EXTHS__S_S GPR:$rs1)>;

let Predicates = [HasExtXcvalu], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_EXTHZ__S_S : RVInst_CV_EXTHZ<(outs GPR:$rd), (ins GPR:$rs1)>;

def : Pat<
	(and GPR:$rs1, (i32 65535)),
	(CV_EXTHZ__S_S GPR:$rs1)>;

let Predicates = [HasExtXcvalu], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_EXTBS__S_S : RVInst_CV_EXTBS<(outs GPR:$rd), (ins GPR:$rs1)>;

def : Pat<
	(sra (shl GPR:$rs1, (i32 24)), (i32 24)),
	(CV_EXTBS__S_S GPR:$rs1)>;

let Predicates = [HasExtXcvalu], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_EXTBZ__S_S : RVInst_CV_EXTBZ<(outs GPR:$rd), (ins GPR:$rs1)>;

def : Pat<
	(and GPR:$rs1, (i32 255)),
	(CV_EXTBZ__S_S GPR:$rs1)>;

let Predicates = [HasExtXcvalu], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_CLIP__S_S : RVInst_CV_CLIP<(outs GPR:$rd), (ins GPR:$rs1, uimm5:$imm)>;

def : Pat<
	(i32 (riscv_selectcc (or (sub (i32 0), (i32 (setcc (i32 uimm5:$imm), (i32 0), SETEQ))), (sub (i32 0), (i32 (shl (i32 1), (add (i32 uimm5:$imm), (i32 4294967295)))))), GPR:$rs1, SETLT, (smin GPR:$rs1, (and (add (i32 (setcc (i32 uimm5:$imm), (i32 0), SETEQ)), (i32 4294967295)), (add (i32 (shl (i32 1), (add (i32 uimm5:$imm), (i32 4294967295)))), (i32 4294967295)))), (or (sub (i32 0), (i32 (setcc (i32 uimm5:$imm), (i32 0), SETEQ))), (sub (i32 0), (i32 (shl (i32 1), (add (i32 uimm5:$imm), (i32 4294967295)))))))),
	(CV_CLIP__S_S GPR:$rs1, uimm5:$imm)>;

let Predicates = [HasExtXcvalu], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_CLIPU__S_S : RVInst_CV_CLIPU<(outs GPR:$rd), (ins GPR:$rs1, uimm5:$imm)>;

def : Pat<
	(and (sub (i32 0), (i32 (setcc (i32 0), GPR:$rs1, SETLT))), (smin GPR:$rs1, (and (add (i32 (setcc (i32 uimm5:$imm), (i32 0), SETEQ)), (i32 4294967295)), (xor (i32 (shl (i32 4294967295), (add (i32 uimm5:$imm), (i32 4294967295)))), (i32 4294967295))))),
	(CV_CLIPU__S_S GPR:$rs1, uimm5:$imm)>;

let Predicates = [HasExtXcvalu], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_CLIPR__S_S_S : RVInst_CV_CLIPR<(outs GPR:$rd), (ins GPR:$rs1, GPR:$rs2)>;

def : Pat<
	(i32 (riscv_selectcc (xor GPR:$rs2, (i32 4294967295)), GPR:$rs1, SETLT, (smin GPR:$rs1, GPR:$rs2), (xor GPR:$rs2, (i32 4294967295)))),
	(CV_CLIPR__S_S_S GPR:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvalu], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_CLIPUR__S_S_S : RVInst_CV_CLIPUR<(outs GPR:$rd), (ins GPR:$rs1, GPR:$rs2)>;

def : Pat<
	(and (sub (i32 0), (i32 (setcc (i32 0), GPR:$rs1, SETLT))), (smin GPR:$rs1, GPR:$rs2)),
	(CV_CLIPUR__S_S_S GPR:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvalu], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_ADDN__S_S_S : RVInst_CV_ADDN<(outs GPR:$rd), (ins GPR:$rs1, GPR:$rs2, uimm5:$imm)>;

def : Pat<
	(i32 (sra (add GPR:$rs2, GPR:$rs1), (i32 uimm5:$imm))),
	(CV_ADDN__S_S_S GPR:$rs1, GPR:$rs2, uimm5:$imm)>;

let Predicates = [HasExtXcvalu], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_ADDUN__S_S_S : RVInst_CV_ADDUN<(outs GPR:$rd), (ins GPR:$rs1, GPR:$rs2, uimm5:$imm)>;

def : Pat<
	(i32 (srl (add GPR:$rs2, GPR:$rs1), (i32 uimm5:$imm))),
	(CV_ADDUN__S_S_S GPR:$rs1, GPR:$rs2, uimm5:$imm)>;

let Predicates = [HasExtXcvalu], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_ADDRN__S_S_S : RVInst_CV_ADDRN<(outs GPR:$rd), (ins GPR:$rs1, GPR:$rs2, uimm5:$imm)>;

def : Pat<
	(i32 (sra (add (add GPR:$rs1, (i32 (shl (i32 1), (add (i32 uimm5:$imm), (i32 4294967295))))), GPR:$rs2), (i32 uimm5:$imm))),
	(CV_ADDRN__S_S_S GPR:$rs1, GPR:$rs2, uimm5:$imm)>;

let Predicates = [HasExtXcvalu], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_ADDURN__S_S_S : RVInst_CV_ADDURN<(outs GPR:$rd), (ins GPR:$rs1, GPR:$rs2, uimm5:$imm)>;

def : Pat<
	(i32 (srl (add (add GPR:$rs1, (i32 (shl (i32 1), (add (i32 uimm5:$imm), (i32 4294967295))))), GPR:$rs2), (i32 uimm5:$imm))),
	(CV_ADDURN__S_S_S GPR:$rs1, GPR:$rs2, uimm5:$imm)>;

let Predicates = [HasExtXcvalu], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_SUBN__S_S_S : RVInst_CV_SUBN<(outs GPR:$rd), (ins GPR:$rs1, GPR:$rs2, uimm5:$imm)>;

def : Pat<
	(i32 (sra (sub GPR:$rs1, GPR:$rs2), (i32 uimm5:$imm))),
	(CV_SUBN__S_S_S GPR:$rs1, GPR:$rs2, uimm5:$imm)>;

let Predicates = [HasExtXcvalu], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_SUBUN__S_S_S : RVInst_CV_SUBUN<(outs GPR:$rd), (ins GPR:$rs1, GPR:$rs2, uimm5:$imm)>;

def : Pat<
	(i32 (srl (sub GPR:$rs1, GPR:$rs2), (i32 uimm5:$imm))),
	(CV_SUBUN__S_S_S GPR:$rs1, GPR:$rs2, uimm5:$imm)>;

let Predicates = [HasExtXcvalu], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_SUBRN__S_S_S : RVInst_CV_SUBRN<(outs GPR:$rd), (ins GPR:$rs1, GPR:$rs2, uimm5:$imm)>;

def : Pat<
	(i32 (sra (sub (add GPR:$rs1, (i32 (shl (i32 1), (add (i32 uimm5:$imm), (i32 4294967295))))), GPR:$rs2), (i32 uimm5:$imm))),
	(CV_SUBRN__S_S_S GPR:$rs1, GPR:$rs2, uimm5:$imm)>;

let Predicates = [HasExtXcvalu], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_SUBURN__S_S_S : RVInst_CV_SUBURN<(outs GPR:$rd), (ins GPR:$rs1, GPR:$rs2, uimm5:$imm)>;

def : Pat<
	(i32 (srl (sub (add GPR:$rs1, (i32 (shl (i32 1), (add (i32 uimm5:$imm), (i32 4294967295))))), GPR:$rs2), (i32 uimm5:$imm))),
	(CV_SUBURN__S_S_S GPR:$rs1, GPR:$rs2, uimm5:$imm)>;

let Predicates = [HasExtXcvalu], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1, Constraints = "$rd = $rd_wb" in def CV_ADDNR__S_S_S_S : RVInst_CV_ADDNR<(outs GPR:$rd_wb), (ins GPR:$rd, GPR:$rs1, GPR:$rs2)>;

def : Pat<
	(i32 (sra (add GPR:$rs1, GPR:$rd), (and GPR:$rs2, (i32 31)))),
	(CV_ADDNR__S_S_S_S GPR:$rd, GPR:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvalu], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1, Constraints = "$rd = $rd_wb" in def CV_ADDUNR__S_S_S_S : RVInst_CV_ADDUNR<(outs GPR:$rd_wb), (ins GPR:$rd, GPR:$rs1, GPR:$rs2)>;

def : Pat<
	(i32 (srl (add GPR:$rs1, GPR:$rd), (and GPR:$rs2, (i32 31)))),
	(CV_ADDUNR__S_S_S_S GPR:$rd, GPR:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvalu], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1, Constraints = "$rd = $rd_wb" in def CV_ADDRNR__S_S_S_S : RVInst_CV_ADDRNR<(outs GPR:$rd_wb), (ins GPR:$rd, GPR:$rs1, GPR:$rs2)>;

def : Pat<
	(i32 (sra (add (add GPR:$rs1, GPR:$rd), (i32 (shl (i32 1), (and (add (and GPR:$rs2, (i32 31)), (i32 4294967295)), (i32 255))))), (and GPR:$rs2, (i32 31)))),
	(CV_ADDRNR__S_S_S_S GPR:$rd, GPR:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvalu], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1, Constraints = "$rd = $rd_wb" in def CV_ADDURNR__S_S_S_S : RVInst_CV_ADDURNR<(outs GPR:$rd_wb), (ins GPR:$rd, GPR:$rs1, GPR:$rs2)>;

def : Pat<
	(i32 (srl (add (add GPR:$rs1, GPR:$rd), (i32 (shl (i32 1), (and (add (and GPR:$rs2, (i32 31)), (i32 4294967295)), (i32 255))))), (and GPR:$rs2, (i32 31)))),
	(CV_ADDURNR__S_S_S_S GPR:$rd, GPR:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvalu], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1, Constraints = "$rd = $rd_wb" in def CV_SUBNR__S_S_S_S : RVInst_CV_SUBNR<(outs GPR:$rd_wb), (ins GPR:$rd, GPR:$rs1, GPR:$rs2)>;

def : Pat<
	(i32 (sra (sub GPR:$rd, GPR:$rs1), (and GPR:$rs2, (i32 31)))),
	(CV_SUBNR__S_S_S_S GPR:$rd, GPR:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvalu], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1, Constraints = "$rd = $rd_wb" in def CV_SUBUNR__S_S_S_S : RVInst_CV_SUBUNR<(outs GPR:$rd_wb), (ins GPR:$rd, GPR:$rs1, GPR:$rs2)>;

def : Pat<
	(i32 (srl (sub GPR:$rd, GPR:$rs1), (and GPR:$rs2, (i32 31)))),
	(CV_SUBUNR__S_S_S_S GPR:$rd, GPR:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvalu], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1, Constraints = "$rd = $rd_wb" in def CV_SUBRNR__S_S_S_S : RVInst_CV_SUBRNR<(outs GPR:$rd_wb), (ins GPR:$rd, GPR:$rs1, GPR:$rs2)>;

def : Pat<
	(i32 (sra (add (sub GPR:$rd, GPR:$rs1), (i32 (shl (i32 1), (and (add (and GPR:$rs2, (i32 31)), (i32 4294967295)), (i32 255))))), (and GPR:$rs2, (i32 31)))),
	(CV_SUBRNR__S_S_S_S GPR:$rd, GPR:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvalu], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1, Constraints = "$rd = $rd_wb" in def CV_SUBURNR__S_S_S_S : RVInst_CV_SUBURNR<(outs GPR:$rd_wb), (ins GPR:$rd, GPR:$rs1, GPR:$rs2)>;

def : Pat<
	(i32 (srl (add (sub GPR:$rd, GPR:$rs1), (i32 (shl (i32 1), (and (add (and GPR:$rs2, (i32 31)), (i32 4294967295)), (i32 255))))), (and GPR:$rs2, (i32 31)))),
	(CV_SUBURNR__S_S_S_S GPR:$rd, GPR:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvalu], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_EXTRACT__S_S : RVInst_CV_EXTRACT<(outs GPR:$rd), (ins GPR:$rs1, uimm5:$imm, uimm5:$imm2)>;

def : Pat<
	(i32 (sra (i32 (shl (and (i32 (srl GPR:$rs1, (i32 uimm5:$imm2))), (add (i32 (shl (i32 2), (i32 uimm5:$imm))), (i32 4294967295))), (sub (i32 31), (i32 uimm5:$imm)))), (sub (i32 31), (i32 uimm5:$imm)))),
	(CV_EXTRACT__S_S GPR:$rs1, uimm5:$imm, uimm5:$imm2)>;

let Predicates = [HasExtXcvalu], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_EXTRACTU__S_S : RVInst_CV_EXTRACTU<(outs GPR:$rd), (ins GPR:$rs1, uimm5:$imm, uimm5:$imm2)>;

def : Pat<
	(and (i32 (srl GPR:$rs1, (i32 uimm5:$imm2))), (add (i32 (shl (i32 2), (i32 uimm5:$imm))), (i32 4294967295))),
	(CV_EXTRACTU__S_S GPR:$rs1, uimm5:$imm, uimm5:$imm2)>;

let Predicates = [HasExtXcvalu], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1, Constraints = "$rd = $rd_wb" in def CV_INSERT__S_S_S : RVInst_CV_INSERT<(outs GPR:$rd_wb), (ins GPR:$rd, GPR:$rs1, uimm5:$imm, uimm5:$imm2)>;

def : Pat<
	(or (i32 (shl (and GPR:$rs1, (add (i32 (shl (i32 2), (i32 uimm5:$imm))), (i32 4294967295))), (i32 uimm5:$imm2))), (and GPR:$rd, (xor (i32 (shl (add (i32 (shl (i32 2), (i32 uimm5:$imm))), (i32 4294967295)), (i32 uimm5:$imm2))), (i32 4294967295)))),
	(CV_INSERT__S_S_S GPR:$rd, GPR:$rs1, uimm5:$imm, uimm5:$imm2)>;

let Predicates = [HasExtXcvalu], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_EXTRACTR__S_S_S : RVInst_CV_EXTRACTR<(outs GPR:$rd), (ins GPR:$rs1, GPR:$rs2)>;

def : Pat<
	(i32 (sra (i32 (shl (and (add (i32 (shl (i32 2), (and (i32 (srl GPR:$rs2, (i32 5))), (i32 31)))), (i32 4294967295)), (i32 (srl GPR:$rs1, (and GPR:$rs2, (i32 31))))), (xor (and (i32 (srl GPR:$rs2, (i32 5))), (i32 31)), (i32 31)))), (xor (and (i32 (srl GPR:$rs2, (i32 5))), (i32 31)), (i32 31)))),
	(CV_EXTRACTR__S_S_S GPR:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvalu], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_EXTRACTUR__S_S_S : RVInst_CV_EXTRACTUR<(outs GPR:$rd), (ins GPR:$rs1, GPR:$rs2)>;

def : Pat<
	(and (add (i32 (shl (i32 2), (and (i32 (srl GPR:$rs2, (i32 5))), (i32 31)))), (i32 4294967295)), (i32 (srl GPR:$rs1, (and GPR:$rs2, (i32 31))))),
	(CV_EXTRACTUR__S_S_S GPR:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvalu], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1, Constraints = "$rd = $rd_wb" in def CV_INSERTR__S_S_S_S : RVInst_CV_INSERTR<(outs GPR:$rd_wb), (ins GPR:$rd, GPR:$rs1, GPR:$rs2)>;

def : Pat<
	(or (and GPR:$rd, (xor (i32 (shl (add (i32 (shl (i32 2), (and (i32 (srl GPR:$rs2, (i32 5))), (i32 31)))), (i32 4294967295)), (and GPR:$rs2, (i32 31)))), (i32 4294967295))), (i32 (shl (and (add (i32 (shl (i32 2), (and (i32 (srl GPR:$rs2, (i32 5))), (i32 31)))), (i32 4294967295)), GPR:$rs1), (and GPR:$rs2, (i32 31))))),
	(CV_INSERTR__S_S_S_S GPR:$rd, GPR:$rs1, GPR:$rs2)>;

let Predicates = [HasExtXcvalu], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in def CV_ROR__S_S_S : RVInst_CV_ROR<(outs GPR:$rd), (ins GPR:$rs1, GPR:$rs2)>;

def : Pat<
	(or (i32 (shl GPR:$rs1, (sub (i32 32), GPR:$rs2))), (i32 (srl GPR:$rs1, GPR:$rs2))),
	(CV_ROR__S_S_S GPR:$rs1, GPR:$rs2)>;

