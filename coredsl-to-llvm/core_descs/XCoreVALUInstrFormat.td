class RVInst_CV_ABS<dag outs, dag ins> : RVInst<outs, ins, "cv.abs", "$rd, $rs1", [], InstFormatOther> {
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-25} = 0x28;
	let Inst{24-20} = 0x0;
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x3;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x2b;
}
class RVInst_CV_SLET<dag outs, dag ins> : RVInst<outs, ins, "cv.slet", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-25} = 0x29;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x3;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x2b;
}
class RVInst_CV_SLETU<dag outs, dag ins> : RVInst<outs, ins, "cv.sletu", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-25} = 0x2a;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x3;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x2b;
}
class RVInst_CV_MIN<dag outs, dag ins> : RVInst<outs, ins, "cv.min", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-25} = 0x2b;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x3;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x2b;
}
class RVInst_CV_MINU<dag outs, dag ins> : RVInst<outs, ins, "cv.minu", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-25} = 0x2c;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x3;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x2b;
}
class RVInst_CV_MAX<dag outs, dag ins> : RVInst<outs, ins, "cv.max", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-25} = 0x2d;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x3;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x2b;
}
class RVInst_CV_MAXU<dag outs, dag ins> : RVInst<outs, ins, "cv.maxu", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-25} = 0x2e;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x3;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x2b;
}
class RVInst_CV_EXTHS<dag outs, dag ins> : RVInst<outs, ins, "cv.exths", "$rd, $rs1", [], InstFormatOther> {
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-25} = 0x30;
	let Inst{24-20} = 0x0;
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x3;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x2b;
}
class RVInst_CV_EXTHZ<dag outs, dag ins> : RVInst<outs, ins, "cv.exthz", "$rd, $rs1", [], InstFormatOther> {
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-25} = 0x31;
	let Inst{24-20} = 0x0;
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x3;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x2b;
}
class RVInst_CV_EXTBS<dag outs, dag ins> : RVInst<outs, ins, "cv.extbs", "$rd, $rs1", [], InstFormatOther> {
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-25} = 0x32;
	let Inst{24-20} = 0x0;
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x3;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x2b;
}
class RVInst_CV_EXTBZ<dag outs, dag ins> : RVInst<outs, ins, "cv.extbz", "$rd, $rs1", [], InstFormatOther> {
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-25} = 0x33;
	let Inst{24-20} = 0x0;
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x3;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x2b;
}
class RVInst_CV_CLIP<dag outs, dag ins> : RVInst<outs, ins, "cv.clip", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<5> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-25} = 0x38;
	let Inst{24-20} = imm{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x3;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x2b;
}
class RVInst_CV_CLIPU<dag outs, dag ins> : RVInst<outs, ins, "cv.clipu", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<5> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-25} = 0x39;
	let Inst{24-20} = imm{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x3;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x2b;
}
class RVInst_CV_CLIPR<dag outs, dag ins> : RVInst<outs, ins, "cv.clipr", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-25} = 0x3a;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x3;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x2b;
}
class RVInst_CV_CLIPUR<dag outs, dag ins> : RVInst<outs, ins, "cv.clipur", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-25} = 0x3b;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x3;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x2b;
}
class RVInst_CV_ADDN<dag outs, dag ins> : RVInst<outs, ins, "cv.addn", "$rd, $rs1, $rs2, $imm", [], InstFormatOther> {
	bits<5> imm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-30} = 0x0;
	let Inst{29-25} = imm{4-0};
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x2;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x5b;
}
class RVInst_CV_ADDUN<dag outs, dag ins> : RVInst<outs, ins, "cv.addun", "$rd, $rs1, $rs2, $imm", [], InstFormatOther> {
	bits<5> imm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-30} = 0x1;
	let Inst{29-25} = imm{4-0};
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x2;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x5b;
}
class RVInst_CV_ADDRN<dag outs, dag ins> : RVInst<outs, ins, "cv.addrn", "$rd, $rs1, $rs2, $imm", [], InstFormatOther> {
	bits<5> imm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-30} = 0x2;
	let Inst{29-25} = imm{4-0};
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x2;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x5b;
}
class RVInst_CV_ADDURN<dag outs, dag ins> : RVInst<outs, ins, "cv.addurn", "$rd, $rs1, $rs2, $imm", [], InstFormatOther> {
	bits<5> imm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-30} = 0x3;
	let Inst{29-25} = imm{4-0};
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x2;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x5b;
}
class RVInst_CV_SUBN<dag outs, dag ins> : RVInst<outs, ins, "cv.subn", "$rd, $rs1, $rs2, $imm", [], InstFormatOther> {
	bits<5> imm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-30} = 0x0;
	let Inst{29-25} = imm{4-0};
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x3;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x5b;
}
class RVInst_CV_SUBUN<dag outs, dag ins> : RVInst<outs, ins, "cv.subun", "$rd, $rs1, $rs2, $imm", [], InstFormatOther> {
	bits<5> imm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-30} = 0x1;
	let Inst{29-25} = imm{4-0};
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x3;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x5b;
}
class RVInst_CV_SUBRN<dag outs, dag ins> : RVInst<outs, ins, "cv.subrn", "$rd, $rs1, $rs2, $imm", [], InstFormatOther> {
	bits<5> imm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-30} = 0x2;
	let Inst{29-25} = imm{4-0};
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x3;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x5b;
}
class RVInst_CV_SUBURN<dag outs, dag ins> : RVInst<outs, ins, "cv.suburn", "$rd, $rs1, $rs2, $imm", [], InstFormatOther> {
	bits<5> imm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-30} = 0x3;
	let Inst{29-25} = imm{4-0};
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x3;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x5b;
}
class RVInst_CV_ADDNR<dag outs, dag ins> : RVInst<outs, ins, "cv.addnr", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-25} = 0x40;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x3;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x2b;
}
class RVInst_CV_ADDUNR<dag outs, dag ins> : RVInst<outs, ins, "cv.addunr", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-25} = 0x41;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x3;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x2b;
}
class RVInst_CV_ADDRNR<dag outs, dag ins> : RVInst<outs, ins, "cv.addrnr", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-25} = 0x42;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x3;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x2b;
}
class RVInst_CV_ADDURNR<dag outs, dag ins> : RVInst<outs, ins, "cv.addurnr", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-25} = 0x43;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x3;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x2b;
}
class RVInst_CV_SUBNR<dag outs, dag ins> : RVInst<outs, ins, "cv.subnr", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-25} = 0x44;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x3;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x2b;
}
class RVInst_CV_SUBUNR<dag outs, dag ins> : RVInst<outs, ins, "cv.subunr", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-25} = 0x45;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x3;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x2b;
}
class RVInst_CV_SUBRNR<dag outs, dag ins> : RVInst<outs, ins, "cv.subrnr", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-25} = 0x46;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x3;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x2b;
}
class RVInst_CV_SUBURNR<dag outs, dag ins> : RVInst<outs, ins, "cv.suburnr", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-25} = 0x47;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x3;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x2b;
}
class RVInst_CV_EXTRACT<dag outs, dag ins> : RVInst<outs, ins, "cv.extract", "$rd, $rs1, $imm, $imm2", [], InstFormatOther> {
	bits<5> imm;
	bits<5> imm2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-30} = 0x0;
	let Inst{29-25} = imm{4-0};
	let Inst{24-20} = imm2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x0;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x5b;
}
class RVInst_CV_EXTRACTU<dag outs, dag ins> : RVInst<outs, ins, "cv.extractu", "$rd, $rs1, $imm, $imm2", [], InstFormatOther> {
	bits<5> imm;
	bits<5> imm2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-30} = 0x1;
	let Inst{29-25} = imm{4-0};
	let Inst{24-20} = imm2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x0;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x5b;
}
class RVInst_CV_INSERT<dag outs, dag ins> : RVInst<outs, ins, "cv.insert", "$rd, $rs1, $imm, $imm2", [], InstFormatOther> {
	bits<5> imm;
	bits<5> imm2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-30} = 0x2;
	let Inst{29-25} = imm{4-0};
	let Inst{24-20} = imm2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x0;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x5b;
}
class RVInst_CV_BCLR<dag outs, dag ins> : RVInst<outs, ins, "cv.bclr", "$rd, $rs1, $imm, $imm2", [], InstFormatOther> {
	bits<5> imm;
	bits<5> imm2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-30} = 0x0;
	let Inst{29-25} = imm{4-0};
	let Inst{24-20} = imm2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x1;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x5b;
}
class RVInst_CV_BSET<dag outs, dag ins> : RVInst<outs, ins, "cv.bset", "$rd, $rs1, $imm, $imm2", [], InstFormatOther> {
	bits<5> imm;
	bits<5> imm2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-30} = 0x1;
	let Inst{29-25} = imm{4-0};
	let Inst{24-20} = imm2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x1;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x5b;
}
class RVInst_CV_BITREV<dag outs, dag ins> : RVInst<outs, ins, "cv.bitrev", "$rd, $rs1, $imm, $imm2", [], InstFormatOther> {
	bits<5> imm;
	bits<5> imm2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-30} = 0x3;
	let Inst{29-25} = imm{4-0};
	let Inst{24-20} = imm2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x1;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x5b;
}
class RVInst_CV_EXTRACTR<dag outs, dag ins> : RVInst<outs, ins, "cv.extractr", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-25} = 0x18;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x3;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x2b;
}
class RVInst_CV_EXTRACTUR<dag outs, dag ins> : RVInst<outs, ins, "cv.extractur", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-25} = 0x19;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x3;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x2b;
}
class RVInst_CV_INSERTR<dag outs, dag ins> : RVInst<outs, ins, "cv.insertr", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-25} = 0x1a;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x3;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x2b;
}
class RVInst_CV_BCLRR<dag outs, dag ins> : RVInst<outs, ins, "cv.bclrr", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-25} = 0x1c;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x3;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x2b;
}
class RVInst_CV_BSETR<dag outs, dag ins> : RVInst<outs, ins, "cv.bsetr", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-25} = 0x1d;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x3;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x2b;
}
class RVInst_CV_ROR<dag outs, dag ins> : RVInst<outs, ins, "cv.ror", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-25} = 0x20;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x3;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x2b;
}
class RVInst_CV_FF1<dag outs, dag ins> : RVInst<outs, ins, "cv.ff1", "$rd, $rs1", [], InstFormatOther> {
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-25} = 0x21;
	let Inst{24-20} = 0x0;
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x3;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x2b;
}
class RVInst_CV_FL1<dag outs, dag ins> : RVInst<outs, ins, "cv.fl1", "$rd, $rs1", [], InstFormatOther> {
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-25} = 0x22;
	let Inst{24-20} = 0x0;
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x3;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x2b;
}
class RVInst_CV_CLB<dag outs, dag ins> : RVInst<outs, ins, "cv.clb", "$rd, $rs1", [], InstFormatOther> {
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-25} = 0x23;
	let Inst{24-20} = 0x0;
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x3;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x2b;
}
class RVInst_CV_CNT<dag outs, dag ins> : RVInst<outs, ins, "cv.cnt", "$rd, $rs1", [], InstFormatOther> {
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-25} = 0x24;
	let Inst{24-20} = 0x0;
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x3;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x2b;
}
class RVInst_CV_BEQIMM<dag outs, dag ins> : RVInst<outs, ins, "cv.beqimm", "$rs1, $imm2, $imm", [], InstFormatOther> {
	bits<5> imm2;
	bits<5> rs1;
	bits<12> imm;
	let Inst{31-31} = imm{11-11};
	let Inst{30-25} = imm{9-4};
	let Inst{24-20} = imm2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x6;
	let Inst{11-8} = imm{3-0};
	let Inst{7-7} = imm{10-10};
	let Opcode = 0xb;
}
class RVInst_CV_BNEIMM<dag outs, dag ins> : RVInst<outs, ins, "cv.bneimm", "$rs1, $imm2, $imm", [], InstFormatOther> {
	bits<5> imm2;
	bits<5> rs1;
	bits<12> imm;
	let Inst{31-31} = imm{11-11};
	let Inst{30-25} = imm{9-4};
	let Inst{24-20} = imm2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x7;
	let Inst{11-8} = imm{3-0};
	let Inst{7-7} = imm{10-10};
	let Opcode = 0xb;
}
