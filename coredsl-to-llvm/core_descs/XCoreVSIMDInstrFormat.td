class RVInst_CV_ADD_H<dag outs, dag ins> : RVInst<outs, ins, "cv.add.h", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x0;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x0;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_ADD_SC_H<dag outs, dag ins> : RVInst<outs, ins, "cv.add.sc.h", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x0;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x4;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_ADD_SCI_H<dag outs, dag ins> : RVInst<outs, ins, "cv.add.sci.h", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x0;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x6;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_ADD_B<dag outs, dag ins> : RVInst<outs, ins, "cv.add.b", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x0;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x1;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_ADD_SC_B<dag outs, dag ins> : RVInst<outs, ins, "cv.add.sc.b", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x0;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x5;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_ADD_SCI_B<dag outs, dag ins> : RVInst<outs, ins, "cv.add.sci.b", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x0;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x7;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_SUB_H<dag outs, dag ins> : RVInst<outs, ins, "cv.sub.h", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x1;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x0;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_SUB_SC_H<dag outs, dag ins> : RVInst<outs, ins, "cv.sub.sc.h", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x1;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x4;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_SUB_SCI_H<dag outs, dag ins> : RVInst<outs, ins, "cv.sub.sci.h", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x1;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x6;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_SUB_B<dag outs, dag ins> : RVInst<outs, ins, "cv.sub.b", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x1;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x1;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_SUB_SC_B<dag outs, dag ins> : RVInst<outs, ins, "cv.sub.sc.b", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x1;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x5;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_SUB_SCI_B<dag outs, dag ins> : RVInst<outs, ins, "cv.sub.sci.b", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x1;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x7;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_AVG_H<dag outs, dag ins> : RVInst<outs, ins, "cv.avg.h", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x2;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x0;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_AVG_SC_H<dag outs, dag ins> : RVInst<outs, ins, "cv.avg.sc.h", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x2;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x4;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_AVG_SCI_H<dag outs, dag ins> : RVInst<outs, ins, "cv.avg.sci.h", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x2;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x6;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_AVG_B<dag outs, dag ins> : RVInst<outs, ins, "cv.avg.b", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x2;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x1;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_AVG_SC_B<dag outs, dag ins> : RVInst<outs, ins, "cv.avg.sc.b", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x2;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x5;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_AVG_SCI_B<dag outs, dag ins> : RVInst<outs, ins, "cv.avg.sci.b", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x2;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x7;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_AVGU_H<dag outs, dag ins> : RVInst<outs, ins, "cv.avgu.h", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x3;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x0;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_AVGU_SC_H<dag outs, dag ins> : RVInst<outs, ins, "cv.avgu.sc.h", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x3;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x4;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_AVGU_SCI_H<dag outs, dag ins> : RVInst<outs, ins, "cv.avgu.sci.h", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x3;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x6;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_AVGU_B<dag outs, dag ins> : RVInst<outs, ins, "cv.avgu.b", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x3;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x1;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_AVGU_SC_B<dag outs, dag ins> : RVInst<outs, ins, "cv.avgu.sc.b", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x3;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x5;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_AVGU_SCI_B<dag outs, dag ins> : RVInst<outs, ins, "cv.avgu.sci.b", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x3;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x7;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_MIN_H<dag outs, dag ins> : RVInst<outs, ins, "cv.min.h", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x4;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x0;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_MIN_SC_H<dag outs, dag ins> : RVInst<outs, ins, "cv.min.sc.h", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x4;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x4;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_MIN_SCI_H<dag outs, dag ins> : RVInst<outs, ins, "cv.min.sci.h", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x4;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x6;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_MIN_B<dag outs, dag ins> : RVInst<outs, ins, "cv.min.b", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x4;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x1;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_MIN_SC_B<dag outs, dag ins> : RVInst<outs, ins, "cv.min.sc.b", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x4;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x5;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_MIN_SCI_B<dag outs, dag ins> : RVInst<outs, ins, "cv.min.sci.b", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x4;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x7;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_MINU_H<dag outs, dag ins> : RVInst<outs, ins, "cv.minu.h", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x5;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x0;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_MINU_SC_H<dag outs, dag ins> : RVInst<outs, ins, "cv.minu.sc.h", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x5;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x4;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_MINU_SCI_H<dag outs, dag ins> : RVInst<outs, ins, "cv.minu.sci.h", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x5;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x6;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_MINU_B<dag outs, dag ins> : RVInst<outs, ins, "cv.minu.b", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x5;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x1;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_MINU_SC_B<dag outs, dag ins> : RVInst<outs, ins, "cv.minu.sc.b", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x5;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x5;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_MINU_SCI_B<dag outs, dag ins> : RVInst<outs, ins, "cv.minu.sci.b", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x5;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x7;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_MAX_H<dag outs, dag ins> : RVInst<outs, ins, "cv.max.h", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x6;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x0;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_MAX_SC_H<dag outs, dag ins> : RVInst<outs, ins, "cv.max.sc.h", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x6;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x4;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_MAX_SCI_H<dag outs, dag ins> : RVInst<outs, ins, "cv.max.sci.h", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x6;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x6;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_MAX_B<dag outs, dag ins> : RVInst<outs, ins, "cv.max.b", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x6;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x1;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_MAX_SC_B<dag outs, dag ins> : RVInst<outs, ins, "cv.max.sc.b", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x6;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x5;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_MAX_SCI_B<dag outs, dag ins> : RVInst<outs, ins, "cv.max.sci.b", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x6;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x7;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_MAXU_H<dag outs, dag ins> : RVInst<outs, ins, "cv.maxu.h", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x7;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x0;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_MAXU_SC_H<dag outs, dag ins> : RVInst<outs, ins, "cv.maxu.sc.h", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x7;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x4;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_MAXU_SCI_H<dag outs, dag ins> : RVInst<outs, ins, "cv.maxu.sci.h", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x7;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x6;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_MAXU_B<dag outs, dag ins> : RVInst<outs, ins, "cv.maxu.b", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x7;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x1;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_MAXU_SC_B<dag outs, dag ins> : RVInst<outs, ins, "cv.maxu.sc.b", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x7;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x5;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_MAXU_SCI_B<dag outs, dag ins> : RVInst<outs, ins, "cv.maxu.sci.b", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x7;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x7;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_SRL_H<dag outs, dag ins> : RVInst<outs, ins, "cv.srl.h", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x8;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x0;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_SRL_SC_H<dag outs, dag ins> : RVInst<outs, ins, "cv.srl.sc.h", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x8;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x4;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_SRL_SCI_H<dag outs, dag ins> : RVInst<outs, ins, "cv.srl.sci.h", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x8;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x6;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_SRL_B<dag outs, dag ins> : RVInst<outs, ins, "cv.srl.b", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x8;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x1;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_SRL_SC_B<dag outs, dag ins> : RVInst<outs, ins, "cv.srl.sc.b", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x8;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x5;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_SRL_SCI_B<dag outs, dag ins> : RVInst<outs, ins, "cv.srl.sci.b", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x8;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x7;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_SRA_H<dag outs, dag ins> : RVInst<outs, ins, "cv.sra.h", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x9;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x0;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_SRA_SC_H<dag outs, dag ins> : RVInst<outs, ins, "cv.sra.sc.h", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x9;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x4;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_SRA_SCI_H<dag outs, dag ins> : RVInst<outs, ins, "cv.sra.sci.h", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x9;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x6;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_SRA_B<dag outs, dag ins> : RVInst<outs, ins, "cv.sra.b", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x9;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x1;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_SRA_SC_B<dag outs, dag ins> : RVInst<outs, ins, "cv.sra.sc.b", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x9;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x5;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_SRA_SCI_B<dag outs, dag ins> : RVInst<outs, ins, "cv.sra.sci.b", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x9;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x7;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_SLL_H<dag outs, dag ins> : RVInst<outs, ins, "cv.sll.h", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0xa;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x0;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_SLL_SC_H<dag outs, dag ins> : RVInst<outs, ins, "cv.sll.sc.h", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0xa;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x4;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_SLL_SCI_H<dag outs, dag ins> : RVInst<outs, ins, "cv.sll.sci.h", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0xa;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x6;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_SLL_B<dag outs, dag ins> : RVInst<outs, ins, "cv.sll.b", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0xa;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x1;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_SLL_SC_B<dag outs, dag ins> : RVInst<outs, ins, "cv.sll.sc.b", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0xa;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x5;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_SLL_SCI_B<dag outs, dag ins> : RVInst<outs, ins, "cv.sll.sci.b", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0xa;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x7;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_OR_H<dag outs, dag ins> : RVInst<outs, ins, "cv.or.h", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0xb;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x0;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_OR_SC_H<dag outs, dag ins> : RVInst<outs, ins, "cv.or.sc.h", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0xb;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x4;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_OR_SCI_H<dag outs, dag ins> : RVInst<outs, ins, "cv.or.sci.h", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0xb;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x6;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_OR_B<dag outs, dag ins> : RVInst<outs, ins, "cv.or.b", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0xb;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x1;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_OR_SC_B<dag outs, dag ins> : RVInst<outs, ins, "cv.or.sc.b", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0xb;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x5;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_OR_SCI_B<dag outs, dag ins> : RVInst<outs, ins, "cv.or.sci.b", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0xb;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x7;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_XOR_H<dag outs, dag ins> : RVInst<outs, ins, "cv.xor.h", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0xc;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x0;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_XOR_SC_H<dag outs, dag ins> : RVInst<outs, ins, "cv.xor.sc.h", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0xc;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x4;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_XOR_SCI_H<dag outs, dag ins> : RVInst<outs, ins, "cv.xor.sci.h", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0xc;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x6;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_XOR_B<dag outs, dag ins> : RVInst<outs, ins, "cv.xor.b", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0xc;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x1;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_XOR_SC_B<dag outs, dag ins> : RVInst<outs, ins, "cv.xor.sc.b", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0xc;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x5;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_XOR_SCI_B<dag outs, dag ins> : RVInst<outs, ins, "cv.xor.sci.b", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0xc;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x7;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_AND_H<dag outs, dag ins> : RVInst<outs, ins, "cv.and.h", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0xd;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x0;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_AND_SC_H<dag outs, dag ins> : RVInst<outs, ins, "cv.and.sc.h", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0xd;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x4;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_AND_SCI_H<dag outs, dag ins> : RVInst<outs, ins, "cv.and.sci.h", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0xd;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x6;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_AND_B<dag outs, dag ins> : RVInst<outs, ins, "cv.and.b", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0xd;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x1;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_AND_SC_B<dag outs, dag ins> : RVInst<outs, ins, "cv.and.sc.b", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0xd;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x5;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_AND_SCI_B<dag outs, dag ins> : RVInst<outs, ins, "cv.and.sci.b", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0xd;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x7;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_ABS_H<dag outs, dag ins> : RVInst<outs, ins, "cv.abs.h", "$rd, $rs1", [], InstFormatOther> {
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0xe;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = 0x0;
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x0;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_ABS_B<dag outs, dag ins> : RVInst<outs, ins, "cv.abs.b", "$rd, $rs1", [], InstFormatOther> {
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0xe;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = 0x0;
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x1;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_EXTRACT_H<dag outs, dag ins> : RVInst<outs, ins, "cv.extract.h", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x17;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x0;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_EXTRACT_B<dag outs, dag ins> : RVInst<outs, ins, "cv.extract.b", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x17;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x1;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_EXTRACTU_H<dag outs, dag ins> : RVInst<outs, ins, "cv.extractu.h", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x17;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x2;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_EXTRACTU_B<dag outs, dag ins> : RVInst<outs, ins, "cv.extractu.b", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x17;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x3;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_INSERT_H<dag outs, dag ins> : RVInst<outs, ins, "cv.insert.h", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x17;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x4;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_INSERT_B<dag outs, dag ins> : RVInst<outs, ins, "cv.insert.b", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x17;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x5;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_DOTUP_H<dag outs, dag ins> : RVInst<outs, ins, "cv.dotup.h", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x10;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x0;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_DOTUP_SC_H<dag outs, dag ins> : RVInst<outs, ins, "cv.dotup.sc.h", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x10;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x4;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_DOTUP_SCI_H<dag outs, dag ins> : RVInst<outs, ins, "cv.dotup.sci.h", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x10;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x6;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_DOTUP_B<dag outs, dag ins> : RVInst<outs, ins, "cv.dotup.b", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x10;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x1;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_DOTUP_SC_B<dag outs, dag ins> : RVInst<outs, ins, "cv.dotup.sc.b", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x10;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x5;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_DOTUP_SCI_B<dag outs, dag ins> : RVInst<outs, ins, "cv.dotup.sci.b", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x10;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x7;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_DOTUSP_H<dag outs, dag ins> : RVInst<outs, ins, "cv.dotusp.h", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x11;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x0;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_DOTUSP_SC_H<dag outs, dag ins> : RVInst<outs, ins, "cv.dotusp.sc.h", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x11;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x4;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_DOTUSP_SCI_H<dag outs, dag ins> : RVInst<outs, ins, "cv.dotusp.sci.h", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x11;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x6;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_DOTUSP_B<dag outs, dag ins> : RVInst<outs, ins, "cv.dotusp.b", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x11;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x1;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_DOTUSP_SC_B<dag outs, dag ins> : RVInst<outs, ins, "cv.dotusp.sc.b", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x11;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x5;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_DOTUSP_SCI_B<dag outs, dag ins> : RVInst<outs, ins, "cv.dotusp.sci.b", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x11;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x7;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_DOTSP_H<dag outs, dag ins> : RVInst<outs, ins, "cv.dotsp.h", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x12;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x0;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_DOTSP_SC_H<dag outs, dag ins> : RVInst<outs, ins, "cv.dotsp.sc.h", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x12;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x4;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_DOTSP_SCI_H<dag outs, dag ins> : RVInst<outs, ins, "cv.dotsp.sci.h", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x12;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x6;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_DOTSP_B<dag outs, dag ins> : RVInst<outs, ins, "cv.dotsp.b", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x12;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x1;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_DOTSP_SC_B<dag outs, dag ins> : RVInst<outs, ins, "cv.dotsp.sc.b", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x12;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x5;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_DOTSP_SCI_B<dag outs, dag ins> : RVInst<outs, ins, "cv.dotsp.sci.b", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x12;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x7;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_SDOTUP_H<dag outs, dag ins> : RVInst<outs, ins, "cv.sdotup.h", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x13;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x0;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_SDOTUP_SC_H<dag outs, dag ins> : RVInst<outs, ins, "cv.sdotup.sc.h", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x13;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x4;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_SDOTUP_SCI_H<dag outs, dag ins> : RVInst<outs, ins, "cv.sdotup.sci.h", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x13;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x6;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_SDOTUP_B<dag outs, dag ins> : RVInst<outs, ins, "cv.sdotup.b", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x13;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x1;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_SDOTUP_SC_B<dag outs, dag ins> : RVInst<outs, ins, "cv.sdotup.sc.b", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x13;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x5;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_SDOTUP_SCI_B<dag outs, dag ins> : RVInst<outs, ins, "cv.sdotup.sci.b", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x13;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x7;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_SDOTUSP_H<dag outs, dag ins> : RVInst<outs, ins, "cv.sdotusp.h", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x14;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x0;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_SDOTUSP_SC_H<dag outs, dag ins> : RVInst<outs, ins, "cv.sdotusp.sc.h", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x14;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x4;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_SDOTUSP_SCI_H<dag outs, dag ins> : RVInst<outs, ins, "cv.sdotusp.sci.h", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x14;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x6;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_SDOTUSP_B<dag outs, dag ins> : RVInst<outs, ins, "cv.sdotusp.b", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x14;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x1;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_SDOTUSP_SC_B<dag outs, dag ins> : RVInst<outs, ins, "cv.sdotusp.sc.b", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x14;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x5;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_SDOTUSP_SCI_B<dag outs, dag ins> : RVInst<outs, ins, "cv.sdotusp.sci.b", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x14;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x7;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_SDOTSP_H<dag outs, dag ins> : RVInst<outs, ins, "cv.sdotsp.h", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x15;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x0;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_SDOTSP_SC_H<dag outs, dag ins> : RVInst<outs, ins, "cv.sdotsp.sc.h", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x15;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x4;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_SDOTSP_SCI_H<dag outs, dag ins> : RVInst<outs, ins, "cv.sdotsp.sci.h", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x15;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x6;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_SDOTSP_B<dag outs, dag ins> : RVInst<outs, ins, "cv.sdotsp.b", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x15;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x1;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_SDOTSP_SC_B<dag outs, dag ins> : RVInst<outs, ins, "cv.sdotsp.sc.b", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x15;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x5;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_SDOTSP_SCI_B<dag outs, dag ins> : RVInst<outs, ins, "cv.sdotsp.sci.b", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x15;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x7;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_SHUFFLE_H<dag outs, dag ins> : RVInst<outs, ins, "cv.shuffle.h", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x18;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x0;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_SHUFFLE_SCI_H<dag outs, dag ins> : RVInst<outs, ins, "cv.shuffle.sci.h", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x18;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x6;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_SHUFFLE_B<dag outs, dag ins> : RVInst<outs, ins, "cv.shuffle.b", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x18;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x1;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_SHUFFLEI0_SCI_B<dag outs, dag ins> : RVInst<outs, ins, "cv.shufflei0.sci.b", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x18;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x7;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_SHUFFLEI1_SCI_B<dag outs, dag ins> : RVInst<outs, ins, "cv.shufflei1.sci.b", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x19;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x7;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_SHUFFLEI2_SCI_B<dag outs, dag ins> : RVInst<outs, ins, "cv.shufflei2.sci.b", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x1a;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x7;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_SHUFFLEI3_SCI_B<dag outs, dag ins> : RVInst<outs, ins, "cv.shufflei3.sci.b", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x1b;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x7;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_SHUFFLE2_H<dag outs, dag ins> : RVInst<outs, ins, "cv.shuffle2.h", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x1c;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x0;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_SHUFFLE2_B<dag outs, dag ins> : RVInst<outs, ins, "cv.shuffle2.b", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x1c;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x1;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_PACK<dag outs, dag ins> : RVInst<outs, ins, "cv.pack", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x1e;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x0;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_PACK_H<dag outs, dag ins> : RVInst<outs, ins, "cv.pack.h", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x1e;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x1;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x0;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_PACKHI_B<dag outs, dag ins> : RVInst<outs, ins, "cv.packhi.b", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x1f;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x1;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x1;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_PACKLO_B<dag outs, dag ins> : RVInst<outs, ins, "cv.packlo.b", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x1f;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x1;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_CMPEQ_H<dag outs, dag ins> : RVInst<outs, ins, "cv.cmpeq.h", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x0;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x0;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_CMPEQ_SC_H<dag outs, dag ins> : RVInst<outs, ins, "cv.cmpeq.sc.h", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x0;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x4;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_CMPEQ_SCI_H<dag outs, dag ins> : RVInst<outs, ins, "cv.cmpeq.sci.h", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x0;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x6;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_CMPEQ_B<dag outs, dag ins> : RVInst<outs, ins, "cv.cmpeq.b", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x0;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x1;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_CMPEQ_SC_B<dag outs, dag ins> : RVInst<outs, ins, "cv.cmpeq.sc.b", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x0;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x5;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_CMPEQ_SCI_B<dag outs, dag ins> : RVInst<outs, ins, "cv.cmpeq.sci.b", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x0;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x7;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_CMPNE_H<dag outs, dag ins> : RVInst<outs, ins, "cv.cmpne.h", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x1;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x0;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_CMPNE_SC_H<dag outs, dag ins> : RVInst<outs, ins, "cv.cmpne.sc.h", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x1;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x4;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_CMPNE_SCI_H<dag outs, dag ins> : RVInst<outs, ins, "cv.cmpne.sci.h", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x1;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x6;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_CMPNE_B<dag outs, dag ins> : RVInst<outs, ins, "cv.cmpne.b", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x1;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x1;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_CMPNE_SC_B<dag outs, dag ins> : RVInst<outs, ins, "cv.cmpne.sc.b", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x1;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x5;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_CMPNE_SCI_B<dag outs, dag ins> : RVInst<outs, ins, "cv.cmpne.sci.b", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x1;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x7;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_CMPGT_H<dag outs, dag ins> : RVInst<outs, ins, "cv.cmpgt.h", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x2;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x0;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_CMPGT_SC_H<dag outs, dag ins> : RVInst<outs, ins, "cv.cmpgt.sc.h", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x2;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x4;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_CMPGT_SCI_H<dag outs, dag ins> : RVInst<outs, ins, "cv.cmpgt.sci.h", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x2;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x6;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_CMPGT_B<dag outs, dag ins> : RVInst<outs, ins, "cv.cmpgt.b", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x2;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x1;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_CMPGT_SC_B<dag outs, dag ins> : RVInst<outs, ins, "cv.cmpgt.sc.b", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x2;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x5;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_CMPGT_SCI_B<dag outs, dag ins> : RVInst<outs, ins, "cv.cmpgt.sci.b", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x2;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x7;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_CMPGE_H<dag outs, dag ins> : RVInst<outs, ins, "cv.cmpge.h", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x3;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x0;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_CMPGE_SC_H<dag outs, dag ins> : RVInst<outs, ins, "cv.cmpge.sc.h", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x3;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x4;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_CMPGE_SCI_H<dag outs, dag ins> : RVInst<outs, ins, "cv.cmpge.sci.h", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x3;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x6;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_CMPGE_B<dag outs, dag ins> : RVInst<outs, ins, "cv.cmpge.b", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x3;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x1;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_CMPGE_SC_B<dag outs, dag ins> : RVInst<outs, ins, "cv.cmpge.sc.b", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x3;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x5;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_CMPGE_SCI_B<dag outs, dag ins> : RVInst<outs, ins, "cv.cmpge.sci.b", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x3;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x7;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_CMPLT_H<dag outs, dag ins> : RVInst<outs, ins, "cv.cmplt.h", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x4;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x0;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_CMPLT_SC_H<dag outs, dag ins> : RVInst<outs, ins, "cv.cmplt.sc.h", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x4;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x4;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_CMPLT_SCI_H<dag outs, dag ins> : RVInst<outs, ins, "cv.cmplt.sci.h", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x4;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x6;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_CMPLT_B<dag outs, dag ins> : RVInst<outs, ins, "cv.cmplt.b", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x4;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x1;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_CMPLT_SC_B<dag outs, dag ins> : RVInst<outs, ins, "cv.cmplt.sc.b", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x4;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x5;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_CMPLT_SCI_B<dag outs, dag ins> : RVInst<outs, ins, "cv.cmplt.sci.b", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x4;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x7;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_CMPLE_H<dag outs, dag ins> : RVInst<outs, ins, "cv.cmple.h", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x5;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x0;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_CMPLE_SC_H<dag outs, dag ins> : RVInst<outs, ins, "cv.cmple.sc.h", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x5;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x4;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_CMPLE_SCI_H<dag outs, dag ins> : RVInst<outs, ins, "cv.cmple.sci.h", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x5;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x6;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_CMPLE_B<dag outs, dag ins> : RVInst<outs, ins, "cv.cmple.b", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x5;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x1;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_CMPLE_SC_B<dag outs, dag ins> : RVInst<outs, ins, "cv.cmple.sc.b", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x5;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x5;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_CMPLE_SCI_B<dag outs, dag ins> : RVInst<outs, ins, "cv.cmple.sci.b", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x5;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x7;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_CMPGTU_H<dag outs, dag ins> : RVInst<outs, ins, "cv.cmpgtu.h", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x6;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x0;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_CMPGTU_SC_H<dag outs, dag ins> : RVInst<outs, ins, "cv.cmpgtu.sc.h", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x6;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x4;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_CMPGTU_SCI_H<dag outs, dag ins> : RVInst<outs, ins, "cv.cmpgtu.sci.h", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x6;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x6;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_CMPGTU_B<dag outs, dag ins> : RVInst<outs, ins, "cv.cmpgtu.b", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x6;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x1;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_CMPGTU_SC_B<dag outs, dag ins> : RVInst<outs, ins, "cv.cmpgtu.sc.b", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x6;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x5;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_CMPGTU_SCI_B<dag outs, dag ins> : RVInst<outs, ins, "cv.cmpgtu.sci.b", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x6;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x7;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_CMPGEU_H<dag outs, dag ins> : RVInst<outs, ins, "cv.cmpgeu.h", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x7;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x0;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_CMPGEU_SC_H<dag outs, dag ins> : RVInst<outs, ins, "cv.cmpgeu.sc.h", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x7;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x4;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_CMPGEU_SCI_H<dag outs, dag ins> : RVInst<outs, ins, "cv.cmpgeu.sci.h", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x7;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x6;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_CMPGEU_B<dag outs, dag ins> : RVInst<outs, ins, "cv.cmpgeu.b", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x7;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x1;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_CMPGEU_SC_B<dag outs, dag ins> : RVInst<outs, ins, "cv.cmpgeu.sc.b", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x7;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x5;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_CMPGEU_SCI_B<dag outs, dag ins> : RVInst<outs, ins, "cv.cmpgeu.sci.b", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x7;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x7;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_CMPLTU_H<dag outs, dag ins> : RVInst<outs, ins, "cv.cmpltu.h", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x8;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x0;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_CMPLTU_SC_H<dag outs, dag ins> : RVInst<outs, ins, "cv.cmpltu.sc.h", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x8;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x4;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_CMPLTU_SCI_H<dag outs, dag ins> : RVInst<outs, ins, "cv.cmpltu.sci.h", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x8;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x6;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_CMPLTU_B<dag outs, dag ins> : RVInst<outs, ins, "cv.cmpltu.b", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x8;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x1;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_CMPLTU_SC_B<dag outs, dag ins> : RVInst<outs, ins, "cv.cmpltu.sc.b", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x8;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x5;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_CMPLTU_SCI_B<dag outs, dag ins> : RVInst<outs, ins, "cv.cmpltu.sci.b", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x8;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x7;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_CMPLEU_H<dag outs, dag ins> : RVInst<outs, ins, "cv.cmpleu.h", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x9;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x0;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_CMPLEU_SC_H<dag outs, dag ins> : RVInst<outs, ins, "cv.cmpleu.sc.h", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x9;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x4;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_CMPLEU_SCI_H<dag outs, dag ins> : RVInst<outs, ins, "cv.cmpleu.sci.h", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x9;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x6;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_CMPLEU_B<dag outs, dag ins> : RVInst<outs, ins, "cv.cmpleu.b", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x9;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x1;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_CMPLEU_SC_B<dag outs, dag ins> : RVInst<outs, ins, "cv.cmpleu.sc.b", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x9;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x5;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_CMPLEU_SCI_B<dag outs, dag ins> : RVInst<outs, ins, "cv.cmpleu.sci.b", "$rd, $rs1, $imm", [], InstFormatOther> {
	bits<6> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x9;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = imm{0-0};
	let Inst{24-20} = imm{5-1};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x7;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_CPLXMUL_R<dag outs, dag ins> : RVInst<outs, ins, "cv.cplxmul.r", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0xa;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x0;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_CPLXMUL_R_DIV2<dag outs, dag ins> : RVInst<outs, ins, "cv.cplxmul.r.div2", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0xa;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x2;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_CPLXMUL_R_DIV4<dag outs, dag ins> : RVInst<outs, ins, "cv.cplxmul.r.div4", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0xa;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x4;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_CPLXMUL_R_DIV8<dag outs, dag ins> : RVInst<outs, ins, "cv.cplxmul.r.div8", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0xa;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x6;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_CPLXMUL_I<dag outs, dag ins> : RVInst<outs, ins, "cv.cplxmul.i", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0xa;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = 0x1;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x0;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_CPLXMUL_I_DIV2<dag outs, dag ins> : RVInst<outs, ins, "cv.cplxmul.i.div2", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0xa;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = 0x1;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x2;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_CPLXMUL_I_DIV4<dag outs, dag ins> : RVInst<outs, ins, "cv.cplxmul.i.div4", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0xa;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = 0x1;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x4;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_CPLXMUL_I_DIV8<dag outs, dag ins> : RVInst<outs, ins, "cv.cplxmul.i.div8", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0xa;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = 0x1;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x6;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_CPLXCONJ<dag outs, dag ins> : RVInst<outs, ins, "cv.cplxconj", "$rd, $rs1", [], InstFormatOther> {
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0xb;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = 0x0;
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x0;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_SUBROTMJ<dag outs, dag ins> : RVInst<outs, ins, "cv.subrotmj", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0xc;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x0;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_SUBROTMJ_DIV2<dag outs, dag ins> : RVInst<outs, ins, "cv.subrotmj.div2", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0xc;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x2;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_SUBROTMJ_DIV4<dag outs, dag ins> : RVInst<outs, ins, "cv.subrotmj.div4", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0xc;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x4;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_SUBROTMJ_DIV8<dag outs, dag ins> : RVInst<outs, ins, "cv.subrotmj.div8", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0xc;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x6;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_ADD_DIV2<dag outs, dag ins> : RVInst<outs, ins, "cv.add.div2", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0xd;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x2;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_ADD_DIV4<dag outs, dag ins> : RVInst<outs, ins, "cv.add.div4", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0xd;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x4;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_ADD_DIV8<dag outs, dag ins> : RVInst<outs, ins, "cv.add.div8", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0xd;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x6;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_SUB_DIV2<dag outs, dag ins> : RVInst<outs, ins, "cv.sub.div2", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0xe;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x2;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_SUB_DIV4<dag outs, dag ins> : RVInst<outs, ins, "cv.sub.div4", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0xe;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x4;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
class RVInst_CV_SUB_DIV8<dag outs, dag ins> : RVInst<outs, ins, "cv.sub.div8", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0xe;
	let Inst{26-26} = 0x1;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x6;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x7b;
}
