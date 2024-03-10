#include "JIT.hpp"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/ExecutionEngine/JITEventListener.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/PassManager.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCDisassembler/MCRelocationInfo.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Scalar/LICM.h"
#include <bits/chrono.h>
#include <chrono>
#include <cstdlib>
#include <elf.h>
#include <err.h>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <ios>
#include <libelf.h>
#include <llvm/ADT/DenseMap.h>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <vector>
#include <span>

#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/TargetSelect.h"
#include <csignal>
#include <tuple>

#define UBFX(val, start, end) (((val) >> start) & ((1 << (end - start + 1)) - 1))

struct __attribute__((packed)) RV32State
{
    uint32_t regs[32];
};

using FragFuncT = uint32_t (*)(RV32State*);
using ExtFuncT = void (*)(uint32_t* rd, uint32_t* rs1, uint32_t* rs2, uint32_t imm0, uint32_t imm1);

bool doExit;

// This will be called (from jit'ed code) on exceptions
void RV32I_Except(RV32State* state)
{
    doExit = 1;
}

// This will be called (from jit'ed code) when executing an ecall
uint32_t RV32I_ECall(RV32State* state)
{
    uint32_t a0 = state->regs[10];
    uint32_t a1 = state->regs[11];

    switch (a0)
    {
        case 0:
        {
            putchar(a1);
            return 0;
        }
        case 1:
        {
            // Details don't matter, just get some time for benchmarking
            return (uint32_t)std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::system_clock::now().time_since_epoch())
                .count();
        }
        case 2: doExit = 1; return 0;
        default: return 0;
    }
}

struct JITInstr
{
    uint32_t mask;
    uint32_t match;
    ExtFuncT impl;
    CDSLInstr const* cdsl;
};
static std::vector<JITInstr> jitInstrs;
static JITInstr const* TryMatchExtInstr(uint32_t pattern)
{
    for (size_t i = 0; i < jitInstrs.size(); i++)
        if ((pattern & jitInstrs[i].mask) == jitInstrs[i].match)
            return &jitInstrs[i];
    return nullptr;
}

static uint32_t GetValueOfField(uint32_t instr, CDSLInstr const& cdsl, CDSLInstr::Field::FieldType f)
{
    uint32_t acc = 0;
    // "src" and "dst" are inverted here, as we read fields from an existing instruction
    for (auto const& field : cdsl.fields)
        if (field.type == f)
            acc |= ((instr >> field.dstOffset) & (((1 << field.len) - 1))) << field.srcOffset;

    if (f == CDSLInstr::Field::ID_IMM0 && cdsl.SignedImm(0))
    {
        int shamt = (32 - cdsl.GetImmLen(0));
        acc = (int32_t)(acc << shamt) >> shamt;
    }

    return acc;
}

llvm::ExecutionEngine* Compile(std::unique_ptr<llvm::Module> mod)
{
    std::string error;
    llvm::TargetOptions options;

    llvm::EngineBuilder builder(std::move(mod));
    builder.setEngineKind(llvm::EngineKind::JIT);
    builder.setErrorStr(&error);
    builder.setOptLevel(llvm::CodeGenOpt::Less);
    builder.setTargetOptions(options);
    builder.setVerifyModules(true);

    llvm::ExecutionEngine* engine = builder.create();
    if (!engine)
        err(1, "could not create engine: %s", error.c_str());

    engine->addGlobalMapping("RV32I_ECall", reinterpret_cast<uint64_t>(RV32I_ECall));
    engine->addGlobalMapping("RV32I_Except", reinterpret_cast<uint64_t>(RV32I_Except));
    for (auto const& inst : jitInstrs)
        engine->addGlobalMapping(std::string("impl" + inst.cdsl->name), reinterpret_cast<uint64_t>(inst.impl));
    return engine;
}

static void PreprocessInstructions(std::vector<CDSLInstr> const& instrs, std::unique_ptr<llvm::Module> instrImpls)
{
    jitInstrs.clear();
    auto engine = Compile(std::move(instrImpls));
    jitInstrs.reserve(instrs.size());

    // 1. Compute Mask and Match uint32_t's
    for (size_t i = 0; i < instrs.size(); i++)
    {
        uint32_t mask = 0;
        uint32_t match = 0;

        for (auto const& f : instrs[i].fields)
            if (f.type == CDSLInstr::Field::CONST)
            {
                mask |= (((1 << f.len) - 1) << f.dstOffset);
                match |= f.value << f.dstOffset;
            }
        jitInstrs.push_back({.mask = mask, .match = match, .cdsl = &instrs[i]});
    }

    // 2. Compile Behavior Funcs
    for (size_t i = 0; i < instrs.size(); i++)
        jitInstrs[i].impl = reinterpret_cast<ExtFuncT>(engine->getFunctionAddress("impl" + instrs[i].name));
}

void* CompileFunc(std::unique_ptr<llvm::Module> mod, const std::string& name)
{
    auto engine = Compile(std::move(mod));
    return reinterpret_cast<void*>(engine->getFunctionAddress(name));
}

static void memrand (uint8_t* dest, size_t n, int seed)
{
    srand(seed);
    for (size_t i = 0; i < n; i++)
        dest[i] = rand() & 0xff;
}

struct Section
{
    std::string name;
    uint32_t addr;
    uint32_t size;
};
std::vector<Section> LoadELF(std::string const& execName, uint8_t* memory, uint32_t& entryPoint)
{
    std::ifstream ifs(execName);
    if (!ifs || ifs.bad()) throw std::invalid_argument("Could not open elf file");
    std::vector<char> content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());

    Elf32_Ehdr* hdr = reinterpret_cast<Elf32_Ehdr*>(content.data());
    Elf32_Shdr* shdr = reinterpret_cast<Elf32_Shdr*>(content.data() + hdr->e_shoff);
    const char* strtab = reinterpret_cast<char*>(content.data() + shdr[hdr->e_shstrndx].sh_offset);

    std::vector<Section> sections;

    for (size_t i = 0; i < hdr->e_shnum; i++)
    {
        //fprintf(stderr, "(%x) %x: %s, %x\n", shdr[i].sh_offset, shdr[i].sh_addr, &strtab[shdr[i].sh_name], shdr[i].sh_size);
        if (shdr[i].sh_addr != 0)
        {
            if (strcmp(&strtab[shdr[i].sh_name], ".bss") == 0)
                memrand(&memory[shdr[i].sh_addr], shdr[i].sh_size, 42);
            else
                memcpy(&memory[shdr[i].sh_addr], content.data() + shdr[i].sh_offset, shdr[i].sh_size);
            sections.push_back({.name = &strtab[shdr[i].sh_name], .addr = shdr[i].sh_addr, .size = shdr[i].sh_size});
        }
    }
    entryPoint = hdr->e_entry;
    return sections;
}
void DumpSection (std::string fileName, uint8_t* memory, Section const& section)
{
    auto of = std::ofstream(fileName, std::ios_base::binary | std::ios_base::out);
    of.write(reinterpret_cast<char*>(&memory[section.addr]), section.size);
}

// Translates Block up to first branch/jump
std::pair<llvm::Function*, llvm::DenseMap<const char*, uint32_t>> TranslateBlock(llvm::Module* mod, uint8_t* memory,
                                                                                 uint32_t startPC)
{
    // For perf measurement, keep track of instruction count in this basic block.
    // Note that this maps from pointer to unsigned, key strings must be non-duplicate.
    llvm::DenseMap<const char*, uint32_t> instrCounts;

    auto ptrT = llvm::PointerType::get(mod->getContext(), 0);
    llvm::LLVMContext& ctx = mod->getContext();
    llvm::FunctionType* funcType = llvm::FunctionType::get(llvm::Type::getInt32Ty(mod->getContext()), {ptrT}, false);

    auto i64 = llvm::Type::getInt64Ty(ctx);
    auto i32 = llvm::Type::getInt32Ty(ctx);
    auto i16 = llvm::Type::getInt16Ty(ctx);
    auto i8 = llvm::Type::getInt8Ty(ctx);
    auto voidT = llvm::Type::getVoidTy(ctx);

    std::stringstream stream;
    stream << "block_" << std::hex << startPC;

    llvm::Function* func = llvm::Function::Create(funcType, llvm::GlobalValue::ExternalLinkage, stream.str(), mod);
    func->addParamAttr(0, llvm::Attribute::NoAlias);

    llvm::BasicBlock* entry = llvm::BasicBlock::Create(mod->getContext(), "entry", func);
    llvm::BasicBlock* block = llvm::BasicBlock::Create(mod->getContext(), "body", func);
    llvm::BasicBlock* firstNonEntry = block;

    llvm::IRBuilder<> build{entry};

    // When calling CoreDSL instructions, this is used as RD
    // (because of noalias/restrict, RD can't point into the register file directly)
    auto tempRD = build.CreateAlloca(i32);

    build.CreateBr(block);
    build.SetInsertPoint(block);

    uint32_t* const memory32 = reinterpret_cast<uint32_t*>(memory);
    uint32_t pc = startPC;

    enum
    {
        OPC_LUI = 0b0110111,
        OPC_AUIPC = 0b0010111,
        OPC_JAL = 0b1101111,
        OPC_JALR = 0b1100111,
        OPC_LOAD = 0b0000011,
        OPC_STORE = 0b0100011,
        OPC_BRANCH = 0b1100011,
        OPC_REG_IMM = 0b0010011,
        OPC_REG_REG = 0b0110011,
        OPC_ENV = 0b1110011,
        OPC_FENCE = 0b0001111,
    };

    llvm::Value* regPtr = func->getArg(0);

    auto getRegister = [&](int reg)
    { return build.CreateGEP(i32, regPtr, {llvm::ConstantInt::get(llvm::Type::getInt64Ty(ctx), reg)}); };

    auto loadRegister = [&](int reg)
    {
        if (reg == 0)
            return static_cast<llvm::Value*>(llvm::ConstantInt::get(i32, 0));
        auto ptr = getRegister(reg);
        return static_cast<llvm::Value*>(build.CreateLoad(i32, ptr));
    };

    auto storeRegister = [&](int reg, llvm::Value* value)
    {
        if (reg != 0)
        {
            build.CreateStore(value, getRegister(reg));
        }
    };

    auto getConstant = [i32](uint32_t asInt) { return llvm::ConstantInt::get(i32, asInt); };

    auto ldStTypeTable = (std::array<llvm::Type*, 3>{i8, i16, i32});
    std::array<llvm::CmpInst::Predicate, 8> branchCondTable = {
        llvm::CmpInst::ICMP_EQ,
        llvm::CmpInst::ICMP_NE,

        // These two are undefined in rv32i
        llvm::CmpInst::BAD_ICMP_PREDICATE,
        llvm::CmpInst::BAD_ICMP_PREDICATE,

        llvm::CmpInst::ICMP_SLT,
        llvm::CmpInst::ICMP_SGE,
        llvm::CmpInst::ICMP_ULT,
        llvm::CmpInst::ICMP_UGE,
    };

    auto memoryPtr = build.CreateIntToPtr(llvm::ConstantInt::get(i64, reinterpret_cast<uint64_t>(memory)), ptrT);

    while (true)
    {
        uint32_t instr = memory32[pc >> 2];

        unsigned rd = (instr >> 7) & 31;
        unsigned rs1 = (instr >> 15) & 31;
        unsigned rs2 = (instr >> 20) & 31;
        unsigned funct3 = (instr >> 12) & 7;
        unsigned funct7 = instr >> 25;

        switch (instr & 127) // major opcode
        {
            case OPC_LUI:
                storeRegister(rd, getConstant(instr & ~(4095)));
                instrCounts["lui"]++;
                break;
            case OPC_AUIPC:
                storeRegister(rd, getConstant((instr & ~(4095)) + pc));
                instrCounts["auipc"]++;
                break;

            case OPC_JAL:
            {
                storeRegister(rd, getConstant(pc + 4));
                uint32_t dst = UBFX(instr, 31, 31) << 20 | UBFX(instr, 12, 19) << 12 | UBFX(instr, 20, 20) << 11 |
                               UBFX(instr, 21, 30) << 1;

                uint32_t mask = (uint32_t)((int32_t)(instr & (1 << 31)) >> 11);
                dst |= mask;
                dst += pc;

                if (dst == startPC)
                    build.CreateBr(firstNonEntry);
                else
                    pc = dst;

                instrCounts[(rd == 0) ? "j" : "jal"]++;
                continue;
            }
            case OPC_JALR:
            {
                storeRegister(rd, getConstant(pc + 4));
                uint32_t offs = instr >> 20 | (uint32_t)((int32_t)(instr & (1 << 31)) >> 20);
                build.CreateRet(build.CreateAdd(loadRegister(rs1), getConstant(offs)));
                instrCounts[(rd == 0) ? "jr" : "jalr"]++;
                return {func, instrCounts};
            }
            case OPC_LOAD:
            {
                if (funct3 > 0b101 || funct3 == 0b011)
                    goto unimplemented;
                uint32_t offs = instr >> 20 | (uint32_t)((int32_t)(instr & (1 << 31)) >> 20);
                // Index to GEP is treated as signed for some reason, so we have to zero extend it manually
                llvm::Value* ptr = build.CreateGEP(i8, memoryPtr, build.CreateZExt(loadRegister(rs1), i64));
                ptr = build.CreateGEP(i8, ptr, {getConstant(offs)});

                llvm::Value* load = build.CreateLoad(ldStTypeTable[funct3 & 0b11], ptr);

                if (funct3 & 0b100)
                    load = build.CreateZExt(load, i32);
                else if (funct3 != 0b010)
                    load = build.CreateSExt(load, i32);

                storeRegister(rd, load);
                instrCounts[std::array<const char*, 6>{"lb", "lh", "lw", "", "lbu", "lhu"}[funct3]]++;
                break;
            }
            case OPC_STORE:
            {
                if (funct3 > 0b010)
                    goto unimplemented;
                uint32_t offs = ((instr >> 20) & ~31) | rd | (uint32_t)((int32_t)(instr & (1 << 31)) >> 20);
                llvm::Value* ptr = build.CreateGEP(i8, memoryPtr, build.CreateZExt(loadRegister(rs1), i64));
                ptr = build.CreateGEP(i8, ptr, {getConstant(offs)});
                llvm::Value* val = loadRegister(rs2);

                if (funct3 != 0b010)
                    val = build.CreateTrunc(val, ldStTypeTable[funct3 & 0b11]);

                build.CreateStore(val, ptr);
                instrCounts[std::array<const char*, 3>{"sb", "sh", "sw"}[funct3]]++;
                break;
            }
            case OPC_BRANCH:
            {
                if (funct3 == 2 || funct3 == 3)
                    goto unimplemented;

                uint32_t offs = UBFX(instr, 31, 31) << 12 | UBFX(instr, 7, 7) << 11 | UBFX(instr, 25, 30) << 5 |
                                UBFX(instr, 8, 11) << 1 | (uint32_t)((int32_t)(instr & (1 << 31)) >> 20);

                llvm::Value* cond = build.CreateICmp(branchCondTable[funct3], loadRegister(rs1), loadRegister(rs2));
                uint32_t dst = pc + offs;
                // single block loop
                if (dst == startPC && false)
                {
                    llvm::BasicBlock* exit = llvm::BasicBlock::Create(mod->getContext(), "exit", func);
                    build.CreateCondBr(cond, firstNonEntry, exit);
                    build.SetInsertPoint(exit);
                    build.CreateRet(getConstant(pc + 4));
                    return {func, instrCounts};
                }
                else
                {
                    llvm::BasicBlock* taken = llvm::BasicBlock::Create(mod->getContext(), "taken", func);
                    llvm::BasicBlock* ntaken = llvm::BasicBlock::Create(mod->getContext(), "ntaken", func);

                    build.CreateCondBr(cond, taken, ntaken);

                    build.SetInsertPoint(taken);
                    build.CreateRet(getConstant(dst));

                    build.SetInsertPoint(ntaken);
                    build.CreateRet(getConstant(pc + 4));

                    instrCounts[std::array<const char*, 8>{"beq", "bne", "", "", "blt", "bge", "bltu",
                                                           "bgeu"}[funct3]]++;
                    return {func, instrCounts};
                }
            }
            case OPC_REG_IMM:
            {
                uint32_t imm = (instr >> 20) | (uint32_t)((int32_t)(instr & (1 << 31)) >> 20);
                llvm::Value* val;
                llvm::Value* immVal = getConstant(imm);
                llvm::Value* regSrc = loadRegister(rs1);

                switch (funct3)
                {
                    case 0:
                        val = build.CreateAdd(regSrc, immVal);
                        instrCounts["addi"]++;
                        break;

                    case 2:
                        val = build.CreateZExt(build.CreateICmpSLT(regSrc, immVal), i32);
                        instrCounts["slti"]++;
                        break;
                    case 3:
                        val = build.CreateZExt(build.CreateICmpULT(regSrc, immVal), i32);
                        instrCounts["sltiu"]++;
                        break;
                    case 4:
                        val = build.CreateXor(regSrc, immVal);
                        instrCounts["xori"]++;
                        break;

                    case 6:
                        val = build.CreateOr(regSrc, immVal);
                        instrCounts["ori"]++;
                        break;
                    case 7:
                        val = build.CreateAnd(regSrc, immVal);
                        instrCounts["andi"]++;
                        break;

                    case 5:
                    case 1:
                    {
                        if (funct3 == 1)
                        {
                            if (funct7 == 0)
                            {
                                val = build.CreateShl(regSrc, getConstant(rs2));
                                instrCounts["slli"]++;
                            }
                            else
                                goto unimplemented;
                        }
                        else
                        {
                            if (funct7 == 0)
                            {
                                val = build.CreateLShr(regSrc, getConstant(rs2));
                                instrCounts["srli"]++;
                            }
                            else if (funct7 == 0b0100000)
                            {
                                val = build.CreateAShr(regSrc, getConstant(rs2));
                                instrCounts["srai"]++;
                            }
                            else
                                goto unimplemented;
                        }
                        break;
                    }
                    default: __builtin_unreachable();
                }

                storeRegister(rd, val);
                break;
            }
            case OPC_REG_REG:
            {
                llvm::Value* src1 = loadRegister(rs1);
                llvm::Value* src2 = loadRegister(rs2);
                llvm::Value* val;

                if (funct7 == 0)
                    switch (funct3)
                    {
                        case 0:
                            val = build.CreateAdd(src1, src2);
                            instrCounts["add"]++;
                            break;
                        case 1:
                            val = build.CreateShl(src1, src2);
                            instrCounts["sll"]++;
                            break;
                        case 2:
                            val = build.CreateZExt(build.CreateICmpSLT(src1, src2), i32);
                            instrCounts["slt"]++;
                            break;
                        case 3:
                            val = build.CreateZExt(build.CreateICmpULT(src1, src2), i32);
                            instrCounts["sltu"]++;
                            break;
                        case 4:
                            val = build.CreateXor(src1, src2);
                            instrCounts["xor"]++;
                            break;
                        case 5:
                            val = build.CreateLShr(src1, src2);
                            instrCounts["srl"]++;
                            break;
                        case 6:
                            val = build.CreateOr(src1, src2);
                            instrCounts["or"]++;
                            break;
                        case 7:
                            val = build.CreateAnd(src1, src2);
                            instrCounts["and"]++;
                            break;
                        default: __builtin_unreachable();
                    }
                else if (funct7 == 0b0100000)
                    switch (funct3)
                    {
                        case 0:
                            val = build.CreateSub(src1, src2);
                            instrCounts["sub"]++;
                            break;
                        case 5:
                            val = build.CreateAShr(src1, src2);
                            instrCounts["sra"]++;
                            break;
                        default: goto unimplemented;
                    }

                else if (funct7 == 1)
                    switch (funct3)
                    {
                        case 0:
                            val = build.CreateMul(src1, src2);
                            instrCounts["mul"]++;
                            break;
                        case 1:
                            val = build.CreateTrunc(build.CreateMul(build.CreateSExt(src1, i64),
                                                                    build.CreateSExt(src2, i64), "", false, true),
                                                    i32);
                            instrCounts["mulh"]++;
                            break;
                        case 2:
                            val = build.CreateTrunc(build.CreateMul(build.CreateSExt(src1, i64),
                                                                    build.CreateZExt(src2, i64), "", false, true),
                                                    i32);
                            instrCounts["mulhsu"]++;
                            break;
                        case 3:
                            val = build.CreateTrunc(build.CreateMul(build.CreateZExt(src1, i64),
                                                                    build.CreateZExt(src2, i64), "", true, false),
                                                    i32);
                            instrCounts["mulhu"]++;
                            break;
                        case 4:
                            val = build.CreateSDiv(src1, src2);
                            instrCounts["div"]++;
                            break;
                        case 5:
                            val = build.CreateUDiv(src1, src2);
                            instrCounts["divu"]++;
                            break;
                        case 6:
                            val = build.CreateSRem(src1, src2);
                            instrCounts["rem"]++;
                            break;
                        case 7:
                            val = build.CreateURem(src1, src2);
                            instrCounts["remu"]++;
                            break;
                        default: __builtin_unreachable();
                    }
                else
                    goto unimplemented;

                storeRegister(rd, val);
                break;
            }
            case OPC_FENCE:
                if (funct3 != 0)
                    goto unimplemented;
                instrCounts["fence"]++;
                break;
            case OPC_ENV:
            {
                if (rs1 == 0 && rd == 0 && funct3 == 0)
                {
                    uint32_t imm12 = instr >> 20;
                    if (imm12 == 0) // ecall
                    {
                        llvm::FunctionType* ecallFuncType = llvm::FunctionType::get(i32, {ptrT}, false);
                        auto ecallFunc = mod->getOrInsertFunction("RV32I_ECall", ecallFuncType);

                        llvm::Value* retval = build.CreateCall(ecallFunc, {func->getArg(0)});
                        storeRegister(10, retval);
                        instrCounts["ecall"]++;
                        break;
                    }
                    else if (imm12 == 1) // ebreak
                    {
                        build.CreateRet(getConstant(pc + 4));
                        instrCounts["ebreak"]++;
                        return {func, instrCounts};
                    }
                }
                goto unimplemented;
            }

            // No rv32im match, try to find CoreDSL instruction match
            default:
            {
                auto impl = TryMatchExtInstr(instr);
                if (impl == nullptr)
                    goto unimplemented;
                
                int rd = GetValueOfField(instr, *impl->cdsl, CDSLInstr::Field::ID_RD);
                build.CreateStore(loadRegister(rd), tempRD);
                llvm::FunctionType* implFuncType = llvm::FunctionType::get(voidT, {ptrT, ptrT, ptrT, i32, i32}, false);
                auto func = mod->getOrInsertFunction("impl" + impl->cdsl->name, implFuncType);
                build.CreateCall(func, {
                                           tempRD,
                                           getRegister(GetValueOfField(instr, *impl->cdsl, CDSLInstr::Field::ID_RS1)),
                                           getRegister(GetValueOfField(instr, *impl->cdsl, CDSLInstr::Field::ID_RS2)),
                                           getConstant(GetValueOfField(instr, *impl->cdsl, CDSLInstr::Field::ID_IMM0)),
                                           getConstant(GetValueOfField(instr, *impl->cdsl, CDSLInstr::Field::ID_IMM1)),
                                       });
                storeRegister(rd, build.CreateLoad(i32, tempRD));
                instrCounts[impl->cdsl->name.c_str()]++;
                break;
            }

            unimplemented:
            {
                llvm::FunctionType* exceptFuncType = llvm::FunctionType::get(voidT, {ptrT}, false);
                auto ecallFunc = mod->getOrInsertFunction("RV32I_Except", exceptFuncType);
                build.CreateCall(ecallFunc, {func->getArg(0)});
                build.CreateRet(getConstant(pc + 4));
                instrCounts["unimp"]++;
                return {func, instrCounts};
            }
        }

        pc += 4;
    }
}

static void PrintState(uint32_t state[32])
{
    const std::array<const char*, 32> names = {"x0", "ra", "sp", "gp", "tp",  "t0",  "t1", "t2", "s0", "s1", "a0",
                                               "a1", "a2", "a3", "a4", "a5",  "a6",  "a7", "s2", "s3", "s4", "s5",
                                               "s6", "s7", "s8", "s9", "s10", "s11", "t3", "t4", "t5", "t6"};

    for (size_t i = 0; i < 32; i += 4)
    {
        for (size_t j = 0; j < 4; j++)
            fprintf(stderr, "%4s %.8x", names[i + j], state[i + j]);
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "\n");
}

struct BlockDebugInfo
{
    uint32_t pc;
    llvm::SmallVector<std::pair<const char*, uint32_t>, 16> iCounts;
};

static FragFuncT CompileAndRegisterBlock(llvm::LLVMContext& ctx, uint8_t* memory, uint32_t startPC,
                                         llvm::DenseMap<uint32_t, std::pair<FragFuncT, uint64_t>>& blocks,
                                         std::vector<BlockDebugInfo>& debugInfo)
{
    auto module = std::make_unique<llvm::Module>("mod", ctx);
    auto counts = TranslateBlock(module.get(), memory, startPC).second;
    // module->print(llvm::errs(), nullptr);
    std::string fname = module->functions().begin()->getName().str();
    FragFuncT func = reinterpret_cast<FragFuncT>(CompileFunc(std::move(module), fname));

    blocks[startPC] = {func, 1};
    auto iCounts = llvm::SmallVector<std::pair<const char*, uint32_t>, 16>(counts.begin(), counts.end());
    debugInfo.push_back({startPC, iCounts});

    return func;
}

static auto CalculateInstrCounts(llvm::DenseMap<uint32_t, std::pair<FragFuncT, uint64_t>> const& blocks,
                                 std::vector<BlockDebugInfo> const& debugInfo, uint64_t* total)
{
    uint64_t tot = 0;
    llvm::DenseMap<const char*, uint64_t> instrCount;
    for (auto& dblock : debugInfo)
    {
        uint64_t vcount = blocks.find(dblock.pc)->second.second;
        for (auto [name, count] : dblock.iCounts)
        {
            tot += count * vcount;
            instrCount[name] += count * vcount;
        }
    }

    using entry = std::pair<const char*, uint64_t>;
    std::vector<entry> countList(instrCount.begin(), instrCount.end());
    std::sort(countList.begin(), countList.end(), [](entry const& a, entry const& b) { return a.second > b.second; });
    if (total)
        *total = tot;
    return countList;
}

void RunJIT(std::vector<CDSLInstr> const& instrs, std::unique_ptr<llvm::Module> instrImpls, const std::string elfFile)
{
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetDisassembler();
    llvm::InitializeNativeTargetAsmParser();
    llvm::InitializeAllTargets();

    PreprocessInstructions(instrs, std::move(instrImpls));

    uint8_t* memory = static_cast<uint8_t*>(
        mmap(NULL, 0x100000000, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON | MAP_NORESERVE, -1, 0));
    if (memory == (uint8_t*)-1)
    {
        perror("could not map memory: ");
        exit(-1);
    }
    uint32_t pc = 0x80000000;
    auto sections = LoadELF(elfFile, memory, pc);

    llvm::DenseMap<uint32_t, std::pair<FragFuncT, uint64_t>> translatedBlocks;
    std::vector<BlockDebugInfo> debugInfo;

    llvm::LLVMContext ctx;

    RV32State state = {};
    state.regs[2] = 0x80000000;

    doExit = false;
    while (!doExit)
    {
        // 1. Check if basic block at pc has been translated
        auto iter = translatedBlocks.find(pc);
        FragFuncT func;

        // If not, translate
        if (iter == translatedBlocks.end())
            func = CompileAndRegisterBlock(ctx, memory, pc, translatedBlocks, debugInfo);
        else
        {
            iter->second.second++; // Increment visit counter
            func = iter->second.first;
        }

        // 2. Execute Basic Block
        pc = func(&state);
    }

    bool outputInstrCount = true;
    if (outputInstrCount)
    {
        uint64_t total;
        auto counts = CalculateInstrCounts(translatedBlocks, debugInfo, &total);
        for (auto& p : counts)
        {
            std::string name = p.first;
            std::replace(name.begin(), name.end(), '_', '.');
            std::transform(name.begin(), name.end(), name.begin(), &tolower);
            llvm::outs() << name << ": " << p.second << "\n";
        }
        llvm::outs() << "total: " << total << "\n";
    }

    bool dumpBSS = true;
    if (dumpBSS)
    {
        auto bss = std::find_if(sections.begin(), sections.end(),
            [](Section const& s) {return s.name == ".bss";});
        if (bss != sections.end())
            DumpSection(elfFile + "_bss.bin", memory, *bss);
    }
}
