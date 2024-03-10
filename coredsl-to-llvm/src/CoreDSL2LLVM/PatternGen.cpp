#include "PatternGen.hpp"
#include "../lib/InstrInfo.hpp"
#include "LLVMOverride.hpp"
#include "RISCVISelLowering.h"
#include "llvm/CodeGen/ISDOpcodes.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MachineValueType.h"
#include "llvm/Support/raw_ostream.h"
#include <array>
#include <exception>
#include <fstream>
#include <initializer_list>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <type_traits>
#include <unordered_map>
#include <utility>

static std::ostream* outStream = nullptr;
static std::vector<CDSLInstr> const* cdslInstrs;
static CDSLInstr const* curInstr = nullptr;
static std::string* extName = nullptr;

using namespace llvm;
using SVT = llvm::MVT::SimpleValueType;

int GeneratePatterns(llvm::Module* M, std::vector<CDSLInstr> const& instrs, std::ostream& ostream, std::string extName)
{
    // All other code in this file is called during code generation
    // by the LLVM pipeline. We thus "pass" arguments as globals in this TU.
    outStream = &ostream;
    cdslInstrs = &instrs;
    ::extName = &extName;

    int rv = RunPatternGenPipeline(M, extName);

    outStream = nullptr;
    cdslInstrs = nullptr;
    ::extName = nullptr;

    return rv;
}

static const std::unordered_map<ISD::CondCode, std::string> cmpStr = {
    {ISD::SETEQ, "SETEQ"},   {ISD::SETNE, "SETNE"},   {ISD::SETLT, "SETLT"},   {ISD::SETLE, "SETLE"},
    {ISD::SETGT, "SETGT"},   {ISD::SETGE, "SETGE"},   {ISD::SETULT, "SETULT"}, {ISD::SETULE, "SETULE"},
    {ISD::SETUGT, "SETUGT"}, {ISD::SETUGE, "SETUGE"},
};

struct PatternNode
{
    // LLVM is compiled with -fno-rtti, so we use
    // LLVM-style custom RTTI for our nodes.
    enum PatternNodeKind
    {
        PN_NOp,
        PN_Binop,
        PN_Shuffle,
        PN_Compare,
        PN_Unop,
        PN_Constant,
        PN_Register,
        PN_Select,
    };

  private:
    const PatternNodeKind kind;

  public:
    PatternNodeKind GetKind() const
    {
        return kind;
    }
    SVT type;
    PatternNode(PatternNodeKind kind, SVT type) : kind(kind), type(type)
    {
    }

    virtual std::string PatternString(int indent = 0) = 0;
    virtual SVT GetRegisterTy(int operandID) const
    {
        if (operandID == -1)
            return type;
        return SVT::INVALID_SIMPLE_VALUE_TYPE;
    }
    virtual ~PatternNode()
    {
    }
};

struct NOpNode : public PatternNode
{
    ISD::NodeType op;
    std::vector<std::unique_ptr<PatternNode>> operands;
    NOpNode(SVT type, ISD::NodeType op, std::vector<std::unique_ptr<PatternNode>> operands)
        : PatternNode(PN_NOp, type), op(op), operands(std::move(operands))
    {
    }

    std::string PatternString(int indent = 0) override
    {
        static const std::unordered_map<ISD::NodeType, std::string> binopStr = {{ISD::BUILD_VECTOR, "build_vector"},
                                                                                {ISD::VSELECT, "vselect"}};

        std::string s = "(" + std::string(binopStr.at(op)) + " ";
        for (auto& operand : operands)
            s += operand->PatternString(indent + 1) + ", ";
        if (!operands.empty())
            s = s.substr(0, s.size() - 2);

        s += ")";
        return s;
    }
    SVT GetRegisterTy(int operandID) const override
    {
        if (operandID == -1)
            return type;

        for (auto& operand : operands)
        {
            auto t = operand->GetRegisterTy(operandID);
            if (t != MVT::INVALID_SIMPLE_VALUE_TYPE)
                return t;
        }
        return MVT::INVALID_SIMPLE_VALUE_TYPE;
    }
    static bool classof(const PatternNode* p)
    {
        return p->GetKind() == PN_NOp;
    }
};

struct BinopNode : public PatternNode
{
    ISD::NodeType op;
    std::unique_ptr<PatternNode> left;
    std::unique_ptr<PatternNode> right;

    BinopNode(SVT type, ISD::NodeType op, std::unique_ptr<PatternNode> left, std::unique_ptr<PatternNode> right)
        : PatternNode(PN_Binop, type), op(op), left(std::move(left)), right(std::move(right))
    {
    }

    std::string PatternString(int indent = 0) override
    {
        static const std::unordered_map<ISD::NodeType, std::string> binopStr = {
            {ISD::ADD, "add"},   {ISD::SUB, "sub"},   {ISD::MUL, "mul"},
            {ISD::SDIV, "div"},  {ISD::AND, "and"},   {ISD::OR, "or"},
            {ISD::XOR, "xor"},   {ISD::SHL, "shl"},   {ISD::SRL, "srl"},
            {ISD::SRA, "sra"},   {ISD::SMAX, "smax"}, {ISD::UMAX, "umax"},
            {ISD::SMIN, "smin"}, {ISD::UMIN, "umin"}, {ISD::EXTRACT_VECTOR_ELT, "vector_extract"},
            {ISD::ROTL, "rotl"}, {ISD::ROTR, "rotr"}};

        std::string typeStr = EVT(type).getEVTString();
        std::string opString = "(" + std::string(binopStr.at(op)) + " " + left->PatternString(indent + 1) + ", " +
                               right->PatternString(indent + 1) + ")";

        // Explicitly specifying types for all ops increases pattern compile time
        // significantly, so we only do for ops where deduction fails otherwise.
        bool printType = false;
        switch (op)
        {
            case ISD::SHL:
            case ISD::SRL:
            case ISD::SRA: printType = true; break;
            default: break;
        }

        if (printType)
            return "(" + typeStr + " " + opString + ")";
        else
            return opString;
    }

    SVT GetRegisterTy(int operandID) const override
    {
        if (operandID == -1)
            return type;

        auto leftT = left->GetRegisterTy(operandID);
        return leftT != MVT::INVALID_SIMPLE_VALUE_TYPE ? leftT : right->GetRegisterTy(operandID);
    }

    static bool classof(const PatternNode* p)
    {
        return p->GetKind() == PN_Binop;
    }
};

struct CompareNode : public BinopNode
{
    ISD::CondCode cond;

    CompareNode(SVT type, ISD::CondCode cond, std::unique_ptr<PatternNode> left, std::unique_ptr<PatternNode> right)
        : BinopNode(type, ISD::SETCC, std::move(left), std::move(right)), cond(cond)
    {
    }

    std::string PatternString(int indent = 0) override
    {
        std::string typeStr = EVT(type).getEVTString();

        return "(" + typeStr + " (setcc " + left->PatternString(indent + 1) + ", " + right->PatternString(indent + 1) +
               ", " + cmpStr.at(cond) + "))";
    }
};

struct SelectNode : public PatternNode
{
    ISD::CondCode cond;
    std::unique_ptr<PatternNode> left;
    std::unique_ptr<PatternNode> right;
    std::unique_ptr<PatternNode> tval;
    std::unique_ptr<PatternNode> fval;

    SelectNode(SVT type, ISD::CondCode cond, std::unique_ptr<PatternNode> left, std::unique_ptr<PatternNode> right,
               std::unique_ptr<PatternNode> tval, std::unique_ptr<PatternNode> fval)
        : PatternNode(PN_Select, type), cond(cond), left(std::move(left)), right(std::move(right)),
          tval(std::move(tval)), fval(std::move(fval))
    {
    }

    std::string PatternString(int indent = 0) override
    {
        std::string typeStr = EVT(type).getEVTString();

        return "(" + typeStr + " (riscv_selectcc " + left->PatternString(indent + 1) + ", " +
               right->PatternString(indent + 1) + ", " + cmpStr.at(cond) + ", " + tval->PatternString(indent + 1) +
               ", " + fval->PatternString(indent + 1) + "))";
    }

    SVT GetRegisterTy(int operandID) const override
    {
        if (operandID == -1)
            return type;

        for (auto operand : {&left, &right, &tval, &fval})
        {
            auto t = (*operand)->GetRegisterTy(operandID);
            if (t != MVT::INVALID_SIMPLE_VALUE_TYPE)
                return t;
        }
        return MVT::INVALID_SIMPLE_VALUE_TYPE;
    }

    static bool classof(const PatternNode* p)
    {
        return p->GetKind() == PN_Select;
    }
};

struct UnopNode : public PatternNode
{
    ISD::NodeType op;
    std::unique_ptr<PatternNode> operand;

    UnopNode(SVT type, ISD::NodeType op, std::unique_ptr<PatternNode> operand)
        : PatternNode(PN_Unop, type), op(op), operand(std::move(operand))
    {
    }

    std::string PatternString(int indent = 0) override
    {
        // clang-format off
        static const std::unordered_map<ISD::NodeType, std::string> unopStr = {
            {ISD::SIGN_EXTEND, "sext"},     {ISD::ZERO_EXTEND, "zext"},
            {ISD::VECREDUCE_ADD, "vecreduce_add"}, {ISD::TRUNCATE, "trunc"},
            {ISD::SPLAT_VECTOR, "splat_vector"},   {ISD::BITCAST, "bitcast"},
            {ISD::SIGN_EXTEND_INREG, "sext"}, {ISD::ABS, "abs"}
        };
        // clang-format on

        std::string typeStr = EVT(type).getEVTString();

        // ignore bitcast ops for now
        if (op == ISD::BITCAST)
            return operand->PatternString(indent);

        return "(" + typeStr + " (" + std::string(unopStr.at(op)) + " " + operand->PatternString(indent + 1) + "))";
    }

    SVT GetRegisterTy(int operandID) const override
    {
        if (operandID == -1 && op != ISD::BITCAST)
            return type;
        return operand->GetRegisterTy(operandID);
    }

    static bool classof(const PatternNode* p)
    {
        return p->GetKind() == PN_Unop;
    }
};

struct ConstantNode : public PatternNode
{
    uint32_t constant;
    ConstantNode(SVT type, uint32_t c) : PatternNode(PN_Constant, type), constant(c)
    {
    }

    std::string PatternString(int indent = 0) override
    {
        return "(i32 " + std::to_string(constant) + ")";
    }

    static bool classof(const PatternNode* p)
    {
        return p->GetKind() == PN_Constant;
    }
};

struct RegisterNode : public PatternNode
{
    int regId;
    int offset; // in bytes
    int size;   // 0 byte, 1 half, 2 word
    ISD::LoadExtType ext;

    RegisterNode(SVT type, int regId, int offset, int size, ISD::LoadExtType ext = ISD::LoadExtType::NON_EXTLOAD)
        : PatternNode(PN_Register, type), regId(regId), offset(offset), size(size), ext(ext)
    {
    }

    std::string PatternString(int indent = 0) override
    {
        static const std::string regNames[] = {"rd", "rs1", "rs2", "imm", "imm2"};

        // Immediate Operands
        if (regId >= 3)
        {
            assert(size == -1 && offset == 0);
            return std::string("(i32 ") + (curInstr->SignedImm(regId - 3) ? "simm" : "uimm") +
                   std::to_string(curInstr->GetImmLen(regId - 3)) + ":$" + regNames[regId] + ")";
        }

        // Full-Size Register Operands
        if (size == 2)
        {
            if (type == MVT::i32)
                return "GPR:$" + regNames[regId];
            if (type == MVT::v4i8)
                return "PulpV4:$" + regNames[regId];
            if (type == MVT::v2i16)
                return "PulpV2:$" + regNames[regId];
            abort();
        }

        // Sub-Register Operands
        if (size == 1 || size == 0)
        {
            std::string str;
            if (type == llvm::MVT::i32)
            {
                assert(offset == 0);
                str = "GPR:$" + regNames[regId];
            }
            else
                str = std::string("(i32 (vector_extract PulpV") + (size ? "2" : "4") + ":$" + regNames[regId] + ", " +
                      std::to_string(size ? (offset / 2) : (offset)) + "))";

            std::string mask = size ? "65535" : "255";
            std::string shamt = size ? "16" : "24";
            switch (ext)
            {
                case ISD::LoadExtType::EXTLOAD: return str;
                case ISD::LoadExtType::ZEXTLOAD: return "(and " + str + ", (i32 " + mask + "))";
                case ISD::LoadExtType::SEXTLOAD:
                    return "(sra (shl " + str + ", (i32 " + shamt + ")), (i32 " + shamt + "))";
                default: break;
            }
        }
        abort();
    }

    SVT GetRegisterTy(int operandID) const override
    {
        if (operandID == -1)
            return type;
        if (operandID == regId)
            return type;
        return SVT::INVALID_SIMPLE_VALUE_TYPE;
    }

    static bool classof(const PatternNode* p)
    {
        return p->GetKind() == PN_Register;
    }
};

std::unique_ptr<PatternNode> NodeToPattern(SDNode* node)
{
    SVT type = node->getSimpleValueType(0).SimpleTy;

    switch (node->getOpcode())
    {
        case ISD::Constant:
        {
            return std::make_unique<ConstantNode>(type, (llvm::cast<llvm::ConstantSDNode>(node))->getLimitedValue());
        }
        case ISD::ADD:
        case ISD::SUB:
        case ISD::MUL:
        case ISD::AND:
        case ISD::OR:
        case ISD::XOR:
        case ISD::SHL:
        case ISD::SRL:
        case ISD::SRA:
        case ISD::SMAX:
        case ISD::UMAX:
        case ISD::SMIN:
        case ISD::UMIN:
        case ISD::ROTL:
        case ISD::ROTR:
        case ISD::EXTRACT_VECTOR_ELT:
            return std::make_unique<BinopNode>(type, (ISD::NodeType)node->getOpcode(),
                                               NodeToPattern(node->getOperand(0).getNode()),
                                               NodeToPattern(node->getOperand(1).getNode()));

        case ISD::SETCC:
        {
            auto condCode = llvm::cast<llvm::CondCodeSDNode>(node->getOperand(2));
            return std::make_unique<CompareNode>(type, condCode->get(), NodeToPattern(node->getOperand(0).getNode()),
                                                 NodeToPattern(node->getOperand(1).getNode()));
        }

        case ISD::VSELECT:
        {
            std::vector<std::unique_ptr<PatternNode>> operands(3);
            for (int i = 0; i < 3; i++)
                operands[i] = NodeToPattern(node->getOperand(i).getNode());
            return std::make_unique<NOpNode>(type, (ISD::NodeType)node->getOpcode(), std::move(operands));
        }

        case ISD::CopyFromReg:
        {
            auto& cfr = *node;
            auto& reg = cfr.getOperand(1);
            if (reg.getOpcode() != ISD::Register)
                goto match_failure;
            auto idx = llvm::cast<llvm::RegisterSDNode>(reg)->getReg().id() - (1U << 31U);
            if (idx == 3 || idx == 4)
                return std::make_unique<RegisterNode>(type, idx, 0, -1);
        }

        case ISD::BITCAST:
        {
            if (node->getOperand(0).getNode()->getOpcode() != ISD::LOAD)
                goto unop;
            // Bitcasts are transparent for loads
            // (bitcasts are already handled in hard-coded patterns)
            type = node->getSimpleValueType(0).SimpleTy;
            node = node->getOperand(0).getNode();
        }
            [[fallthrough]];
        case ISD::LOAD:
        {
            MemSDNode* mem = llvm::cast<MemSDNode>(node);
            LoadSDNode* ld = llvm::cast<LoadSDNode>(node);
            SVT memType = mem->getMemoryVT().getSimpleVT().SimpleTy;
            switch (memType)
            {
                case MVT::i8:
                case MVT::i16:
                {
                    int offset = 0;
                    auto& add = node->getOperand(1);
                    const SDValue* cfr;
                    if (add->getOpcode() == ISD::ADD)
                    {
                        auto& offs = add->getOperand(1);
                        if (offs->getOpcode() != ISD::Constant)
                            goto variable_offset;
                        auto offsConst = llvm::cast<llvm::ConstantSDNode>(offs);
                        offset = offsConst->getLimitedValue();
                        cfr = &add->getOperand(0);
                    }
                    else if (add->getOpcode() == ISD::CopyFromReg)
                        cfr = &add;
                    else
                        goto match_failure;

                    if (memType == MVT::i16 && (offset != 0 && offset != 2))
                        goto match_failure;

                    auto& reg = cfr->getOperand(1);
                    if (reg.getOpcode() != ISD::Register)
                        goto match_failure;

                    auto regReg = llvm::cast<llvm::RegisterSDNode>(reg);
                    unsigned regID = regReg->getReg().id() - (1U << 31U);

                    // We generate accesses to i8 or i16 elements as vector_extract
                    // of v4i8 or v2i16 registers respectively.
                    type = (memType == MVT::i8 ? MVT::v4i8 : MVT::v2i16);

                    return std::make_unique<RegisterNode>(type, regID, offset, (memType == MVT::i8 ? 0 : 1),
                                                          ld->getExtensionType());
                }

                case MVT::i32:
                case MVT::v4i8:
                case MVT::v2i16:
                {
                    auto& cfr = node->getOperand(1);
                    if (cfr->getOpcode() != ISD::CopyFromReg)
                        goto match_failure;
                    auto& reg = cfr->getOperand(1);
                    if (reg.getOpcode() != ISD::Register)
                        goto match_failure;

                    auto regReg = llvm::cast<llvm::RegisterSDNode>(reg);
                    unsigned regID = regReg->getReg().id() - (1U << 31U);
                    return std::make_unique<RegisterNode>(type, regID, 0, 2);
                }

                variable_offset:
                {
                    auto& add = node->getOperand(1);
                    auto& offs = add->getOperand(1);
                    if (offs->getOpcode() != ISD::AND)
                        goto match_failure;
                    if (offs->getOperand(1).getOpcode() != ISD::Constant ||
                        llvm::cast<ConstantSDNode>(offs->getOperand(1))->getLimitedValue() != 3)
                        goto match_failure;
                }
                default: goto match_failure;
            }
            break;
        }

        case RISCVISD::SELECT_CC:
        {
            auto condCode = llvm::cast<llvm::CondCodeSDNode>(node->getOperand(2));
            return std::make_unique<SelectNode>(type, condCode->get(), NodeToPattern(node->getOperand(0).getNode()),
                                                NodeToPattern(node->getOperand(1).getNode()),
                                                NodeToPattern(node->getOperand(3).getNode()),
                                                NodeToPattern(node->getOperand(4).getNode()));
        }

        case ISD::TRUNCATE:
        case ISD::SIGN_EXTEND_INREG: // type we sExt from is node->getOperand(0).getValueType().getSimpleVT().SimpleTy
        case ISD::ZERO_EXTEND:
        case ISD::SIGN_EXTEND:
        case ISD::VECREDUCE_ADD:
        case ISD::SPLAT_VECTOR:
        case ISD::ABS:
        unop:
            return std::make_unique<UnopNode>(type, (ISD::NodeType)node->getOpcode(),
                                              NodeToPattern(node->getOperand(0).getNode()));

        case ISD::BUILD_VECTOR:
        {
            std::vector<std::unique_ptr<PatternNode>> operands(node->getNumOperands());
            for (size_t i = 0; i < node->getNumOperands(); i++)
                operands[i] = NodeToPattern(node->getOperand(i).getNode());
            return std::make_unique<NOpNode>(type, (ISD::NodeType)node->getOpcode(), std::move(operands));
        }

        default: goto match_failure;
    }
match_failure:
    throw std::invalid_argument("match failure: " + node->getOperationName());
}

// Convert vector-operand nodes which exclusively access element 0 of a vector register
// to scalar operand nodes.
void TryConvertVectorRegToScalar(std::unique_ptr<PatternNode>& pattern, int regId)
{
    SmallVector<PatternNode*, 64> stack = {pattern.get()};
    SmallVector<RegisterNode*> toConvert;

    while (!stack.empty())
    {
        PatternNode* cur = stack.back();
        stack.pop_back();

        switch (cur->GetKind())
        {
            case PatternNode::PN_Register:
            {
                RegisterNode* reg = llvm::cast<RegisterNode>(cur);
                if (reg->regId != regId)
                    break;
                if (reg->offset != 0)
                    return;
                if (reg->type != MVT::v4i8 && reg->type != MVT::v2i16)
                    return;
                if (reg->size != 0 && reg->size != 1)
                    return;
                toConvert.push_back(reg);
                break;
            }
            case PatternNode::PN_NOp:
            {
                NOpNode* nop = llvm::cast<NOpNode>(cur);
                for (auto& op : nop->operands)
                    stack.push_back(op.get());
                break;
            }
            case PatternNode::PN_Unop:
            {
                UnopNode* nop = llvm::cast<UnopNode>(cur);
                stack.push_back(nop->operand.get());
                break;
            }
            case PatternNode::PN_Binop:
            {
                BinopNode* bop = llvm::cast<BinopNode>(cur);
                stack.push_back(bop->left.get());
                stack.push_back(bop->right.get());
                break;
            }
            default: break;
        }
    }

    for (auto n : toConvert)
        n->type = llvm::MVT::i32;
}

void PrintPattern(SelectionDAG& DAG)
{
    std::string instName = DAG.getMachineFunction().getName().str().substr(4);
    std::string instNameO = instName;

    if (DAG.getMachineFunction().size() != 1)
        llvm::errs() << "Match Failure for " + instName + ": multiple basic blocks\n";

    auto& root = DAG.getRoot();
    if (root->getNumOperands() != 1)
        return;
    auto& store = root->getOperand(0);
    if (store.getOpcode() != ISD::STORE || store.getOperand(1).getSimpleValueType().getSizeInBits() != 32 ||
        llvm::cast<StoreSDNode>(store)->isTruncatingStore())
    {
        llvm::errs() << "Match Failure for " + instName + ": does not end in 32-bit store\n";
        return;
    }

    {
        // Just do a linear search to find the instruction
        auto const& it = std::find_if(cdslInstrs->begin(), cdslInstrs->end(),
                                      [&instNameO](CDSLInstr const& c) { return c.name == instNameO; });
        assert(it != cdslInstrs->end() && "instruction should be defined");
        curInstr = it.base();
    }

    std::unique_ptr<PatternNode> pattern;
    try
    {
        pattern = NodeToPattern(store.getOperand(1).getNode());
    }
    catch (std::invalid_argument e)
    {
        llvm::errs() << "Match Failure for " + instName + ": " + e.what() << "\n";
        return;
    }

    for (int i = 0; i < 3; i++)
        TryConvertVectorRegToScalar(pattern, i);

    instName += "_";
    std::array<std::string, 6> typeStrings = {"", "", "",
                                              "", "", ""}; // types for: rd, rd (as src), rs1, rs2, imm, imm2
    for (int i = 0; i < 4; i++)
    {
        auto type = pattern->GetRegisterTy(i - 1);
        switch (type)
        {
            case MVT::i8:
            case MVT::i16:
            case MVT::i32:
                typeStrings[i] = "GPR";
                instName += "_S";
                break;
            case MVT::v4i8:
                typeStrings[i] = "PulpV4";
                instName += "_V4";
                break;
            case MVT::v2i16:
                typeStrings[i] = "PulpV2";
                instName += "_V2";
                break;
            default:;
        }
    }
    for (int i = 0; i < 2; i++)
        if (pattern->GetRegisterTy(3 + i) != MVT::INVALID_SIMPLE_VALUE_TYPE)
            typeStrings[4 + i] = std::string(curInstr->SignedImm(i) ? "simm" : "uimm") + std::to_string(curInstr->GetImmLen(i));

    static const auto opNames = std::array<std::string, 5>{"rd", "rs1", "rs2", "imm", "imm2"};

    std::string outsString = typeStrings[0] + (typeStrings[1].empty() ? ":$rd" : ":$rd_wb");
    
    std::string insString;
    for (int i = 1; i < 6; i++)
        if (!typeStrings[i].empty())
        {
            insString += typeStrings[i] + ":$" + (opNames[i - 1]) + ", ";
        }
    insString = insString.substr(0, insString.size() - 2);

    *outStream << "let Predicates = [HasExt" + *extName + "], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1";
    if (!typeStrings[1].empty())
        *outStream << ", Constraints = \"$rd = $rd_wb\"";
    *outStream << " in ";
    *outStream << "def " << instName << " : RVInst_" << instNameO << "<(outs " << outsString << "), (ins " << insString
               << ")>;\n";

    std::string patternStr = pattern->PatternString();
    std::string code = "def : Pat<\n\t" + patternStr + ",\n\t(" + instName + " ";

    code += insString;
    code += ")>;";
    *outStream << "\n" << code << "\n\n";

    llvm::errs() << "Successfully matched " + instName + "\n";
}
