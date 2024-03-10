/*
This file contains some code duplicated or slightly adjusted from LLVM,
mostly overriding specific virtual functions to inject our own code.

The alternative to using a file like this is modifying LLVM source
more aggressively directly.
*/
#include "PatternGen.hpp"
#include "RISCVISelDAGToDAG.h"
#include "RISCVMacroFusion.h"
#include "RISCVTargetMachine.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/CodeGen/FunctionLoweringInfo.h"
#include "llvm/CodeGen/GlobalISel/IRTranslator.h"
#include "llvm/CodeGen/GlobalISel/InstructionSelect.h"
#include "llvm/CodeGen/GlobalISel/Legalizer.h"
#include "llvm/CodeGen/GlobalISel/RegBankSelect.h"
#include "llvm/CodeGen/ISDOpcodes.h"
#include "llvm/CodeGen/LinkAllAsmWriterComponents.h"
#include "llvm/CodeGen/LinkAllCodegenComponents.h"
#include "llvm/CodeGen/MIRParser/MIRParser.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Pass.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/Timer.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include <cctype>
#define DEBUG_TYPE "isel"

using namespace llvm;
static codegen::RegisterCodeGenFlags CGF;

class RISCVDAGToPatterns : public RISCVDAGToDAGISel
{
  public:
    RISCVDAGToPatterns(RISCVTargetMachine& TM, CodeGenOpt::Level OptLevel) : RISCVDAGToDAGISel(TM, OptLevel)
    {
    }
    void PreprocessISelDAG()
    {
        RISCVDAGToDAGISel::PreprocessISelDAG();
        PrintPattern(*CurDAG);
        CurDAG->clear();
    }
};

/// addPassesToX helper drives creation and initialization of TargetPassConfig.
static TargetPassConfig* addPassesToGenerateCode(LLVMTargetMachine& TM, PassManagerBase& PM, bool DisableVerify,
                                                 MachineModuleInfoWrapperPass& MMIWP)
{
    // Targets may override createPassConfig to provide a target-specific
    // subclass.
    TargetPassConfig* PassConfig = TM.createPassConfig(PM);
    // Set PassConfig options provided by TargetMachine.
    PassConfig->setDisableVerify(DisableVerify);
    PM.add(PassConfig);
    PM.add(&MMIWP);

    if (PassConfig->addISelPasses())
        return nullptr;
    PassConfig->addMachinePasses();
    PassConfig->setInitialized();
    return PassConfig;
}

namespace
{
class RISCVPatternPassConfig : public TargetPassConfig
{
  public:
    RISCVPatternPassConfig(RISCVTargetMachine& TM, PassManagerBase& PM) : TargetPassConfig(TM, PM)
    {
    }

    RISCVTargetMachine& getRISCVTargetMachine() const
    {
        return getTM<RISCVTargetMachine>();
    }

    ScheduleDAGInstrs* createMachineScheduler(MachineSchedContext* C) const override
    {
        const RISCVSubtarget& ST = C->MF->getSubtarget<RISCVSubtarget>();
        if (ST.hasMacroFusion())
        {
            ScheduleDAGMILive* DAG = createGenericSchedLive(C);
            DAG->addMutation(createRISCVMacroFusionDAGMutation());
            return DAG;
        }
        return nullptr;
    }

    ScheduleDAGInstrs* createPostMachineScheduler(MachineSchedContext* C) const override
    {
        const RISCVSubtarget& ST = C->MF->getSubtarget<RISCVSubtarget>();
        if (ST.hasMacroFusion())
        {
            ScheduleDAGMI* DAG = createGenericSchedPostRA(C);
            DAG->addMutation(createRISCVMacroFusionDAGMutation());
            return DAG;
        }
        return nullptr;
    }

    void addIRPasses() override;
    bool addPreISel() override;
    bool addInstSelector() override;
    bool addIRTranslator() override;
    bool addLegalizeMachineIR() override;
    bool addRegBankSelect() override;
    bool addGlobalInstructionSelect() override;
    void addPreEmitPass() override;
    void addPreEmitPass2() override;
    void addPreSched2() override;
    void addMachineSSAOptimization() override;
    void addPreRegAlloc() override;
    void addPostRegAlloc() override;

  private:
    bool EnableGlobalMerge = true;
    bool EnableRedundantCopyElimination = true;
    bool EnableMachineCombiner = true;
};
} // namespace

void RISCVPatternPassConfig::addIRPasses()
{
    addPass(createAtomicExpandPass());

    if (getOptLevel() != CodeGenOpt::None)
        addPass(createRISCVGatherScatterLoweringPass());

    if (getOptLevel() != CodeGenOpt::None)
        addPass(createRISCVCodeGenPreparePass());

    TargetPassConfig::addIRPasses();
}

bool RISCVPatternPassConfig::addPreISel()
{
    if (TM->getOptLevel() != CodeGenOpt::None)
    {
        // Add a barrier before instruction selection so that we will not get
        // deleted block address after enabling default outlining. See D99707 for
        // more details.
        addPass(createBarrierNoopPass());
        addPass(createHardwareLoopsPass());
    }

    if (EnableGlobalMerge)
    {
        addPass(createGlobalMergePass(TM, /* MaxOffset */ 2047,
                                      /* OnlyOptimizeForSize */ false,
                                      /* MergeExternalByDefault */ true));
    }

    return false;
}

FunctionPass* createRISCVPatternsISelDag(RISCVTargetMachine& TM, CodeGenOpt::Level OptLevel)
{
    return new RISCVDAGToPatterns(TM, OptLevel);
}

bool RISCVPatternPassConfig::addInstSelector()
{
    addPass(createRISCVPatternsISelDag(getRISCVTargetMachine(), getOptLevel()));

    return false;
}

bool RISCVPatternPassConfig::addIRTranslator()
{
    addPass(new IRTranslator(getOptLevel()));
    return false;
}

bool RISCVPatternPassConfig::addLegalizeMachineIR()
{
    addPass(new Legalizer());
    return false;
}

bool RISCVPatternPassConfig::addRegBankSelect()
{
    addPass(new RegBankSelect());
    return false;
}

bool RISCVPatternPassConfig::addGlobalInstructionSelect()
{
    addPass(new InstructionSelect(getOptLevel()));
    return false;
}

void RISCVPatternPassConfig::addPreSched2()
{
}

void RISCVPatternPassConfig::addPreEmitPass()
{
    addPass(&BranchRelaxationPassID);
    addPass(createRISCVMakeCompressibleOptPass());
}

void RISCVPatternPassConfig::addPreEmitPass2()
{
    addPass(createRISCVExpandPseudoPass());
    // Schedule the expansion of AMOs at the last possible moment, avoiding the
    // possibility for other passes to break the requirements for forward
    // progress in the LR/SC block.
    addPass(createRISCVExpandAtomicPseudoPass());
}

void RISCVPatternPassConfig::addMachineSSAOptimization()
{
    TargetPassConfig::addMachineSSAOptimization();
    if (EnableMachineCombiner)
        addPass(&MachineCombinerID);

    if (TM->getTargetTriple().getArch() == Triple::riscv64)
        addPass(createRISCVSExtWRemovalPass());
}

void RISCVPatternPassConfig::addPreRegAlloc()
{
    addPass(createRISCVPreRAExpandPseudoPass());
    if (TM->getOptLevel() != CodeGenOpt::None)
        addPass(createRISCVMergeBaseOffsetOptPass());
    addPass(createRISCVInsertVSETVLIPass());
}

void RISCVPatternPassConfig::addPostRegAlloc()
{
    if (TM->getOptLevel() != CodeGenOpt::None && EnableRedundantCopyElimination)
        addPass(createRISCVRedundantCopyEliminationPass());
}

class RISCVPatternTargetMachine : public RISCVTargetMachine
{
  public:
    bool addPassesToEmitFile(PassManagerBase& PM, raw_pwrite_stream& Out, raw_pwrite_stream* DwoOut,
                             CodeGenFileType FileType, bool DisableVerify, MachineModuleInfoWrapperPass* MMIWP) override
    {
        // Add common CodeGen passes.
        if (!MMIWP)
            MMIWP = new MachineModuleInfoWrapperPass(this);
        TargetPassConfig* PassConfig = addPassesToGenerateCode(*this, PM, DisableVerify, *MMIWP);
        if (!PassConfig)
            return true;

        if (TargetPassConfig::willCompleteCodeGenPipeline())
        {
            if (addAsmPrinter(PM, Out, DwoOut, FileType, MMIWP->getMMI().getContext()))
                return true;
        }
        else
        {
            // MIR printing is redundant with -filetype=null.
            if (FileType != CGFT_Null)
                PM.add(createPrintMIRPass(Out));
        }

        PM.add(createFreeMachineFunctionPass());
        return false;
    }

    TargetPassConfig* createPassConfig(PassManagerBase& PM) override
    {
        return new RISCVPatternPassConfig(*this, PM);
    }

  public:
    RISCVPatternTargetMachine(const Target& T, const Triple& TT, StringRef CPU, StringRef FS,
                              const TargetOptions& Options, std::optional<Reloc::Model> RM, std::optional<CodeModel::Model> CM,
                              CodeGenOpt::Level OL, bool JIT)
        : RISCVTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL, JIT)
    {
    }
};

namespace {

void addOptPasses(
  llvm::legacy::PassManagerBase &passes,
  llvm::legacy::FunctionPassManager &fnPasses,
  llvm::TargetMachine *machine
) {
  llvm::PassManagerBuilder builder;
  builder.OptLevel = 3;
  builder.SizeLevel = 0;
  builder.Inliner = llvm::createFunctionInliningPass(3, 0, false);
  builder.LoopVectorize = true;
  builder.SLPVectorize = true;

  builder.populateFunctionPassManager(fnPasses);
  builder.populateModulePassManager(passes);
}

void addLinkPasses(llvm::legacy::PassManagerBase &passes) {
  llvm::PassManagerBuilder builder;
  builder.VerifyInput = true;
  builder.Inliner = llvm::createFunctionInliningPass(3, 0, false);
}

}

// https://stackoverflow.com/questions/53738883/run-default-optimization-pipeline-using-modern-llvm
void optimizeModule(llvm::TargetMachine* machine, llvm::Module* module)
{
    module->setTargetTriple(machine->getTargetTriple().str());
    module->setDataLayout(machine->createDataLayout());

    llvm::legacy::PassManager passes;
    passes.add(new llvm::TargetLibraryInfoWrapperPass(machine->getTargetTriple()));
    passes.add(llvm::createTargetTransformInfoWrapperPass(machine->getTargetIRAnalysis()));

    llvm::legacy::FunctionPassManager fnPasses(module);
    fnPasses.add(llvm::createTargetTransformInfoWrapperPass(machine->getTargetIRAnalysis()));

    addOptPasses(passes, fnPasses, machine);
    addLinkPasses(passes);

    fnPasses.doInitialization();
    for (llvm::Function& func : *module)
    {
        fnPasses.run(func);
    }
    fnPasses.doFinalization();

    passes.add(llvm::createVerifierPass());
    passes.run(*module);
}

static void set_options()
{
    const char* args[] = {"", "--slp-threshold=-3"};
    cl::ParseCommandLineOptions(sizeof(args) / sizeof(args[0]), args);
}

// Adapted from LLVM llc
int RunPatternGenPipeline(llvm::Module* M, std::string extName)
{
    set_options();

    InitializeAllTargets();
    InitializeAllTargetMCs();
    InitializeAllAsmPrinters();
    InitializeAllAsmParsers();

    PassRegistry* Registry = PassRegistry::getPassRegistry();
    initializeCore(*Registry);
    initializeCodeGen(*Registry);
    initializeLoopStrengthReducePass(*Registry);
    initializeLowerIntrinsicsPass(*Registry);
    initializeUnreachableBlockElimLegacyPassPass(*Registry);
    initializeConstantHoistingLegacyPassPass(*Registry);
    initializeScalarOpts(*Registry);
    initializeVectorization(*Registry);
    initializeScalarizeMaskedMemIntrinLegacyPassPass(*Registry);
    initializeExpandReductionsPass(*Registry);
    initializeExpandVectorPredicationPass(*Registry);
    initializeHardwareLoopsPass(*Registry);
    initializeTransformUtils(*Registry);
    initializeReplaceWithVeclibLegacyPass(*Registry);
    initializeTLSVariableHoistLegacyPassPass(*Registry);

    // Load the module to be compiled...
    // SMDiagnostic Err;
    Triple TheTriple("riscv32", "unknown", "linux", "gnu");
    std::transform(extName.begin(), extName.end(), extName.begin(), tolower);
    codegen::InitTargetOptionsFromCodeGenFlags(TheTriple);
    std::string CPUStr = codegen::getCPUStr(),
                FeaturesStr = codegen::getFeaturesStr() + "+m,+unaligned-scalar-mem,+" + extName;

    TargetOptions Options;
    Options = codegen::InitTargetOptionsFromCodeGenFlags(TheTriple);

    std::optional<Reloc::Model> RM = codegen::getExplicitRelocModel();
    std::optional<CodeModel::Model> CM = codegen::getExplicitCodeModel();

    M->setTargetTriple("riscv32-unknown-linux-gnu");

    std::string error;
    const class Target* TheTarget = TargetRegistry::lookupTarget(codegen::getMArch(), TheTriple, error);

    TargetMachine* Target = new RISCVPatternTargetMachine(*TheTarget, TheTriple, CPUStr, FeaturesStr, Options, RM, CM,
                                                          llvm::CodeGenOpt::Aggressive, false);
    // TheTarget->createTargetMachine(TheTriple.getTriple(), CPUStr, FeaturesStr, Options, RM, CM,
    //                                llvm::CodeGenOpt::Aggressive);
    // llvm::DebugFlag = true;
    M->setDataLayout(Target->createDataLayout().getStringRepresentation());
    optimizeModule(Target, M);
    llvm::outs() << *M << "\n";
    // llvm::DebugFlag = false;

    static_assert(sizeof(RISCVTargetMachine) == sizeof(RISCVPatternTargetMachine));
    Target = static_cast<RISCVPatternTargetMachine*>(Target);

    legacy::PassManager PM;
    TargetLibraryInfoImpl TLII(Triple(M->getTargetTriple()));
    PM.add(new TargetLibraryInfoWrapperPass(TLII));

    codegen::setFunctionAttributes(CPUStr, FeaturesStr, *M);

    LLVMTargetMachine& LLVMTM = static_cast<LLVMTargetMachine&>(*Target);
    MachineModuleInfoWrapperPass* MMIWP = new MachineModuleInfoWrapperPass(&LLVMTM);

    bool NoVerify = false;

    static llvm::raw_null_ostream nullStream{};
    if (Target->addPassesToEmitFile(PM, nullStream, nullptr, codegen::getFileType(), NoVerify, MMIWP))
    {
        return -1;
    }

    const_cast<TargetLoweringObjectFile*>(LLVMTM.getObjFileLowering())
        ->Initialize(MMIWP->getMMI().getContext(), *Target);

    PM.run(*M);

    return 0;
}
