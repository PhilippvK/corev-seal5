#pragma once
#include <vector>
#include <memory>
#include "../lib/InstrInfo.hpp"
#include <llvm/IR/Module.h>

void RunJIT(std::vector<CDSLInstr> const& instrs, std::unique_ptr<llvm::Module> instrImpls, const std::string elfFile);
