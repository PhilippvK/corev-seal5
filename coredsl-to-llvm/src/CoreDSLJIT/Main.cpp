#include <exception>

#include <llvm/IR/LLVMContext.h>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <map>
#include <ctype.h>
#include <cstdio>
#include <fstream>
#include <tuple>
#include <memory>

#include "../lib/InstrInfo.hpp"
#include "../lib/Parser.hpp"
#include "../lib/TokenStream.hpp"
#include "../lib/Token.hpp"
#include "../lib/Lexer.hpp"
#include "JIT.hpp"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/InitLLVM.h"

int main (int argc, char** argv)
{
    if (argc <= 2)
    {
        fprintf(stderr, "usage: %s <CORE DESC> <ELF FILE>\n", argv[0]);
        return -1;
    }

    const char* srcPath = argv[1];

    TokenStream ts(srcPath);
    llvm::LLVMContext ctx;
    auto mod = std::make_unique<llvm::Module>("mod", ctx);
    auto instrs = ParseCoreDSL2(ts, mod.get());
    
    RunJIT(instrs, std::move(mod), argv[2]);
}
