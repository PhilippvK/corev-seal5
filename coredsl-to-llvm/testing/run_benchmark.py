from ast import Dict, List
from collections import defaultdict
from math import sqrt
import operator
import subprocess
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

CC = "../out/bin/clang"
CFLAGS = ["--target=riscv32-unknown-linux", "-O3", "-nostdlib", "-static",
          "-mllvm", "-lsr-term-fold",
          "-Xclang", "-target-feature", "-Xclang", "+no-default-unroll", 
          "-Xlinker", "-Tlinker.ld",          
          "-Xclang", "-target-feature", "-Xclang", "+unaligned-scalar-mem",
          "-mllvm", "--prefer-inloop-reductions",
          ]

JIT = "../cdsljit"
CORE_DESC = "../core_descs/XCoreVSIMD.core_desc"

SOURCE_FILES = ["src/entry.S", "src/benchmark.S", "src/test.c"]

BENCHMARKS = [
    ("add8", 8),
    ("add16", 16),
    ("dot8", 8),
    ("dot16", 16),
    #("saxpy8", 8),
    #("saxpy16", 16),
    #("matmul8", 8),
    #("matmul16", 16),
]
SIZES = [ 65536 ]
FLAG_SETS = [["-march=rv32im"], ["-march=rv32im_xcvsimd"]]#, ["-march=rv32im_xcvsimd_xcvsimdmul"]]
DESCRIPTIONS = ["", "_simd", "_simd_mac"]

def compile(func, name, size, elem_size, extra_flags):
    os.makedirs("build", exist_ok=True)
    binary = "build/{}_{}".format(name, size)
    subprocess.run(
        [CC, "-o", binary, 
         "-DFUNCTION={}".format(func), 
         "-DN={}".format(size if name.find("matmul") == -1 else int(sqrt(size))),
         "-DSIZE={}".format(size),
         "-DCHAR=\'{}\'".format('A' if name.find("to_upper_z") != -1 else 'a'),
         "-DELEM_SIZE={}".format(elem_size)] + 
         SOURCE_FILES + CFLAGS + extra_flags)
    
    subprocess.run(
        [CC, "-S", "-o", binary + ".s", 
         "-DFUNCTION={}".format(func),
         "-DN={}".format(size if name.find("matmul") == -1 else int(sqrt(size))),
         "-DSIZE={}".format(size), 
         "-DELEM_SIZE={}".format(elem_size)] + 
         ["src/test.c"] + CFLAGS + extra_flags)
    return binary

def run(binary):
    proc = subprocess.run(
        [JIT, CORE_DESC, binary], stdout=subprocess.PIPE)
    return proc.stdout.decode()

counts: "Dict[str, [(str, int)]]" = {}
def register_counts(output: str, name: str):
    global counts
    for line in output.splitlines():
        tok = [t.strip() for t in line.split(':')]
        if (counts.get(name) == None): counts[name] = []
        counts[name].append((tok[0], int(tok[1])))
    counts[name].pop()

def compare_files(files):
    if (len(files) == 0): return
    for i in range(1, len(files)):
        proc = subprocess.run(["diff", files[0], files[i]])
        if (proc.returncode != 0):
            print("mismatch between {} and {}".format(files[0], files[i]))
            exit(1)

def make_graph (cutoff):
    funcs = []
    instrs = []
    hits = []
    
    # Preprocess
    instrs_total = {}
    for name in counts:
        other_cnt = 0
        for (inst, n) in counts[name]:
            if instrs_total.get(inst) == None: instrs_total[inst] = 0
            instrs_total[inst] += n
            if n > cutoff:
                funcs.append(name)
                instrs.append(inst)
                hits.append(n)
            else: other_cnt += n
        if other_cnt != 0:
            funcs.append(name)
            instrs.append("other")
            hits.append(other_cnt)
    
    # Get unique color list for instrs
    instr_total_sorted = [k for k,_ in sorted(instrs_total.items(), key=operator.itemgetter(1), reverse=True)]
    print(instr_total_sorted)
    instr_total_sorted.append("other")
    cmap = plt.get_cmap("tab20c")
    color_dict = {}
    for idx, name in enumerate(instr_total_sorted):
        color_dict[name] = cmap(idx)
    color_dict["other"] = "darkgray"
    
    # Plot Setup
    df = pd.DataFrame({"Funcs": funcs, "Instrs": instrs, "Hits": hits})
    pivot_df = df.pivot_table(values='Hits', index='Funcs', columns='Instrs', aggfunc=np.sum, fill_value=0)
    unique_funcs = pivot_df.index.tolist()
    unique_instrs = pivot_df.columns.tolist()

    bar_data = defaultdict(list)
    for func in unique_funcs:
        for instr in unique_instrs:
            bar_data[func].append(pivot_df.loc[func, instr])

    # Plotting
    fig, ax = plt.subplots()
    fig.set_size_inches(w=7.2, h=5.5)
    x = []
    width = 0.8
    max_count = 0

    for idx, func in enumerate(unique_funcs):
        pos_x = idx + (width/10 if (idx % 2 == 0) else -width/10)
        x.append(pos_x)
        plt.gca().set_prop_cycle(None)
        bottom = 0
        data = sorted([(value, instr) for instr, value in zip(unique_instrs, bar_data[func])], reverse=True)
        for value, instr in data:
            bar = ax.bar(pos_x, value, bottom=bottom, width=width, color=color_dict[instr])
            if (value > cutoff):
                ax.text(bar[0].get_x() + bar[0].get_width() * 0.5, bottom + value / 2, 
                    instr, ha='center', va='center', color='black', fontsize=6)
            bottom += value
    
        ax.text(bar[0].get_x() + bar[0].get_width() * 0.5, bottom, 
                f'{int(bottom):,}', ha='center', va='bottom', color='black', fontsize=6)
        if bottom > max_count: max_count = bottom
    
    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.set_xlabel('Function')
    #ax.set_ylabel('Count')
    ax.set_title('Dynamic Instruction Count')
    ax.set_ylim(top=max_count * 1.1, bottom=0)
    ax.set_xticks(x)
    ax.set_xticklabels(unique_funcs)
    
    plt.savefig("plot.pgf")
    plt.show()
        
       
def run_benchmarks():
    for (func, elem_size) in BENCHMARKS:
        for flags_id, flags in enumerate(FLAG_SETS):
            for size in SIZES:
                name = func + DESCRIPTIONS[flags_id]

                binary = compile(func, name, size, elem_size, flags)
                output = run(binary)

                print("Results {}, size={}, flags={}".format(name, size, flags))
                print(output)
                print("\n\n")

                register_counts(output, name)

def compare_results():
    for (func, elem_size) in BENCHMARKS:
        for size in SIZES:
            dump_files = [] 
            for flags_id, flags in enumerate(FLAG_SETS):
                name = func + DESCRIPTIONS[flags_id] + f"_{size}_bss.bin"
                dump_files.append("build/" + name)
            compare_files(dump_files)

if __name__ == "__main__":
    run_benchmarks()
    compare_results()
    make_graph(7000)
