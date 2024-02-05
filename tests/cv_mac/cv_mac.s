# RUN: llvm-mc %s -triple=riscv32 -mattr=+xcvmac -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+xcvmac < %s \
# RUN:     | llvm-objdump --mattr=+xcvmac -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: cv.mac a4, ra, s0
# CHECK-ASM: encoding: [0x2b,0xb7,0x80,0x90]
cv.mac a4, ra, s0
