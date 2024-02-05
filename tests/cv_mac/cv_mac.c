// RUN: clang -cc1 -triple riscv32 -target-feature +m -target-feature +xcvmac -S -O3 %s -o - \
// RUN:     | FileCheck %s  -check-prefix=CHECK

// CHECK-LABEL: mac:
// CHECK-COUNT-1: cv.mac {{.*}}
int mac(int acc, int a, int b)
{
  acc += a * b;
  return acc;
}

// CHECK-LABEL: mac2:
// CHECK-COUNT-2: cv.mac {{.*}}
int mac2(int acc, int a, int b, int c, int d)
{
  acc += a * b;
  acc += c * d;
  return acc;
}
