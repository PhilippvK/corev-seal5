//.section .data
//.array1: .fill (SIZE * ELEM_SIZE / 8), 1, CHAR
//.array2: .fill (SIZE * ELEM_SIZE / 8), 1, CHAR
//.array3: .fill (SIZE * ELEM_SIZE / 8), 1, CHAR

.section .bss
.array1: .zero (SIZE * ELEM_SIZE / 8)
.array2: .zero (SIZE * ELEM_SIZE / 8)
.array3: .zero (SIZE * ELEM_SIZE / 8)

.text
.globl main
main:
    addi sp, sp, -16
    sw ra, 12(sp)

    li a0, N
    la a1, .array1
    la a2, .array2
    la a3, .array3

    // This is only used as "a" for saxpy
    li a4, 42
    
    call FUNCTION
    
    lw ra, 12(sp)
    addi sp, sp, -16
    ret
