.text
.globl _start
_start:
    li sp, 0x80000000
    call main

    li a0, 2
    ecall

.globl puts
puts:
    mv a2, a0
    li a0, 0
    .puts_loop:
        lbu a1, 0(a2)
        beqz a1, .puts_break
        ecall
        addi a2, a2, 1
        j .puts_loop
    .puts_break:
    li a1, '\n'
    ecall
    ret

.globl printhex
printhex:
    mv a3, a0
    li a4, 28
    .printhex_loop:
        srl a1, a3, a4
        andi a1, a1, 0xf
        addi a1, a1, '0'
        li a2, '9'
        ble a1, a2, .dec_digit
        addi a1, a1, 'a'-'0'-10
        .dec_digit:
        li a0, 0
        ecall
        addi a4, a4, -4
        bgez a4, .printhex_loop
    ret

.globl memcpy
.type	memcpy, @function
memcpy:
    beqz a2, .memcpy_end
    mv a3, a0
    .memcpy_loop:
        lb a4, 0(a1)
        sb a4, 0(a3)
        addi a1, a1, 1
        addi a3, a3, 1
        addi a2, a2, -1
        bnez a2, .memcpy_loop
    .memcpy_end:
    ret
    
.globl memset
.type	memset, @function
memset:
    beqz a2, .memset_end
    mv a3, a0
    .memset_loop:
        sb a1, 0(a3)
        addi a3, a3, 1
        addi a2, a2, -1
        bnez a2, .memset_loop
    .memset_end:
    ret
