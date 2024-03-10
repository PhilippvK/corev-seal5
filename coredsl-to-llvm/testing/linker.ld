MEMORY
{
    prog : ORIGIN = 0x80000000, LENGTH = 0x10000000
    data : ORIGIN = 0xA0000000, LENGTH = 0x10000000
}
SECTIONS
{
    .text : { *(.text*) } > prog
    # put everything in data
    .data : ALIGN(4) { *(.rodata*); *(.data*); *(.sdata*); *(.srodata*); } > data
} 
