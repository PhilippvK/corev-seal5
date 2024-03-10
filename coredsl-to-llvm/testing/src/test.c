#include <stdint.h>
#include <stddef.h>

void to_upper(size_t n, char c[n])
{
    for (size_t i = 0; i < n; i++)
        c[i] += (c[i] >= 'a' && c[i] <= 'z') ? ('A'-'a') : 0;
}

void add8(size_t n, int8_t d[restrict n], int8_t a[n], int8_t b[n])
{
    for (size_t i = 0; i < n; i++)
        d[i] = a[i] + b[i];
}
void add16(size_t n, int16_t d[restrict n], int16_t a[n], int16_t b[n])
{
    for (size_t i = 0; i < n; i++)
        d[i] = a[i] + b[i];
}

int32_t dot8(size_t n, int8_t a[n], int8_t b[n])
{
    int32_t acc = 0;
    for (size_t i = 0; i < n; i++)
        acc += a[i] * b[i];

    return acc;
}
int32_t dot16(size_t n, int16_t a[n], int16_t b[n])
{
    int32_t acc = 0;
    for (size_t i = 0; i < n; i++)
        acc += a[i] * b[i];

    return acc;
}

void saxpy8(size_t n, int8_t d[restrict n], int8_t x[n], int8_t y[n], int8_t a)
{
    for (size_t i = 0; i < n; i++)
        d[i] = a * x[i] + y[i];
}
void saxpy16(size_t n, int16_t d[restrict n], int16_t x[n], int16_t y[n], int16_t a)
{
    for (size_t i = 0; i < n; i++)
        d[i] = a * x[i] + y[i];
}

void matmul8 (size_t n, int8_t d[restrict n*n], int8_t a[n*n], int8_t b[n*n])
{
    for (size_t a_y = 0; a_y < n; a_y++)
    {
        for (size_t b_y = 0; b_y < n; b_y++)
        {
            for (size_t i = 0; i < n; i++)
                d[a_y * n + i] += a[a_y * n + b_y] * b[b_y * n + i];
        }
    }
}
void matmul16 (size_t n, int16_t d[restrict n*n], int16_t a[n*n], int16_t b[n*n])
{
    for (size_t a_y = 0; a_y < n; a_y++)
    {
        for (size_t b_y = 0; b_y < n; b_y++)
        {
            for (size_t i = 0; i < n; i++)
                d[a_y * n + i] += a[a_y * n + b_y] * b[b_y * n + i];
        }
    }
}
