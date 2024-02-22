int word[4][120];

int
MixColumn_AddRoundKey (int statemt[32], int nb, int n)
{
  int ret[8 * 4], j;
  register int x;

  for (j = 0; j < nb; ++j)
    {
      ret[j * 4] = (statemt[j * 4] << 1);
      if ((ret[j * 4] >> 8) == 1)
        ret[j * 4] ^= 283;
      x = statemt[1 + j * 4];
      x ^= (x << 1);
      if ((x >> 8) == 1)
        ret[j * 4] ^= (x ^ 283);
      else
        ret[j * 4] ^= x;
      ret[j * 4] ^=
        statemt[2 + j * 4] ^ statemt[3 + j * 4] ^ word[0][j + nb * n];

      ret[1 + j * 4] = (statemt[1 + j * 4] << 1);
      if ((ret[1 + j * 4] >> 8) == 1)
        ret[1 + j * 4] ^= 283;
      x = statemt[2 + j * 4];
      x ^= (x << 1);
      if ((x >> 8) == 1)
        ret[1 + j * 4] ^= (x ^ 283);
      else
        ret[1 + j * 4] ^= x;
      ret[1 + j * 4] ^=
        statemt[3 + j * 4] ^ statemt[j * 4] ^ word[1][j + nb * n];

      ret[2 + j * 4] = (statemt[2 + j * 4] << 1);
      if ((ret[2 + j * 4] >> 8) == 1)
        ret[2 + j * 4] ^= 283;
      x = statemt[3 + j * 4];
      x ^= (x << 1);
      if ((x >> 8) == 1)
        ret[2 + j * 4] ^= (x ^ 283);
      else
        ret[2 + j * 4] ^= x;
      ret[2 + j * 4] ^=
        statemt[j * 4] ^ statemt[1 + j * 4] ^ word[2][j + nb * n];

      ret[3 + j * 4] = (statemt[3 + j * 4] << 1);
      if ((ret[3 + j * 4] >> 8) == 1)
        ret[3 + j * 4] ^= 283;
      x = statemt[j * 4];
      x ^= (x << 1);
      if ((x >> 8) == 1)
        ret[3 + j * 4] ^= (x ^ 283);
      else
        ret[3 + j * 4] ^= x;
      ret[3 + j * 4] ^=
        statemt[1 + j * 4] ^ statemt[2 + j * 4] ^ word[3][j + nb * n];
    }
  for (j = 0; j < nb; ++j)
    {
      statemt[j * 4] = ret[j * 4];
      statemt[1 + j * 4] = ret[1 + j * 4];
      statemt[2 + j * 4] = ret[2 + j * 4];
      statemt[3 + j * 4] = ret[3 + j * 4];
    }
  return 0;
}

/* ******** InversMixColumn ********** */
int
AddRoundKey_InversMixColumn (int statemt[32], int nb, int n)
{
  int ret[8 * 4], i, j;
  register int x;

  for (j = 0; j < nb; ++j)
    {
      statemt[j * 4] ^= word[0][j + nb * n];
      statemt[1 + j * 4] ^= word[1][j + nb * n];
      statemt[2 + j * 4] ^= word[2][j + nb * n];
      statemt[3 + j * 4] ^= word[3][j + nb * n];
    }
  for (j = 0; j < nb; ++j)
    for (i = 0; i < 4; ++i)
      {
        x = (statemt[i + j * 4] << 1);
        if ((x >> 8) == 1)
          x ^= 283;
        x ^= statemt[i + j * 4];
        x = (x << 1);
        if ((x >> 8) == 1)
          x ^= 283;
        x ^= statemt[i + j * 4];
        x = (x << 1);
        if ((x >> 8) == 1)
          x ^= 283;
        ret[i + j * 4] = x;

        x = (statemt[(i + 1) % 4 + j * 4] << 1);
        if ((x >> 8) == 1)
          x ^= 283;
        x = (x << 1);
        if ((x >> 8) == 1)
          x ^= 283;
        x ^= statemt[(i + 1) % 4 + j * 4];
        x = (x << 1);
        if ((x >> 8) == 1)
          x ^= 283;
        x ^= statemt[(i + 1) % 4 + j * 4];
        ret[i + j * 4] ^= x;

        x = (statemt[(i + 2) % 4 + j * 4] << 1);
        if ((x >> 8) == 1)
          x ^= 283;
        x ^= statemt[(i + 2) % 4 + j * 4];
        x = (x << 1);
        if ((x >> 8) == 1)
          x ^= 283;
        x = (x << 1);
        if ((x >> 8) == 1)
          x ^= 283;
        x ^= statemt[(i + 2) % 4 + j * 4];
        ret[i + j * 4] ^= x;

        x = (statemt[(i + 3) % 4 + j * 4] << 1);
        if ((x >> 8) == 1)
          x ^= 283;
        x = (x << 1);
        if ((x >> 8) == 1)
          x ^= 283;
        x = (x << 1);
        if ((x >> 8) == 1)
          x ^= 283;
        x ^= statemt[(i + 3) % 4 + j * 4];
        ret[i + j * 4] ^= x;
      }
  for (i = 0; i < nb; ++i)
    {
      statemt[i * 4] = ret[i * 4];
      statemt[1 + i * 4] = ret[1 + i * 4];
      statemt[2 + i * 4] = ret[2 + i * 4];
      statemt[3 + i * 4] = ret[3 + i * 4];
    }
  return 0;
}
