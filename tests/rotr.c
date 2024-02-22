int rotr32(int x, int n) {
    return (x >> n) | (x << (32 - n));
}
