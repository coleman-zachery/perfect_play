def _gen(stop=4):
    v = [0, 1]
    for _ in range(stop): v = [i + j for i in v for j in v]
    return v

_BYTE_POPCOUNT = None
def _byte_popcount_table():
    global _BYTE_POPCOUNT
    if _BYTE_POPCOUNT is None: _BYTE_POPCOUNT = _gen(4)
    return _BYTE_POPCOUNT

def count_bits(n: int) -> int:
    table = _byte_popcount_table()
    count = 0
    while n:
        count += table[n & 0xFF]  # lowest 8 bits
        n >>= 8
    return count

###################################################################

def chunk_slice_bits(n: int, indices, chunk_size: int, num_chunks=None) -> int:
    """
    Build a new integer by taking bits at `indices` from each chunk of size `chunk_size`.
    Indices are 0=LSB within each chunk.
    Output packs selected bits tightly (first chunk contributes the lowest output bits).
    """
    if num_chunks is None:
        num_chunks = max(1, (n.bit_length() + chunk_size - 1) // chunk_size)

    mask = (1 << chunk_size) - 1
    out = 0
    out_shift = 0

    for c in range(num_chunks):
        chunk = (n >> (c * chunk_size)) & mask
        for idx in indices:
            out |= ((chunk >> idx) & 1) << out_shift
            out_shift += 1

    return out

#   :: usage ::
#   n = 0b1010
#   print(bin(bit_slice_chunks(n, [1], 2)))
