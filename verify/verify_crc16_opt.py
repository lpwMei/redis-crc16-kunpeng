#!/usr/bin/env python3
"""Full verification suite for the Redis CRC16 (XMODEM 0x1021) optimization.

Verifies, at bit level, BEFORE any C code is written:
 1. Anchor: our constant-derivation framework reproduces ISA-L's shipped
    crc16_t10dif_pmull.S fold constants for poly 0x8BB7 (known-good silicon):
    fold pair = (rem(x^496), rem(x^560)) with reduction poly 0x18BB7
    (ISA-L lanes carry an extra x^16 headroom: 512-16 / 576-16).
 2. Slice-by-8 tables: T_k[i] = rem(i * x^(8k+16)) mod P, k=0..7, and the
    8-byte recurrence. Verified against the reference byte loop.
 3. PMULL fold pipeline (exactly what the C intrinsics will do):
    - 4 x 128-bit accumulators, 64B-stride fold, K_lo=rem(x^512), K_hi=rem(x^576)
      (drifting-position invariant: each fold advances the lane's data by one
      64B group = 512 bits; hi half of the 128b accumulator sits 64 bits higher)
    - endgame: x3 ^ fd(x2,128b) ^ fd(x1,256b) ^ fd(x0,384b)  (+64b hi lanes)
      (last-group lane j ends 16*(3-j) bytes before lane 3)
    - final: t*x^16 (C16/C80), 80->64->48->32 chains, Barrett 32->16
    Verified against reference for lengths 64..8192 x 4 data patterns.

Run:  python verify_crc16_opt.py
"""
import random

# ---------------------------------------------------------------- reference
def make_tab(poly_reg):
    tab = []
    for i in range(256):
        c = i << 8
        for _ in range(8):
            c = ((c << 1) ^ (poly_reg if c & 0x8000 else 0)) & 0xFFFF
        tab.append(c)
    return tab

def crc16_ref(data, init=0, poly_reg=0x1021):
    """Byte-at-a-time, identical to Redis src/crc16.c (XMODEM)."""
    tab = make_tab(poly_reg)
    c = init & 0xFFFF
    for b in data:
        c = ((c << 8) & 0xFFFF) ^ tab[((c >> 8) ^ b) & 0xFF]
    return c

# ------------------------------------------------------------ GF(2) helpers
def gmul(a, b):
    r = 0
    while b:
        lb = b & -b
        r ^= a << (lb.bit_length() - 1)
        b ^= lb
    return r

def gmod(a, p, plen):
    assert p.bit_length() == plen + 1, "p must carry the x^plen term"
    while a and (a.bit_length() - 1) >= plen:
        a ^= p << (a.bit_length() - 1 - plen)
    return a

def gdiv(numer, p, plen):
    q = 0
    while numer and (numer.bit_length() - 1) >= plen:
        s = numer.bit_length() - 1 - plen
        numer ^= p << s
        q ^= 1 << s
    return q

def xpow_mod(p, plen, e_bits):
    r = 1
    base = 2
    while e_bits:
        if e_bits & 1:
            r = gmod(gmul(r, base), p, plen)
        base = gmod(gmul(base, base), p, plen)
        e_bits >>= 1
    return r

def clmul64(a, b):
    """64x64->128 carry-less product (simulates vmull_p64)."""
    return gmul(a, b)

M64 = (1 << 64) - 1
M32 = 0xFFFFFFFF

# ------------------------------------------------------------------ anchor
def check_anchors():
    """ISA-L crc16_t10dif_pmull.S ships fold_constant {0x87e7, 0x371d}<<16
    for T10-DIF. Their lanes carry an extra x^16 headroom, so the pair is
    rem(x^496)/rem(x^560) = rem(x^512)/rem(x^576) shifted down by x^16."""
    p = 0x18BB7
    k_lo = xpow_mod(p, 16, 496)
    k_hi = xpow_mod(p, 16, 560)
    ok = (k_lo == 0x87e7) and (k_hi == 0x371d)
    print(f"[anchor] T10DIF fold pair derived lo={k_lo:#06x} hi={k_hi:#06x}"
          f" (shipped 0x87e7/0x371d) ->", "PASS" if ok else "FAIL")
    return ok

# --------------------------------------------------------------- slice-by-8
def gen_slice_tables(poly_reg=0x1021, p_red=0x11021):
    """T[k][i] = rem(i * x^(8k+16)) mod P, k=0..7. T[0] == crc16tab."""
    tabs = []
    for k in range(8):
        xp = xpow_mod(p_red, 16, 8 * k + 16)
        tabs.append([gmod(gmul(i, xp), p_red, 16) for i in range(256)])
    classic = make_tab(poly_reg)
    ok0 = tabs[0] == classic
    print("[slice8] T[0] == crc16tab:", "PASS" if ok0 else "FAIL")
    return tabs, ok0

def crc16_slice8(data, init, tabs):
    """crcspeed16little recurrence (verified against Redis crc16.c above)."""
    c = init & 0xFFFF
    i, n = 0, len(data)
    while n - i >= 8:
        w = int.from_bytes(data[i:i + 8], 'little')
        c = (tabs[7][(w & 0xFF) ^ ((c >> 8) & 0xFF)]
             ^ tabs[6][((w >> 8) & 0xFF) ^ (c & 0xFF)]
             ^ tabs[5][(w >> 16) & 0xFF]
             ^ tabs[4][(w >> 24) & 0xFF]
             ^ tabs[3][(w >> 32) & 0xFF]
             ^ tabs[2][(w >> 40) & 0xFF]
             ^ tabs[1][(w >> 48) & 0xFF]
             ^ tabs[0][w >> 56])
        i += 8
    t0 = tabs[0]
    while i < n:
        c = ((c << 8) & 0xFFFF) ^ t0[((c >> 8) ^ data[i]) & 0xFF]
        i += 1
    return c

def check_slice8(tabs):
    rng = random.Random(7)
    ok = True
    for L in list(range(0, 72)) + [100, 255, 256, 1000]:
        for _ in range(6):
            data = bytes(rng.getrandbits(8) for _ in range(L))
            seed = rng.getrandbits(16)
            if crc16_slice8(data, seed, tabs) != crc16_ref(data, seed):
                ok = False
                print(f"[slice8] FAIL L={L} seed={seed:#x}")
    print("[slice8] recurrence vs reference (lens 0..72,100,255,256,1000):",
          "PASS" if ok else "FAIL")
    return ok

# ------------------------------------------------------------- PMULL sim
P_RED = 0x11021                        # XMODEM reduction poly (x^16 term included)
K_LO = xpow_mod(P_RED, 16, 512)        # fold: low half advances 512 bits (64B group)
K_HI = xpow_mod(P_RED, 16, 576)        # high half: +64 bits
EG = {d: (xpow_mod(P_RED, 16, d), xpow_mod(P_RED, 16, d + 64))
      for d in (128, 256, 384)}        # endgame bit-distances (x2/x1/x0)
C16 = xpow_mod(P_RED, 16, 16)          # final multiply: t * x^16
C80 = xpow_mod(P_RED, 16, 80)
C64 = xpow_mod(P_RED, 16, 64)          # reduction chain 80->64->48->32
C32 = xpow_mod(P_RED, 16, 32)
MU = gdiv(1 << 47, P_RED, 16)          # Barrett: floor(x^47 / P)

def rev16(b):
    """16B lane byte-reversed -> integer == MSB-first data polynomial."""
    return int.from_bytes(bytes(reversed(b)), 'little')

def fold_step(x, y):
    return (clmul64(x & M64, K_LO) ^ clmul64(x >> 64, K_HI) ^ y)

def crc16_pmull(data, tabs):
    """Simulates the intrinsics pipeline for len >= 64 (init 0)."""
    ngroups = len(data) // 64
    x = [rev16(data[16 * j:16 * j + 16]) for j in range(4)]
    off = 64
    for _ in range(ngroups - 1):
        y = [rev16(data[off + 16 * j:off + 16 * j + 16]) for j in range(4)]
        x = [fold_step(x[j], y[j]) for j in range(4)]
        off += 64
    # endgame: t = x3 ^ fd(x2,128b) ^ fd(x1,256b) ^ fd(x0,384b)
    t = x[3]
    for j, d in ((2, 128), (1, 256), (0, 384)):
        klo, khi = EG[d]
        t ^= clmul64(x[j] & M64, klo) ^ clmul64(x[j] >> 64, khi)
    # t * x^16 : u = t_lo*C16 ^ t_hi*C80   (u <= 80 bits)
    u = clmul64(t & M64, C16) ^ clmul64(t >> 64, C80)
    # 80 -> 64: v = u_lo ^ (u_hi * C64)
    v = (u & M64) ^ clmul64(u >> 64, C64)
    # 64 -> 48: w = v_lo32 ^ (v_hi32 * C32)
    w = (v & M32) ^ clmul64(v >> 32, C32)
    # 48 -> 32: z = w_lo32 ^ (w_hi16 * C32)
    z = (w & M32) ^ clmul64(w >> 32, C32)
    # Barrett 32 -> 16: q = (z*mu)>>47 ; r = z ^ q*P  (r < 2^16 exactly)
    q = clmul64(z, MU) >> 47
    r = z ^ clmul64(q, P_RED)
    assert r < (1 << 16), "Barrett failed to reduce"
    head = r
    tail = data[ngroups * 64:]
    return crc16_slice8(tail, head, tabs)

def check_pmull(tabs):
    rng = random.Random(42)
    ok = True
    lens = list(range(64, 200)) + [255, 256, 257, 511, 512, 513, 1023, 1024,
                                    1025, 2047, 2048, 4096, 5000, 8192]
    for L in lens:
        for pat in ("rand", "zero", "ff", "alt"):
            if pat == "rand":
                data = bytes(rng.getrandbits(8) for _ in range(L))
            elif pat == "zero":
                data = b"\x00" * L
            elif pat == "ff":
                data = b"\xff" * L
            else:
                data = bytes((i * 37 + 11) & 0xFF for i in range(L))
            a = crc16_pmull(data, tabs)
            b = crc16_ref(data)
            if a != b:
                ok = False
                print(f"[pmull] FAIL L={L} pat={pat}: {a:#06x} != {b:#06x}")
    print("[pmull] fold pipeline vs reference (lens 64..8192 x 4 patterns):",
          "PASS" if ok else "FAIL")
    return ok

def check_kat():
    kats = [
        (b"", 0x0000),
        (b"123456789", 0x31C3),                    # XMODEM check value
        (b"key:__rand_int__", crc16_ref(b"key:__rand_int__")),
        (b"\x00" * 8, 0x0000),
    ]
    ok = all(crc16_ref(d) == v for d, v in kats)
    print("[kat] reference known answers:", "PASS" if ok else "FAIL")
    return ok

if __name__ == "__main__":
    print("CRC16 XMODEM optimization verification")
    print("=" * 60)
    r = check_kat()
    r &= check_anchors()
    tabs, ok0 = gen_slice_tables()
    r &= ok0
    r &= check_slice8(tabs)
    r &= check_pmull(tabs)
    print("=" * 60)
    print("OVERALL:", "ALL PASS" if r else "FAILURES PRESENT")
    print("\n// constants for crc16_aarch64.c:")
    print(f"K_LO={K_LO:#06x} K_HI={K_HI:#06x} C16={C16:#06x} C80={C80:#06x} "
          f"C64={C64:#06x} C32={C32:#06x} MU={MU:#010x}")
    for d in (128, 256, 384):
        print(f"EG{d}: lo={EG[d][0]:#06x} hi={EG[d][1]:#06x}")
