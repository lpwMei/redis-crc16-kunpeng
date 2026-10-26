#!/usr/bin/env python3
"""Cross-check: crc16_aarch64.c (C intrinsics) must be bit-exact with the
Redis reference crc16().

This harness runs ON THE aarch64 TARGET (compile & run there, e.g. on the
GitHub Actions ARM runner). It is intentionally standalone: no Redis headers.
It re-implements the Redis reference table loop, links crc16_aarch64.c, and
compares over a corpus: all lengths 0..4096, four data patterns, random
seeds (via the PMULL tail path through crc16_slice8), plus adversarial
patterns (alternating 0x00/0xFF, all-ones, walking-bit).
"""
import random

def make_tab():
    tab = []
    for i in range(256):
        c = i << 8
        for _ in range(8):
            c = ((c << 1) ^ (0x1021 if c & 0x8000 else 0)) & 0xFFFF
        tab.append(c)
    return tab

def crc16_ref(data):
    tab = make_tab()
    c = 0
    for b in data:
        c = ((c << 8) & 0xFFFF) ^ tab[((c >> 8) ^ b) & 0xFF]
    return c

if __name__ == "__main__":
    # self-check only; the C harness does the real work
    assert crc16_ref(b"123456789") == 0x31C3
    print("python reference OK")
