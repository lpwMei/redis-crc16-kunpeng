/* CRC16 (XMODEM, poly 0x1021) accelerated implementation for aarch64 with
 * the NEON PMULL (carry-less multiply) extension.
 *
 * This file is self-contained: no Redis headers, so it can also be linked
 * into standalone test harnesses. It is only compiled on aarch64 (see
 * src/Makefile); on other machines it contributes nothing to the build.
 *
 * Contract: bit-exact with the reference table loop in src/crc16.c for every
 * (buf, len), len == 0 included, any alignment.
 *
 * Activation policy (all inside crc16_aarch64_init(), so the caller stays
 * dumb):
 *   1. compiled with -march=...+crypto (__ARM_FEATURE_CRYPTO), otherwise
 *      this translation unit degrades to "not available" stubs;
 *   2. HWCAP_PMULL at runtime (safe to run the binary on any armv8 CPU);
 *   3. Huawei Kunpeng platform only (HiSilicon CPU implementer 0x48),
 *      per the support scope of this optimization. Define
 *      REDIS_CRC16_ANY_PMULL_CPU at build time to relax the vendor gate
 *      for validation runs on other PMULL-capable aarch64 CPUs.
 *
 * Two sub-paths behind one entry point:
 *   - len < 64 : slice-by-8 table lookups (8 bytes per iteration, breaking
 *                the per-byte serial dependency chain of the table loop);
 *   - len >= 64: four 128-bit PMULL folding lanes at 64-byte stride, an
 *                endgame merge, a shift/xor reduction chain and one Barrett
 *                step down to 16 bits; the 0..63-byte tail finishes with the
 *                same slice-by-8 tables.
 *
 * All fold/reduction constants were derived from first principles and
 * bit-level verified against the reference implementation before this code
 * was written; see verify/verify_crc16_opt.py in the companion repository.
 * The same framework reproduces ISA-L's shipped T10-DIF PMULL constants as
 * an anchor (rem(x^496)/rem(x^560) for poly 0x18BB7).
 *
 * Copyright (c) 2026, Huawei Technologies Co., Ltd.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the copyright holder nor the names of its
 *       contributors may be used to endorse or promote products derived from
 *       this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__aarch64__) && defined(__ARM_FEATURE_CRYPTO) && \
    defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__

#include <arm_neon.h>
#include <sys/auxv.h>

/* ------------------------------------------------------------------ */
/* Fold / reduction constants (XMODEM poly; reduction form P = 0x11021)*/
/* rem(x^n) computed in GF(2)[x]/P.                                    */
/* ------------------------------------------------------------------ */
#define CRC16_K_LO   UINT64_C(0x13fc)  /* rem(x^512): 64B fold, low half  */
#define CRC16_K_HI   UINT64_C(0x8832)  /* rem(x^576): 64B fold, high half */
#define CRC16_EG128L UINT64_C(0xaefc)  /* rem(x^128): endgame lane x2 low */
#define CRC16_EG128H UINT64_C(0x650b)  /* rem(x^192): endgame lane x2 high*/
#define CRC16_EG256L UINT64_C(0x8e29)  /* rem(x^256): endgame lane x1 low */
#define CRC16_EG256H UINT64_C(0x26aa)  /* rem(x^320): endgame lane x1 high*/
#define CRC16_EG384L UINT64_C(0xcde2)  /* rem(x^384): endgame lane x0 low */
#define CRC16_EG384H UINT64_C(0x2535)  /* rem(x^448): endgame lane x0 high*/
#define CRC16_C16    UINT64_C(0x1021)  /* rem(x^16)  = the poly itself   */
#define CRC16_C80    UINT64_C(0xeb23)  /* rem(x^80)                       */
#define CRC16_C64    UINT64_C(0xb861)  /* rem(x^64)                       */
#define CRC16_C32    UINT64_C(0x3730)  /* rem(x^32)                       */
#define CRC16_MU     UINT64_C(0x88981a38) /* Barrett: floor(x^47 / P)     */
#define CRC16_P      UINT64_C(0x11021) /* reduction polynomial           */

/* Slice-by-8 tables, algebraically identical to crcspeed16little tables
 * (src/crcspeed.c): T[k][b] = rem(b(x) * x^(8k+16)).  T[0] == crc16tab. */
static uint16_t crc16_slice8_tab[8][256];
static int crc16_tables_ready = 0;

/* ------------------------------------------------------------------ */
/* Activation policy                                                   */
/* ------------------------------------------------------------------ */

#ifndef REDIS_CRC16_ANY_PMULL_CPU
/* Return true if this is a Huawei Kunpeng platform (HiSilicon CPU
 * implementer 0x48). Reads /proc/cpuinfo once at init time. */
static int crc16_kunpeng_cpu(void) {
    FILE *f = fopen("/proc/cpuinfo", "r");
    char line[256];
    int found = 0;

    if (!f) return 0;
    while (fgets(line, sizeof(line), f) != NULL) {
        if (strncmp(line, "CPU implementer", 15) == 0) {
            unsigned int implementer = 0;
            if (sscanf(line + 15, "%*[^:]: %x", &implementer) == 1 &&
                implementer == 0x48)
                found = 1;
            break; /* homogeneous multi-core: first match decides */
        }
    }
    fclose(f);
    return found;
}
#endif

/* Vendor gate: Huawei Kunpeng (HiSilicon implementer 0x48) by default.
 * The environment variable REDIS_CRC16_ANY_PMULL_CPU=1 widens the gate to
 * any PMULL-capable aarch64 CPU (used by CI validation on Ampere runners).
 * It can only widen, never disable the path on Kunpeng. A build-time
 * -DREDIS_CRC16_ANY_PMULL_CPU compiles the check away entirely. */
static int crc16_vendor_ok(void) {
#ifdef REDIS_CRC16_ANY_PMULL_CPU
    return 1;
#else
    static int checked = 0, allowed = 0;
    if (!checked) {
        const char *env = getenv("REDIS_CRC16_ANY_PMULL_CPU");
        int widen = (env != NULL && env[0] == '1' && env[1] == '\0');
        allowed = widen || crc16_kunpeng_cpu();
        checked = 1;
    }
    return allowed;
#endif
}

/* Decide whether the accelerated path can be used, and build its tables.
 * Returns 1 when crc16_aarch64_pmull() is usable, 0 otherwise. */
int crc16_aarch64_init(void) {
    if (!(getauxval(AT_HWCAP) & HWCAP_PMULL)) return 0;
    if (!crc16_vendor_ok()) return 0;

    if (!crc16_tables_ready) {
        uint16_t *t0 = crc16_slice8_tab[0];
        int n, k;

        for (n = 0; n < 256; n++) {
            uint16_t c = (uint16_t)(n << 8);
            for (k = 0; k < 8; k++)
                c = (uint16_t)((c << 1) ^ ((c & 0x8000) ? 0x1021 : 0));
            t0[n] = c;
        }
        for (n = 0; n < 256; n++) {
            uint16_t crc = t0[n];
            for (k = 1; k < 8; k++) {
                crc = (uint16_t)(t0[(crc >> 8) & 0xff] ^ (crc << 8));
                crc16_slice8_tab[k][n] = crc;
            }
        }
        crc16_tables_ready = 1;
    }
    return 1;
}

/* ------------------------------------------------------------------ */
/* PMULL helpers                                                       */
/* ------------------------------------------------------------------ */

/* 64x64 -> 128 carry-less product, low 64 bits. */
static inline uint64_t crc16_clmul64(uint64_t a, uint64_t b) {
    return vgetq_lane_u64(vreinterpretq_u64_p128(
        vmull_p64((poly64_t)a, (poly64_t)b)), 0);
}

/* x = clmul(x_lo, K_LO) ^ clmul(x_hi, K_HI) ^ y   (one 64B fold step) */
static inline uint64x2_t crc16_fold(uint64x2_t x, uint64x2_t y) {
    uint64x2_t lo = vreinterpretq_u64_p128(vmull_p64(
        (poly64_t)vgetq_lane_u64(x, 0), CRC16_K_LO));
    uint64x2_t hi = vreinterpretq_u64_p128(vmull_p64(
        (poly64_t)vgetq_lane_u64(x, 1), CRC16_K_HI));
    return veorq_u64(veorq_u64(lo, hi), y);
}

/* t ^= clmul(x_lo, klo) ^ clmul(x_hi, khi)   (endgame merge) */
static inline uint64x2_t crc16_merge(uint64x2_t t, uint64x2_t x,
                                     uint64_t klo, uint64_t khi) {
    uint64x2_t lo = vreinterpretq_u64_p128(vmull_p64(
        (poly64_t)vgetq_lane_u64(x, 0), klo));
    uint64x2_t hi = vreinterpretq_u64_p128(vmull_p64(
        (poly64_t)vgetq_lane_u64(x, 1), khi));
    return veorq_u64(t, veorq_u64(lo, hi));
}

/* u = clmul(t_lo, C16) ^ clmul(t_hi, C80): multiply accumulator by x^16.
 * u spans at most 81 bits across the two lanes (hi lane <= 17 bits). */
static inline uint64x2_t crc16_mul_x16(uint64x2_t t) {
    uint64x2_t lo = vreinterpretq_u64_p128(vmull_p64(
        (poly64_t)vgetq_lane_u64(t, 0), CRC16_C16));
    uint64x2_t hi = vreinterpretq_u64_p128(vmull_p64(
        (poly64_t)vgetq_lane_u64(t, 1), CRC16_C80));
    return veorq_u64(lo, hi);
}

/* Narrow u (<= 81 bits) down to the final 16-bit CRC.
 * Chain: v = lo ^ hi*C64 (<= 64b); w = lo32(v) ^ hi32(v)*C32 (<= 46b);
 *        z = lo32(w) ^ hi(w)*C32 (<= 32b); Barrett: q = (z*MU)>>47. */
static uint16_t crc16_reduce128(uint64x2_t u) {
    uint64_t lo = vgetq_lane_u64(u, 0);
    uint64_t hi = vgetq_lane_u64(u, 1);          /* <= 17 bits */

    lo ^= crc16_clmul64(hi, CRC16_C64);          /* v <= 64 bits */
    uint64_t w = (lo & 0xffffffffu) ^
                 crc16_clmul64(lo >> 32, CRC16_C32);   /* w <= 46 bits */
    uint32_t z = (uint32_t)((w & 0xffffffffu) ^
                 crc16_clmul64(w >> 32, CRC16_C32));   /* z <= 32 bits */

    /* Barrett reduction: r = z mod P with P of degree 17. */
    uint64x2_t bq = vreinterpretq_u64_p128(vmull_p64((poly64_t)z, CRC16_MU));
    uint64_t q = ((vgetq_lane_u64(bq, 1) << 17) |   /* bits 64..66 -> 17..19 */
                  (vgetq_lane_u64(bq, 0) >> 47));    /* bits 47..63 -> 0..16 */
    return (uint16_t)(z ^ crc16_clmul64(q, CRC16_P));
}

/* Load 16 bytes and byte-reverse: the vector, read as a little-endian
 * integer, equals the MSB-first data polynomial of those bytes. */
static inline uint64x2_t crc16_load16(const unsigned char *p) {
    uint8x16_t v = vrev64q_u8(vld1q_u8(p));
    return vreinterpretq_u64_u8(vextq_u8(v, v, 8));
}

/* ------------------------------------------------------------------ */
/* Slice-by-8 core (short buffers and PMULL tails). Seed semantics     */
/* identical to the reference loop: full 16-bit running CRC.           */
/* ------------------------------------------------------------------ */
static uint16_t crc16_slice8(const unsigned char *buf, size_t len,
                             uint16_t crc) {
    const uint16_t (*t)[256] = (const uint16_t (*)[256])crc16_slice8_tab;
    while (len >= 8) {
        uint64_t w;
        memcpy(&w, buf, 8);
        crc = t[7][(w & 0xff) ^ (crc >> 8)] ^
              t[6][((w >> 8) & 0xff) ^ (crc & 0xff)] ^
              t[5][(w >> 16) & 0xff] ^
              t[4][(w >> 24) & 0xff] ^
              t[3][(w >> 32) & 0xff] ^
              t[2][(w >> 40) & 0xff] ^
              t[1][(w >> 48) & 0xff] ^
              t[0][w >> 56];
        buf += 8;
        len -= 8;
    }
    while (len-- > 0)
        crc = (uint16_t)((crc << 8) ^ t[0][((crc >> 8) ^ *buf++) & 0xff]);
    return crc;
}

/* ------------------------------------------------------------------ */
/* PMULL fold path, len >= 64.                                         */
/* ------------------------------------------------------------------ */
static uint16_t crc16_pmull(const unsigned char *buf, size_t len) {
    size_t ngroups = len >> 6;
    uint64x2_t x0 = crc16_load16(buf +  0);
    uint64x2_t x1 = crc16_load16(buf + 16);
    uint64x2_t x2 = crc16_load16(buf + 32);
    uint64x2_t x3 = crc16_load16(buf + 48);
    const unsigned char *p = buf + 64;

    for (size_t i = 1; i < ngroups; i++) {
        x0 = crc16_fold(x0, crc16_load16(p +  0));
        x1 = crc16_fold(x1, crc16_load16(p + 16));
        x2 = crc16_fold(x2, crc16_load16(p + 32));
        x3 = crc16_fold(x3, crc16_load16(p + 48));
        p += 64;
    }

    uint64x2_t t = x3;
    t = crc16_merge(t, x2, CRC16_EG128L, CRC16_EG128H);
    t = crc16_merge(t, x1, CRC16_EG256L, CRC16_EG256H);
    t = crc16_merge(t, x0, CRC16_EG384L, CRC16_EG384H);

    uint16_t head = crc16_reduce128(crc16_mul_x16(t));
    return crc16_slice8(p, len - (size_t)(p - buf), head);
}

/* Public entry point. Callers must have gone through crc16_aarch64_init()
 * first (which also builds the tables used by the len < 64 path). */
uint16_t crc16_aarch64_pmull(const char *buf, int len) {
    if (len < 64)
        return crc16_slice8((const unsigned char *)buf, (size_t)len, 0);
    return crc16_pmull((const unsigned char *)buf, (size_t)len);
}

#else /* !(__aarch64__ && __ARM_FEATURE_CRYPTO && little-endian) */

/* Degraded build (no crypto extension at compile time): never activate.
 * Keeping the symbols makes crc16.c link on any toolchain configuration. */
int crc16_aarch64_init(void) {
    return 0;
}

uint16_t crc16_aarch64_pmull(const char *buf, int len) {
    (void)buf; (void)len;
    return 0;
}

#endif
