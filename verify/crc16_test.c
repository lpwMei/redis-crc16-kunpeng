/* Standalone equivalence + performance harness for the Redis CRC16 patch.
 *
 * Build (on aarch64):
 *   gcc -O2 -march=armv8-a+crypto -o crc16_test crc16_test.c crc16_aarch64.c
 * Run:
 *   ./crc16_test            # equivalence over the full corpus
 *   ./crc16_test bench      # micro-benchmark, reference vs accelerated
 *
 * The reference re-implements src/crc16.c verbatim (public domain, same
 * interface) so this harness has no Redis build dependency.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

/* ---- reference: verbatim from Redis src/crc16.c ---- */
static const uint16_t crc16tab[256]= {
    0x0000,0x1021,0x2042,0x3063,0x4084,0x50a5,0x60c6,0x70e7,
    0x8108,0x9129,0xa14a,0xb16b,0xc18c,0xd1ad,0xe1ce,0xf1ef,
    0x1231,0x0210,0x3273,0x2252,0x52b5,0x4294,0x72f7,0x62d6,
    0x9339,0x8318,0xb37b,0xa35a,0xd3bd,0xc39c,0xf3ff,0xe3de,
    0x2462,0x3443,0x0420,0x1401,0x64e6,0x74c7,0x44a4,0x5485,
    0xa56a,0xb54b,0x8528,0x9509,0xe5ee,0xf5cf,0xc5ac,0xd58d,
    0x3653,0x2672,0x1611,0x0630,0x76d7,0x66f6,0x5695,0x46b4,
    0xb75b,0xa77a,0x9719,0x8738,0xf7df,0xe7fe,0xd79d,0xc7bc,
    0x48c4,0x58e5,0x6886,0x78a7,0x0840,0x1861,0x2802,0x3823,
    0xc9cc,0xd9ed,0xe98e,0xf9af,0x8948,0x9969,0xa90a,0xb92b,
    0x5af5,0x4ad4,0x7ab7,0x6a96,0x1a71,0x0a50,0x3a33,0x2a12,
    0xdbfd,0xcbdc,0xfbbf,0xeb9e,0x9b79,0x8b58,0xbb3b,0xab1a,
    0x6ca6,0x7c87,0x4ce4,0x5cc5,0x2c22,0x3c03,0x0c60,0x1c41,
    0xedae,0xfd8f,0xcdec,0xddcd,0xad2a,0xbd0b,0x8d68,0x9d49,
    0x7e97,0x6eb6,0x5ed5,0x4ef4,0x3e13,0x2e32,0x1e51,0x0e70,
    0xff9f,0xefbe,0xdfdd,0xcffc,0xbf1b,0xaf3a,0x9f59,0x8f78,
    0x9188,0x81a9,0xb1ca,0xa1eb,0xd10c,0xc12d,0xf14e,0xe16f,
    0x1080,0x00a1,0x30c2,0x20e3,0x5004,0x4025,0x7046,0x6067,
    0x83b9,0x9398,0xa3fb,0xb3da,0xc33d,0xd31c,0xe37f,0xf35e,
    0x02b1,0x1290,0x22f3,0x32d2,0x4235,0x5214,0x6277,0x7256,
    0xb5ea,0xa5cb,0x95a8,0x8589,0xf56e,0xe54f,0xd52c,0xc50d,
    0x34e2,0x24c3,0x14a0,0x0481,0x7466,0x6447,0x5424,0x4405,
    0xa7db,0xb7fa,0x8799,0x97b8,0xe75f,0xf77e,0xc71d,0xd73c,
    0x26d3,0x36f2,0x0691,0x16b0,0x6657,0x7676,0x4615,0x5634,
    0xd94c,0xc96d,0xf90e,0xe92f,0x99c8,0x89e9,0xb98a,0xa9ab,
    0x5844,0x4865,0x7806,0x6827,0x18c0,0x08e1,0x3882,0x28a3,
    0xcb7d,0xdb5c,0xeb3f,0xfb1e,0x8bf9,0x9bd8,0xabbb,0xbb9a,
    0x4a75,0x5a54,0x6a37,0x7a16,0x0af1,0x1ad0,0x2ab3,0x3a92,
    0xfd2e,0xed0f,0xdd6c,0xcd4d,0xbdaa,0xad8b,0x9de8,0x8dc9,
    0x7c26,0x6c07,0x5c64,0x4c45,0x3ca2,0x2c83,0x1ce0,0x0cc1,
    0xef1f,0xff3e,0xcf5d,0xdf7c,0xaf9b,0xbfba,0x8fd9,0x9ff8,
    0x6e17,0x7e36,0x4e55,0x5e74,0x2e93,0x3eb2,0x0ed1,0x1ef0
};

static uint16_t crc16_ref(const char *buf, int len) {
    int counter;
    uint16_t crc = 0;
    for (counter = 0; counter < len; counter++)
            crc = (crc<<8) ^ crc16tab[((crc>>8) ^ *(const unsigned char *)buf++)&0x00FF];
    return crc;
}

/* ---- candidate under test ---- */
uint16_t crc16_aarch64_pmull(const char *buf, int len);
int crc16_aarch64_init(void);

/* xorshift PRNG so results are reproducible across runs/platforms */
static uint64_t rng_state = 0x9E3779B97F4A7C15ULL;
static uint64_t rnd64(void) {
    uint64_t x = rng_state;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    return rng_state = x;
}

static int check_equivalence(void) {
    size_t maxlen = 4096;
    unsigned char *buf = malloc(maxlen);
    long checked = 0;
    int failures = 0;

    if (!crc16_aarch64_init()) {
        printf("FATAL: accelerated path not available on this host\n");
        return 1;
    }

    /* 1. exhaustive lengths with several patterns */
    for (size_t len = 0; len <= maxlen; len++) {
        /* pattern set: random, zeros, ff, alternating, walking bit */
        memset(buf, 0x00, len);                    /* zeros   */
        if (crc16_ref((char*)buf, (int)len) != crc16_aarch64_pmull((char*)buf, (int)len)) { printf("FAIL zeros len=%zu\n", len); failures++; }
        memset(buf, 0xff, len);                    /* ff      */
        if (crc16_ref((char*)buf, (int)len) != crc16_aarch64_pmull((char*)buf, (int)len)) { printf("FAIL ff len=%zu\n", len); failures++; }
        for (size_t i = 0; i < len; i++) buf[i] = (unsigned char)(i & 1 ? 0x55 : 0xaa);
        if (crc16_ref((char*)buf, (int)len) != crc16_aarch64_pmull((char*)buf, (int)len)) { printf("FAIL alt len=%zu\n", len); failures++; }
        for (size_t i = 0; i < len; i++) buf[i] = (unsigned char)(0x01 << (i & 7));
        if (crc16_ref((char*)buf, (int)len) != crc16_aarch64_pmull((char*)buf, (int)len)) { printf("FAIL walk len=%zu\n", len); failures++; }
        for (size_t i = 0; i < len; i++) buf[i] = (unsigned char)rnd64();
        if (crc16_ref((char*)buf, (int)len) != crc16_aarch64_pmull((char*)buf, (int)len)) { printf("FAIL rand len=%zu\n", len); failures++; }
        checked += 5;
        if (failures > 20) { printf("too many failures, aborting\n"); break; }
    }

    /* 2. adversarial around the 64B boundary */
    for (int trial = 0; trial < 20000; trial++) {
        int len = 55 + (int)(rnd64() % 20);
        for (int i = 0; i < len; i++) buf[i] = (unsigned char)rnd64();
        if (crc16_ref((char*)buf, len) != crc16_aarch64_pmull((char*)buf, len)) {
            printf("FAIL boundary len=%d\n", len); failures++;
        }
        checked++;
    }

    /* 3. unaligned buffers (PMULL path must not assume alignment) */
    for (int off = 0; off < 16; off++) {
        for (int len = 60; len <= 76; len++) {
            for (int i = 0; i < len; i++) buf[off + i] = (unsigned char)rnd64();
            if (crc16_ref((char*)buf + off, len) != crc16_aarch64_pmull((char*)buf + off, len)) {
                printf("FAIL unalign off=%d len=%d\n", off, len); failures++;
            }
            checked++;
        }
    }

    /* 4. known answers */
    struct { const char *s; uint16_t v; } kat[] = {
        {"", 0x0000}, {"123456789", 0x31C3}, {"key:__rand_int__", 0},
    };
    kat[2].v = crc16_ref(kat[2].s, (int)strlen(kat[2].s));
    for (size_t i = 0; i < sizeof(kat)/sizeof(kat[0]); i++) {
        uint16_t got = crc16_aarch64_pmull(kat[i].s, (int)strlen(kat[i].s));
        if (got != kat[i].v) { printf("FAIL kat '%s': %04x != %04x\n", kat[i].s, got, kat[i].v); failures++; }
        checked++;
    }

    free(buf);
    printf("equivalence: %ld cases, %s\n", checked, failures ? "FAIL" : "ALL PASS");
    return failures;
}

/* ---- micro-benchmark: lengths matching redis-benchmark traffic ---- */
static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static void bench(const char *name, uint16_t (*fn)(const char *, int),
                  const unsigned char *buf, int len, long iters) {
    /* warm */
    for (int i = 0; i < 1000; i++) fn((const char *)buf, len);
    double t0 = now_ms();
    uint16_t sink = 0;
    for (long i = 0; i < iters; i++)
        sink ^= fn((const char *)buf, len);
    double dt = now_ms() - t0;
    printf("  %-8s len=%-5d %8.1f Mops/s   (sink=%04x)\n",
           name, len, iters / (dt / 1e3) / 1e6, sink);
}

int main(int argc, char **argv) {
    if (argc > 1 && strcmp(argv[1], "bench") == 0) {
        static unsigned char big[1 << 20];
        if (!crc16_aarch64_init()) {
            printf("FATAL: accelerated path not available on this host\n");
            return 1;
        }
        for (size_t i = 0; i < sizeof(big); i++) big[i] = (unsigned char)(i * 131 + 17);
        long iters = 20000000;
        printf("micro-benchmark (lower name = faster impl)\n");
        bench("ref", crc16_ref, (const unsigned char *)"key:000123456789", 16, iters);
        bench("opt", crc16_aarch64_pmull, (const unsigned char *)"key:000123456789", 16, iters);
        iters = 2000000;
        bench("ref", crc16_ref, big, 64, iters);
        bench("opt", crc16_aarch64_pmull, big, 64, iters);
        bench("ref", crc16_ref, big, 128, iters);
        bench("opt", crc16_aarch64_pmull, big, 128, iters);
        iters = 200000;
        bench("ref", crc16_ref, big, 1024, iters);
        bench("opt", crc16_aarch64_pmull, big, 1024, iters);
        bench("ref", crc16_ref, big, 65536, iters / 10);
        bench("opt", crc16_aarch64_pmull, big, 65536, iters / 10);
        return 0;
    }
    return check_equivalence() ? 1 : 0;
}
