# Redis 7.0.15 CRC16 (XMODEM) optimization for aarch64 Kunpeng

消除 Redis cluster 模式下 `keyHashSlot()` → `crc16()` 的逐字节查表热点，
目标：`redis-benchmark -c 200 -t get -P 16`（cluster 模式）吞吐 +3% 以上。

## 内容

```
src/crc16_aarch64.c    加速实现（自包含，无 Redis 头依赖）
patches/0001-*.patch   crc16.c dispatcher（crc16_init/crc16_fast）
patches/0002-*.patch   cluster.c keyHashSlot 切换到 crc16_fast
patches/0003-*.patch   server.h 声明（非 aarch64 编译为直通宏）
patches/0004-*.patch   server.c main() 启动时探测一次
patches/0005-*.patch   Makefile：仅 aarch64 链接 crc16_aarch64.o（+crypto 编译）
verify/verify_crc16_opt.py  位级仿真器（写代码前验证全部数学）
verify/crc16_test.c         等效性 + 微基准独立测试台
tests/slot_equiv_test.py    双服务器 CLUSTER KEYSLOT 全量对比
tests/cluster_bench.py      cluster 吞吐对比（ABBA 交替消除漂移）
.github/workflows/verify.yml  GitHub Actions ARM runner 全流程门禁
```

## 算法

两条子路径，一个入口 `crc16_aarch64_pmull()`：

1. **len < 64：slice-by-8**。benchmark 默认 key（`key:{tag}:__rand_int__`）
   全部短于 64B，这是热点的真正路径。8 字节/迭代，打破逐字节
   ~5 cycle 串行依赖链。表与 Redis 自带 crcspeed16little 代数相同：
   `T[k][b] = rem(b·x^(8k+16)) mod P`。
2. **len ≥ 64：PMULL 折叠**。4×128b 累加器按 64B 步进折叠
   （K_LO=rem(x^512), K_HI=rem(x^576)），端局三路合并
   （x^128/x^256/x^384），×x^16 后 80→64→48→32 链式归约，
   Barrett（μ=⌊x^47/P⌋）到 16bit。尾部 ≤63B 走 slice-8。

## 验证链（每一层独立可复核）

| 层 | 工具 | 结论 |
|---|---|---|
| 数学 | `verify_crc16_opt.py`：锚点=第一性原理复现 ISA-L T10DIF 硅上常量；slice-8 与 PMULL 管线 vs 参考实现（64..8192B × 4 模式） | ALL PASS |
| C 实现 | `crc16_test.c`：0..4096 全长度 × 5 模式 + 64B 边界 2 万组 + 非对齐 | CI 门禁 |
| 服务器 | `slot_equiv_test.py`：vanilla vs patched 双活对比 CLUSTER KEYSLOT | CI 门禁 |
| 吞吐 | `cluster_bench.py`：1 节点 cluster，-P 16，ABBA 四轮交替 | CI 报告 |

## 平台门控（需求第 2 条）

- 编译期：`src/Makefile` 仅在 `uname -m == aarch64` 时链接加速对象；
  x86 及其他平台二进制逻辑零变化（`crc16_fast` 宏直通 `crc16`）。
- 运行期：`crc16_aarch64_init()` 三重门——
  `__ARM_FEATURE_CRYPTO`（编译）+ `HWCAP_PMULL`（硬件）+
  **Kunpeng 厂商检测**（/proc/cpuinfo CPU implementer 0x48）。
  等效性/性能验证用环境变量 `REDIS_CRC16_ANY_PMULL_CPU=1` 放开厂商门
  （位等效性与厂商无关，仅取决于 PMULL 语义；该变量只能放宽，不能
  在鲲鹏上关闭加速）。
- 非 aarch64 / 无 PMULL：自动回落原表驱动循环，逐位相同。

## 使用

```bash
# 生成 patched 源码树
tar xzf redis-7.0.15.tar.gz && cd redis-7.0.15
for p in /path/to/patches/0*.patch; do patch -p1 < "$p"; done
cp /path/to/src/crc16_aarch64.c src/
make -j$(nproc)

# 验证（非鲲鹏机器需环境变量放开厂商门）
gcc -O2 -march=armv8-a+crypto \
    -o crc16_test verify/crc16_test.c src/crc16_aarch64.c
REDIS_CRC16_ANY_PMULL_CPU=1 ./crc16_test

# 基准（需求场景：cluster server + 普通客户端，全 16B key 哈希）
./src/redis-server --cluster-enabled yes ...  # 单节点全槽
./src/redis-benchmark -h <host> -c 200 -t get -n <N> -P 16
```
