# GLM-5.3-Flash 在 8×A800 (sm_80) 上的真实部署笔记

> 状态：**已成功部署并对外服务**（当前生产配置见 §5.12；首次跑通历程见 §5.10-5.11）。本文档为完整工程实录。

## 1. 问题定性（为什么不能开箱即用）

官方镜像 `vllm/vllm-openai:glm53-flash`（vLLM dev20051, FlashInfer 0.6.17,
torch 2.13.0+cu130）在 A800 上**首启即失败**，报错：

```
ValueError: No valid attention backend found for cuda with
AttentionSelectorConfig(head_size=512, ..., use_mla=True, use_sparse=True, ...)
Reasons: {FLASH_ATTN_MLA: [sparse not supported, compute capability not supported...],
 FLASHMLA: [FlashMLA Sparse is only supported on Hopper and Blackwell DC devices.],
 FLASHINFER_MLA: [requires qk_nope_head_dim in [64,128,192], but got 256],
 TRITON_MLA: [sparse not supported],
 FLASHINFER_MLA_SPARSE_SM90: [requires ... SM90 ...], ...}
```

定性：**FP8 权重侧没问题**（`MarlinFP8ScaledMMLinearKernel` 在 sm_80 正常选中，
`--moe-backend marlin` 生效）；卡死点是 **NoPE 稀疏 MLA（DSA 式 indexer）的注意力后端**，
全部候选都是 Hopper/Blackwell 专属。这与官方 "supports NVIDIA Hopper and newer" 一致，
也与社区调研结论一致（vllm#35021 记录 SM80 缺稀疏后端；PR #38476/#47629 提供
`TRITON_MLA_SPARSE` Triton 后端；GLM-5.2 曾凭该 PR 在 8×A800 部署成功）。

## 2. 选型决策

| 方案 | 评估 |
|---|---|
| 官方 glm53-flash + 移植 TRITON_MLA_SPARSE | ✅ 采用。模型栈完整、只需动注意力侧 ~7 文件 |
| v0.28.0 官方版为基座 | ❌ 同样没有 Glm5Next 模型栈 |
| v0.26.0-glm52-sm80 直接用 | ❌ 不支持 glm5_next 架构（只到 GLM-5.2） |

用户确认选择方案一（2026-08-27）。

## 3. 补丁移植实录（核心章节）

方法论照搬 `vllm_glm_5.2/glm52-a800-vllm-deploy`（PR #47629 九文件纯 Python 覆盖），
对 dev20051 新代码基逐文件适配：

### 3.1 文件清单（overlay：`_port/patches_glm53_sm80/` → 镜像 `Dockerfile.glm53-sm80`）

| 文件 | 来源 | 改动 |
|---|---|---|
| `ops/triton_mla_sparse_kernel.py` | PR#47629 原样 | 无（纯 Triton，依赖在新树同样存在） |
| `ops/mqa_logits_triton.py` | PR#47629 原样 | 无（新树的 `triton_fp8_mqa_logits.py` 是 ROCm 变体，不通用） |
| `backends/mla/xpu_mla_sparse.py` | 新树上游版 + 嫁接 | +`split_decodes_and_prefills` 导入；Metadata 增加 decode/prefill 计数字段与 seq_lens（统一 MLA forward 与 DSA indexer 的消费缺口，与当年 GLM-5.2 团队补的同款） |
| `backends/mla/triton_mla_sparse.py` | PR#47629 后端 + 适配 | head-size 白名单 `[512,576]`；576 走 split-KV 快核（原样），**512/NoPE（GLM-5.3）分派到上游通用接口 `triton_bf16_mla_sparse_interface(block_dpe=0)`** |
| `backends/registry.py` | 新树上游版 + 枚举 | 注册 `TRITON_MLA_SPARSE` |
| `platforms/cuda.py` | 新树上游版 + 候选表 | else 分支两处 NoPE 候选列表尾部追加该后端 |

### 3.2 关键发现（文档价值）

1. **上游通用 Triton 稀疏核已支持 NoPE**：`ops/xpu_mla_sparse.py::triton_bf16_mla_sparse_interface`
   明确有 `block_dpe: int = 64` 参数且注释 "Set to 0 when q/kv contain only the nope
   latent" —— 说明上游在为 DSv4/GLM-5.x NoPE 形状铺路，只是 XPU 后端没把参数传进去。
   我们的 `_forward_bf16_kv` 按 head_size 分派后即可正确服务 512。
2. **PR 快核是形状特化的**：`_DIM_QK=576(512+64)` 硬断言，直接复用会在 GLM-5.3 上炸。
   保留它给 512+64 rope 布局（未来 DSv3.2/GLM-5.2 类），是免费的速度优化路径。
3. **新树的 XPU 基类缺计数字段**：`mla_attention.py` 无条件读取
   `attn_metadata.num_decode_tokens/.num_prefills`（可选性判空），缺失即 AttributeError，
   与 0.26.0 时代 GLM-5.2 团队踩的坑完全同构 → 必须嫁接。
4. 当年"indexer 两处 has_deep_gemm 修复"在 dev20051 **已不需要**：新 indexer 重写为
   `_supports_varlen_paged_mqa_logits()` 门控 + Triton 回退，sm80 无 deep_gemm 自动走 Triton。

### 3.3 构建与自检

```bash
DOCKER_BUILDKIT=0 docker build -t vllm/vllm-openai:glm53-flash-sm80 -f Dockerfile.glm53-sm80 .
```

静态自检全绿：6 文件 py_compile 通过；枚举解析→backend class 加载 OK；
`is_sparse=True`、head sizes `[512,576]`。（注意 buildx 的 activity 文件写宿主 HOME 会
遇到只读 FS，需 legacy builder。）

## 4. 部署命令要点（相对官方 recipe 的 A800 差异）

在 `deploy_glm53_flash_a800.sh` 中：

| 设置 | 原因 |
|---|---|
| 镜像 → `glm53-flash-sm80` | 补丁叠加层 |
| `-e VLLM_ATTENTION_BACKEND=TRITON_MLA_SPARSE` | 显式钉死后端（与 GLM-5.2 成功配置一致） |
| `--attention-config '{"sparse_mla_force_mqa": true}'` | ⚠️ 必填。否则真实 prefill 崩在 forward_mha（GLM-5.2 血泪教训）。日志佐证："sparse MLA will use the top-k MQA path only (no dense-MHA prefill)" |
| `--moe-backend marlin` + `VLLM_TEST_FORCE_FP8_MARLIN=1` | sm80 FP8 模拟 |
| KV dtype 保持默认 BF16 | FP8 KV 需 SM90+ |
| 其余 = 官方 recipe | TP8 / EP / MTP5 / glm47 / glm45 / 8192 batch tokens |

## 5. 运行记录（真实时间线）

| 时刻 | 事件 |
|---|---|
| 13:28 | 部署 v1（补丁镜像）启动 |
| 13:29 | 架构解析 + 后端选择全部通过（v1 阵亡点已越过） |
| 13:30 | `TRITON_MLA_SPARSE` + MARLIN Fp8 MoE 日志确认，62 分片开载 |
| 13:38 | 权重加载完成（62/62，≈8.5 min，~0.63 GiB/s，TP8 并行拉流） |
| 13:40 | **第二轮踩坑**：`kpool_compress.py::_fwht_quant_kernel` Triton 编译报 `type fp8e4nv not supported` |

### 5.1 坑 2：GLM-5.3 indexer 全链 FP8 vs SM80 Triton

- 定性：GLM-5.3 的 DSA indexer（k-pool 压缩/门控）**系统性使用原生 FP8**——
  query 走 FWHT-128→e4m3 量化、K 池缓存按 e4m3 字节+fp32 scale 交错布局存储。
  上游 `glm5next/nvidia` 全线假设 Hopper+（SM89+ 的 Triton 才有 fp8e4nv 类型）。
- 解法（沿用 donor PR 的 uint8+LUT 惯例，对称补写侧）：
  内核一律**输出 BF16 暂存**，落盘用 `Tensor.to(float8_e4m3fn)`
  （PyTorch 软件编码，任意架构可用，字节与原生编码完全一致——已探针验证）。
- 改动点（overlay `kpool_compress.py`，全文件仅 3 处写侧）：
  1. `_fwht_quant_kernel` + `fwht128_quant_fp8()`：BF16 暂存 → torch cast
  2. `_kpool_softmax_rotate_write_cache_kernel` + wrapper：BF16 暂存行 →
     `index_copy_` 字节散射进 uint8 页缓存（scale 的 fp32 布局不变）
  3. `_kpool_decode_update_batched_kernel` + wrapper：原子计数分配暂存槽 →
     完成后统一散射（支持一次 launch 内多 pool 补全）
- 探针（`_port/probe_kpool.py`，A800 实测）：**PROBE PASS**
  - A FWHT+量化 vs torch 参考链：字节 100% 匹配，scale 误差 0
  - B prefill 压缩写缓存：4096 字节 0 失配，scale 误差 0
  - C decode pool 补全：写入与 scale 布局正确
- 注意：读侧无恙——donor `mqa_logits_triton.py` 本来就用 uint8+LUT 解码。

### 5.2 部署 v2（镜像 e857b9ae7c4d）

13:46 起跑，进入 TileLang JIT → warmup 阶段（后续在此追加）。

### 5.3 坑 3：deep_gemm paged MQA logits 硬拒 SM80（v2 实测 13:57）

- 现象：62/62 权重加载完成后、`profile_cudagraph_memory` 阶段崩溃：
  `AssertionError (/workspace/.deps/deepgemm-src/csrc/apis/attention.hpp:270):
  Unsupported architecture`，抛自 `get_paged_mqa_logits_metadata`。
- 根因：`vllm/utils/import_utils.py` 的 `has_deep_gemm()` 是**纯 import 探测**——
  镜像内 vendored deep_gemm 可正常 import，于是 `indexer.py` 旧门
  `is_cuda() and has_deep_gemm()` 在 A800 上放行，直到 CUDA 内核层面才炸。
  与 GLM-5.2 当年"vendored deep_gemm 误判"同源、不同位置的第三个变体。
- 修复（`_port/patches_glm53_sm80/vllm/v1/attention/backends/mla/indexer.py`）：
  metadata 门收紧为架构感知的 `is_deep_gemm_supported()`
  （= `VLLM_USE_DEEP_GEMM and has_deep_gemm() and platform.support_deep_gemm()`，
  SM80 判 False → 走 Triton 路径）。同文件 653 行的 varlen 门本就要求
  Blackwell 家族（`is_device_capability_family(100)`），无需改。
- 连带排查：全库 grep 确认 deep_gemm 消费点仅 indexer.py + 
  `model_executor/layers/sparse_attn_indexer.py` 两处；后者还有
  (a) `__init__` 的 `not has_deep_gemm()` 硬 raise、(b) prefill 非分页 logits
  `fp8_fp4_mqa_logits`、(c) decode 分页 logits `fp8_fp4_paged_mqa_logits`——
  均为 SM80 上的必炸点（cudagraph warmup 会依次踩到）。
- 修复（`_port/patches_glm53_sm80/vllm/model_executor/layers/sparse_attn_indexer.py`）：
  新增 `_sm80_triton_logits_enabled()`（cuda & major==8 & 非 deep_gemm），
  init 守卫放行 SM80；prefill/decode 两个 logits 调用点各插一个 elif 分支，
  换名调用 donor 的 `fp8_mqa_logits_triton` / `fp8_paged_mqa_logits_triton`
  ——两者与 deep_gemm API 参数完全镜像，纯 drop-in。分页内核自管调度，
  不消费 deep_gemm 的 scheduler metadata。

### 5.4 donor Triton logits 内核正确性验证（`_port/probe_logits.py`）

- **LOGITS PROBE PASS**（A800 实测）：
  - 非分页（prefill 路径）：rel_err_max = **2.0e-3**（bf16 点积舍入噪声级）
  - 分页（decode 路径）：rel_err_max = **2.8e-2**，因果有效区 0 inf
- 内核语义两要点（探针构象必须遵守，否则会误报）：
  1. 内核第 339 行有 **ReLU 正门控**（`where(s>0,s,0)*w`）——模型设计的
     正向门控语义，oracle 必须同步；权重可负，输出可负。
  2. 分页内核是 **per-token 因果**（`k_offset <= q_offset`，verify 行只看
     自身位置之前），`clean_logits=False` 时越界列**不写**（留垃圾）。
     top_k_per_row 按每 token 行边界读，天然无视。
- 排查过程中的三次"假失败"皆为我方 oracle 的错（漏 ReLU、k_scale 双乘、
  mask 过宽），donor 内核自始至终正确——与 GLM-5.2 生产同源可信。

### 5.5 kpool 散射去 sync 化（已落地）

`write_mask.nonzero()` 会强制 host sync（仅 prefill 带 mask 路径）。
修法分两步落地：

- **§5.6（镜像 9dc2dde0d374）**：`kpool_compress.py` overlay 首次带入去 sync
  散射，全异步三步：`where(mask, loc, 0)` 把被 mask 行重定向到缓存槽 0 →
  `index_select` 读回该槽现值 → `where` 拼回 → 全量 `index_copy_`
  （被 mask 行等于原值写回，语义无损，零 host sync）。
- **§5.9（v5/v6 实测 refined）**：CUDA-graph capture 阶段发现上述方案含
  `index_select`/GPU 张量索引仍触发 D2H sync，统一改为纯算术预计算槽 0 最终
  字节（`has0/sel0/vals0`），prefill + decode-K + decode-S 三处同一终版；
  最终形态见 `kpool_compress.py` 的 async scatter 实现。

### 5.6 部署 v3（镜像 9dc2dde0d374）

新增 overlay：indexer.py（metadata 门收紧）+ sparse_attn_indexer.py
（守卫放行 + 双 logits 分支）+ kpool_compress.py（去 sync 散射）。
同时 deploy 脚本补挂 `/root/.cache/triton`、`/root/.cache/tilelang`
持久卷——重启免数分钟 JIT 重编译。起跑后日志守望中（后续追加）。

### 5.7 坑 4：GLM-5.3 真身 indexer 是 `sparse_attn_indexer_kpool.py`（v3 实测）

v3 起跑后在同一阶段崩 deep_gemm paged logits——但栈来自
`model_executor/layers/sparse_attn_indexer_kpool.py`（kpool 变体），
与 5.3 修的 DSA 标准版 `sparse_attn_indexer.py` 是**两个平行模块**；
GLM-5.3 的 `mla.py` 实际走 kpool 版。同构三触点（init 守卫 968 /
prefill 523 / decode 788），同法接线 donor 两个 Triton logits 内核。
kpool 的 `kv_cache_as_quant_view` FP8 分支同为 uint8 `[nb,bs,1,D+4]`，
donor 断言天然兼容。由此 overlay 增至 10 个文件。

### 5.8 坑 5：topk buffer 宽度 2176 ≠ metadata.topk_tokens 2048（v4 实测）

`assert token_indices.shape[1] == NUM_TOPK_TOKENS` 炸。链路真相：
- config：`index_topk=2048`，`index_kpool=4`；
- 模型侧分配 buffer 宽 = round_up(2048+(4-1), 128) = **2176**（尾块
  kpool-1 个不满池 token + 对齐 128 供 remap kernel 分块）；
- 我们 graft 的 `xpu_mla_sparse.forward_mqa` 传
  `NUM_TOPK_TOKENS=attn_metadata.topk_tokens`(=2048)。
模型侧注释明确"backend 必须按 `topk_indices.shape[1]` 动态读宽"——
改为 `NUM_TOPK_TOKENS=topk_indices.shape[1]` 即修。
（donor 0.26 同签名但 GLM-5.2 kpool 未启用故未触发。）

### 5.9 坑 6/7：CUDA-graph capture 内的 host sync 双杀（v5/v6 实测）

v5：`if bool(done.any())` + 布尔掩码索引（内部 nonzero）→
`cudaErrorStreamCaptureUnsupported`。
v6 第一版重定向修复仍炸：`cand0[r0]`（GPU 张量做索引）同样强制 D2H。
终版方案（prefill + decode-K + decode-S 三处）：
1. 未完成/mask 行钳位重定向到缓存槽 0；
2. 槽 0 的**最终字节**用纯算术预计算：
   `has0 = cand0.sum()`；`sel0 = (staged.int() * cand0[:,None]).sum(0)`
   （loc 唯一 ⇒ 至多一行命中槽 0）；`vals0 = old*(1-has0) + sel0*has0`；
3. 所有重定向行写 vals0 → 重复索引写相同字节，无论仲裁谁赢都正确。
探针复验 **PROBE PASS**（A 100% / B 0/4096 / C 过）。
教训：capture 内禁 `bool(t.any())`、`.item()`、`nonzero()`、布尔掩码
索引、GPU 张量索引——只允许纯张量运算。

### 5.10 部署 v7（镜像 ec445419668c 后继 tag）——成功 🚀

| 阶段 | 实测 |
|---|---|
| 权重加载 | 39.8 GiB/卡 ×8，~36 s（页缓存热） |
| 引擎 init 全程 | 335.5 s（含 TileLang JIT、autotune、图捕获） |
| Graph 捕获 | 86 s，3.46 GiB |
| KV cache | 18.39 GiB/卡 → **1,535,933 tokens**（1M ctx 并发 1.46×） |
| VRAM 稳态 | 76.7/80 GiB 每卡（GMU 0.90 + graphs） |

### 5.11 API 验证（test_api.sh + 真实调用）

- `/v1/models`：`zai-org/GLM-5.3-Flash`，max_model_len=1,048,576 ✓
- 数学 sanity：13×17 → content=`221`，reasoning=`13*17=221`，
  `reasoning_effort=low` 生效，finish=stop ✓
- 中文长文生成质量正常（量子计算短文，结构完整）
- **单流吞吐：61.9→72.2 tok/s**（含 TTFT；MTP×5 生效）

### 5.12 生产配置定稿（端口 8008，2025-08-27）

在 §5.10 基础上调整运行参数并重测（脚本 `deploy_glm53_flash_a800.sh` 默认值）：

| 变更 | v7（§5.10） | **生产（v8）** |
|---|---|---|
| 端口 | 8006 | **8008** |
| GMU | 0.90 | **0.95** |
| max_num_seqs | 256 | **32**（图捕获集缩小：86s/3.46GiB → 66s/2.71GiB） |
| max_model_len | 1,048,576 | 1,048,576（保持，KV 池容纳 1.91×） |

实测（同机同镜像）：

- **KV 池 2,004,983 tokens**（GMU 0.95 后每卡 KV ≈22.4 GiB）
- VRAM 稳态 ~78.0/80 GiB 每卡
- 引擎 init ~5.5 min；sanity（13×17）响应 **0.43 s**
- **单流吞吐 79.0 tok/s**（含 TTFT，较 v7 的 72.2 提升约 9%）
- 1M 上下文为真实可跑满的单路上限（池 2.0M > 1M），并发满长上限 ~1.9 路

运维口径：重启 `./deploy_glm53_flash_a800.sh`（JIT 缓存持久化，跳过重编译）；
冒烟 `./test_api.sh 8008`；调试前台 `./run_debug.sh`（默认端口 8007，
已对齐生产参数，避免调试配置与生产行为不一致的误导）。
