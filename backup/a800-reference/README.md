# GLM-5.3-Flash on 8×A800 (sm_80) — 部署套件

> 目标：在 SM80（Ampere，无原生 FP8 Tensor Core）的 8×A800 80GB 机器上，
> 基于 `vllm/vllm-openai:glm53-flash` 镜像部署原生 FP8 权重的
> **zai-org/GLM-5.3-Flash**（320B-total / 18B-active 多模态 MoE，KDA 混合线性注意力
> + NoPE 稀疏 MLA，288 专家 top-8 + 1 shared，1 个 MTP 草稿层，1M 上下文），
> 并以 OpenAI 兼容 API 对外服务。
>
> 官方 recipe 仅覆盖 Hopper/Blackwell；本项目把缺失的 sm_80 稀疏 MLA 路径
> 以纯 Python overlay 方式补齐（10 个补丁文件，见下文），已实测稳定服务。

## 1. 这套套件是什么

本仓库**不含模型权重**，只含三样东西：

1. **补丁层**（`_port/patches_glm53_sm80/`，10 个文件）：覆盖到 vLLM 镜像内的
   Python 源码补丁，让 sm_80 跑通该模型的稀疏 MLA / FP8 indexer 全链路；
2. **镜像构建脚本**（`Dockerfile.glm53-sm80`）：基座镜像 + 补丁层 = 自定义镜像
   `vllm/vllm-openai:glm53-flash-sm80`；
3. **部署/测试脚本**（`deploy_glm53_flash_a800.sh` 等）：容器编排、参数与冒烟测试。

## 2. 前置条件

| 项 | 要求 |
|---|---|
| GPU | 8× A800 80GB（sm_80），全部空闲；`--gpus all` 可用 |
| 驱动 | 实测 580.159.03 正常（支持 CUDA 13 的版本即可） |
| Docker | 任意较新版本；**不依赖 buildx**（用传统构建器，见第 4 节） |
| 磁盘 | 模型 ~306 GiB + `vllm_cache/`（Triton/TileLang/torch 编译缓存，数 GB） |
| 基座镜像 | `vllm/vllm-openai:glm53-flash`（vLLM dev20051，torch 2.13.0+cu130，triton 3.7.1，flashinfer 0.6.17，python 3.12） |

说明：官方镜像 `vllm/vllm-openai:glm53-flash` 需在能访问镜像源的机器
`docker pull`；离线环境用 `docker save / docker load` 迁移。

先自检 GPU 通道（部分环境宿主机 `nvidia-smi` 不可用，以容器内为准）：

```bash
docker run --rm --gpus all --entrypoint nvidia-smi vllm/vllm-openai:glm53-flash-sm80
# 镜像还没构建时，可用任意 CUDA 镜像替代验证 --gpus all 通道
```

## 3. 准备模型权重

从 HuggingFace 获取 `zai-org/GLM-5.3-Flash` 全量权重（62 个 safetensors 分片，
共 ≈306 GiB，标准 HF 目录结构），放到本机任意路径，例如
`/data/models/GLM-5.3-Flash`。目录内应能看到：

```
config.json
model-00001-of-00062.safetensors ... model-00062-of-00062.safetensors
model.safetensors.index.json
tokenizer.json / tokenizer_config.json / chat template 等词表文件
```

然后把路径写进**本地配置文件**（git 忽略，不会进仓库）：

```bash
cp env.local.example env.local
vi env.local          # 改 MODEL_HOST_PATH=/data/models/GLM-5.3-Flash
```

> `env.local` 承载所有本机差异（权重路径等）。仓库里只有
> `env.local.example` 模板，这是本套件可直接开源的原因之一。

## 4. 构建自定义镜像（一次性，约 1 分钟）

基座镜像在 sm_80 上启动该模型会在 attention backend 选择处直接失败
（所有稀疏 MLA 后端都要求 SM90+）。`Dockerfile.glm53-sm80` 把 10 个补丁
文件覆盖进镜像：

| 补丁文件（overlay 路径省略 `_port/patches_glm53_sm80/`） | 作用 |
|---|---|
| `platforms/cuda.py`、`v1/attention/backends/registry.py` | 注册 `TRITON_MLA_SPARSE` 并加入 SM80 候选列表 |
| `v1/attention/backends/mla/triton_mla_sparse.py` | 后端入口；576 rope 布局走 PR 快路径，GLM-5.3 的 512 NoPE 走通用接口（block_dpe=0） |
| `v1/attention/backends/mla/xpu_mla_sparse.py` | 上游类 + decode/prefill 计数字段 graft + topk 宽度动态化（kpool 下 buffer 2176 ≠ config 2048） |
| `v1/attention/ops/triton_mla_sparse_kernel.py` | PR #47629 的稀疏 MLA Triton 内核（原样） |
| `v1/attention/ops/mqa_logits_triton.py` | PR 的 FP8 MQA logits Triton 内核（LUT 解码，sm80 无 fp8e4nv 的替代） |
| `v1/attention/backends/mla/indexer.py` | deep_gemm 门收紧为架构感知的 `is_deep_gemm_supported()`（vendored deep_gemm 在 Ampere 会运行时拒判） |
| `model_executor/layers/sparse_attn_indexer.py` | DSA indexer：init 守卫放行 SM80 + prefill/decode logits 切换 Triton 内核 |
| `model_executor/layers/sparse_attn_indexer_kpool.py` | 同上，GLM-5.3 实际使用的 **kpool 变体** indexer |
| `models/glm5next/nvidia/ops/kpool_compress.py` | FP8 落盘改 BF16 sidecar + torch 软件量化 + 字节散射（含 CUDA-graph 捕获安全的确定性重定向） |

构建（**必须用传统构建器**——部分环境 buildx 因 builder activity 文件只读
无法使用；`.dockerignore` 已排除缓存与参考源码，上下文很小）：

```bash
DOCKER_BUILDKIT=0 docker build -t vllm/vllm-openai:glm53-flash-sm80 \
    -f Dockerfile.glm53-sm80 .
```

构建成功会输出 `Successfully tagged vllm/vllm-openai:glm53-flash-sm80`。

## 5. 一键部署

```bash
./deploy_glm53_flash_a800.sh
```

脚本默认值（全部支持环境变量覆盖）：

| 变量 | 默认 | 说明 |
|---|---|---|
| `PORT` | 8008 | 对外端口（容器内固定 8000） |
| `GMU` | 0.95 | GPU 显存利用率 |
| `MAX_SEQS` | 32 | 最大并发序列 |
| `MAX_MODEL_LEN` | 1048576 | 上下文长度（KV 池 ~2.0M tokens，1M 单路真实可跑满） |
| `NUM_SPEC_TOKENS` | 5 | MTP 投机解码 token 数 |
| `TP` / `GPUS` | 8 / 0..7 | 张量并行 / 卡映射 |
| `IMAGE` | glm53-flash-sm80 | 使用的镜像 |
| `EXTRA_ARGS` | — | 追加 vLLM 参数 |

盯启动日志，关键里程碑（二次启动参考值；首次冷 NFS 加载权重 15~30+ 分钟）：

```bash
docker logs -f glm53-flash-8008
```

| 里程碑 | 参考耗时 | 日志关键字 |
|---|---|---|
| 权重加载 | ~36 s（页缓存热） | `Model loading took 39.8 GiB` |
| 引擎初始化（含 TileLang JIT、autotune） | ~5.5 min | `init engine ... took` |
| CUDA 图捕获 | ~66 s（2.71 GiB） | `Graph capturing finished` |
| **就绪** | — | **`Application startup complete.`** |

## 6. 验证

```bash
./test_api.sh 8008                     # 探活 + 数学 sanity（13*17→221）+ reasoning 解析
curl -s http://localhost:8008/v1/models | head -c 300
```

直接对话示例（OpenAI 兼容，含 reasoning 解析）：

```bash
curl -s http://localhost:8008/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm5.3-flash",
       "messages":[{"role":"user","content":"用三句话介绍卷积神经网络"}],
       "chat_template_kwargs":{"reasoning_effort":"low"}}' | python3 -m json.tool
```

实测（8×A800，2025-08）：**单流 79.0 tok/s**（含 TTFT）、KV 池
2,004,983 tokens、VRAM 稳态 ~78/80 GiB 每卡。

## 7. 日常运维

- **重启**：重复执行 `./deploy_glm53_flash_a800.sh`（脚本自动删旧容器；
  Triton/TileLang JIT 缓存已持久化到 `vllm_cache/`，重启跳过重编译）
- **停止**：`docker rm -f glm53-flash-8008`
- **换端口试跑**：`PORT=9000 ./deploy_glm53_flash_a800.sh`
- **小上下文快速验证**：`MAX_MODEL_LEN=32768 ./deploy_glm53_flash_a800.sh`
- **前台调试**（看全量日志，Ctrl+C 停）：`./run_debug.sh`

## 8. 文件说明

| 文件 | 用途 |
|---|---|
| `deploy_glm53_flash_a800.sh` | 一键生产部署（TEP8 + graph + MTP，端口 8008） |
| `run_debug.sh` | 前台调试启动（同配置，看全量日志） |
| `test_api.sh` | 冒烟测试（探活 + 数学 sanity + reasoning 解析检查） |
| `Dockerfile.glm53-sm80` | 基座镜像 + 补丁层 → 自定义镜像 |
| `_port/patches_glm53_sm80/` | 10 个镜像内补丁文件（见第 4 节） |
| `_port/probe_*.py` | GPU 探针：logits 内核正确性、kpool 量化散射字节级校验 |
| `env.local.example` | 本机配置模板（复制为 `env.local`，git 忽略） |
| `DEPLOY_NOTES_A800.md` | 完整工程实录：7 连坑根因与修复、探针数据、时间线 |

## 9. A800 (sm_80) 必备 workaround（与官方 recipe 的差异）

官方 recipe 仅验证 H100/H200/Blackwell/MI355X（"supports NVIDIA Hopper and newer"）。
A800 = sm_80 无 FP8/无 TMA/wgmma，必须做以下替换：

| 官方（Hopper+） | 本项目（sm_80） | 原因 |
|---|---|---|
| —（默认 attention/MoE 后端） | `--moe-backend marlin` + 环境变量 `VLLM_TEST_FORCE_FP8_MARLIN=1` | Triton block-FP8 MoE 要求 sm_89+；Marlin 走 weight-only FP8 模拟 |
| `--kv-cache-dtype fp8`（仅 Blackwell/H100 特例） | 不设，保持 BF16 KV | 该模型 FP8 KV cache 需要 SM90+；sm_80 更不可能 |
| GPU 映射随意 | `--gpus all` + `-e CUDA_VISIBLE_DEVICES=0..7` | vLLM 多进程 worker 在容器内 CUDA_VISIBLE_DEVICES 为空时会错用物理 id 查显存 |
| — | `VLLM_ENGINE_READY_TIMEOUT_S=3600` | 306 GiB 网络存储权重加载远超默认超时 |
| FlashInfer autotune | `--no-enable-flashinfer-autotune` | autotune 依赖的 FlashInfer kernel 假设 Hopper+ |

其余照抄官方 recipe：TP=GPU 数、`--enable-expert-parallel`、MTP
`num_speculative_tokens=5`、`--tool-call-parser glm47`、
`--reasoning-parser glm45`、`--enable-auto-tool-choice`、
`max-num-batched-tokens=8192`。

## 10. 显存账（为何必须 8 卡）

- 权重 ≈306 GiB / 8 卡 ≈ **38.3 GiB/卡**（实测加载 39.8 GiB）
- `gpu-memory-utilization 0.95` → 每卡预算 76 GiB → **KV 池 2,004,983 tokens**
- 4 卡方案：仅权重就 76.5 GiB/卡 > 物理上限，**不可行**

## 11. 部署记录与已知问题

见 `DEPLOY_NOTES_A800.md`：7 连坑的完整根因链（backend 缺失 → fp8e4nv →
deep_gemm×2 → topk 宽度 → CUDA-graph 内 host sync×2）、每步的 GPU 探针
校验数据、真实时间线与基准。

**许可**：Apache-2.0（见 `LICENSE`/`NOTICE`；`_port/patches_glm53_sm80/`
为 vLLM（Apache-2.0）衍生代码，含 PR #47629 社区内核）。GLM-5.3-Flash
模型权重由 Z.ai 单独分发，不在本仓库内。
