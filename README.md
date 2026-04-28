# triton-kernel-optimization

A comprehensive AI coding agent skill for accelerating PyTorch code with [OpenAI Triton](https://triton-lang.org/) GPU kernels.

## What is this?

This is a **skill file** designed for AI coding assistants (Claude Code, etc.). When loaded, it gives the AI agent deep, structured knowledge of how to write, optimize, and debug Triton kernels — turning it from a generic coding assistant into a Triton optimization specialist.

The skill was distilled from a real-world production optimization journey: accelerating [Kimi-style attention residuals](https://arxiv.org/abs/2501.12599) in a Megatron-LM training framework, where a 5-commit iterative process replaced naive PyTorch with fully fused Triton kernels (forward + backward), eliminated host-side `torch.cat`/`torch.stack` overhead, and achieved significant end-to-end speedup.

## What's Covered

### Complete Optimization Workflow (12 chapters)

| # | Topic | What You Get |
|---|-------|-------------|
| 1 | **Core Patterns** | Kernel structure, masking, precision, stride-based addressing — the non-negotiables |
| 2 | **Python → Triton Workflow** | 5-phase playbook: profile → fuse forward → Triton backward → autograd → eliminate torch.cat |
| 3 | **Fused Normalization** | RMSNorm, LayerNorm, GroupNorm — standalone and fused into attention epilogues |
| 4 | **FlashAttention v2** | Online softmax, causal masking, GQA head routing, multi-stream accumulators |
| 5 | **Tiled GEMM & Autotune** | Grouped tile ordering for L2 locality, autotune config design |
| 6 | **Persistent Matmul** | Warp specialization, TMA descriptors (Hopper/SM90+), software pipelining |
| 7 | **Memory-Efficient Patterns** | Seed-based Philox PRNG, activation recomputation, fused epilogues |
| 8 | **Quantized GEMM** | Block-scaled mxfp4/mxfp8, `tl.dot_scaled`, dequantize fallback |
| 9 | **Sequential Stateful Processing** | Register-resident mutable state for LRU routing, sequential assignment |
| 10 | **Dynamic Launcher** | Inference-time tile selection without autotune warmup |
| 11 | **Correctness Verification** | Reference testing, backward gradcheck, boundary cases, benchmarking |
| 12 | **Optimization Order** | 8-step priority sequence + quick-reference checklist + common mistakes |

### Also Includes

- **Bottleneck classification table** (memory-bound / compute-bound / launch-bound / host-bound)
- **NCU profiling guidance** with specific metric names
- **GPU hardware quick reference** (H100, A100, RTX 4090 specs)
- **Arithmetic intensity thresholds** for roofline analysis
- **8 common mistakes** with fixes

## Usage

### With Claude Code / AI Coding Agent

Place the skill where your agent discovers skills:

```bash
# Claude Code (user-level)
cp SKILL.md ~/.claude/skills/triton-kernel-optimization/SKILL.md

# Or project-level
cp SKILL.md .claude/skills/triton-kernel-optimization/SKILL.md
```

The agent will automatically load this skill when you ask it to optimize PyTorch code with Triton, write custom CUDA kernels, or fuse operators.

**Example prompts that trigger this skill:**

- "Profile this PyTorch module and rewrite the bottleneck as a Triton kernel"
- "Fuse these RMSNorm + attention + softmax ops into a single Triton kernel"
- "Write a Triton backward kernel for this custom attention"
- "This torch.cat is showing up in profiling, help me eliminate it"
- "Write a FlashAttention v2 kernel with causal masking and GQA"

### As a Human Reference

The `SKILL.md` is also a self-contained reference guide. Read it top-to-bottom for a complete Triton optimization curriculum, or jump to specific sections:

- New to Triton? Start with **Section 1 (Core Patterns)** and **Section 2 (Workflow)**
- Optimizing attention? Jump to **Section 4 (FlashAttention)**
- Debugging numerical issues? Check **Section 11 (Verification)** and **Common Mistakes**
- Need to choose tile sizes? See **Section 10 (Dynamic Launcher)**

## Origin

This skill was synthesized from three sources:

1. **Production optimization experience** — A 5-commit journey accelerating Kimi attention residuals in Megatron-LM, covering: fused Triton forward → Triton backward with LSE saving → autograd integration → split-input kernels eliminating torch.cat → tensor accumulation replacing list operations.

2. **Triton kernel pattern library** — 9 specialized guides covering FlashAttention v2, persistent warp matmul, fused normalizations, quantized block-scaled GEMM, memory-efficient patterns, fused epilogue kernels, sequential stateful blocks, dynamic launcher tiling, and general GPU kernel optimization.

3. **Triton optimization methodology** — A systematic framework organized by bottleneck type (memory-bound / compute-bound / latency-bound), with detailed NCU profiling strategies, roofline analysis, and an 8-category optimization taxonomy covering tile tuning, memory access, pipelining, compute optimization, parallelism, register management, special data paths, and engineering diagnostics.

## Requirements

- **Triton** >= 2.1
- **GPU**: SM70+ (Volta) or CDNA2+ (AMD)
- **PyTorch**: for autograd integration and benchmarking
- Some features (persistent matmul, TMA) require SM90+ (Hopper)
- Quantized GEMM `tl.dot_scaled` requires SM100+ (Blackwell)

## License

MIT
