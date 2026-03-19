## ADDED Requirements

### Requirement: Load LLM weights from GGUF file
The system SHALL load all 311 tensors from `./models/Fun-ASR-Nano-2512-llm-Q5_K_M.gguf` into a GGML CUDA backend buffer using `gguf_init_from_file` with `no_alloc=true`, followed by manual `fseek`/`fread` per tensor and `ggml_backend_tensor_set`.

#### Scenario: All LLM tensors loaded
- **WHEN** `load_llm(path, backend)` is called with a valid GGUF path
- **THEN** exactly 311 tensors are loaded with shapes matching: `token_embd.weight [1024,151936]`, `output_norm.weight [1024]`, `output.weight [1024,151936]`, and per-layer tensors `blk.{0..27}.attn_q/k/v/output/q_norm/k_norm/attn_norm/ffn_norm/ffn_gate/ffn_up/ffn_down.weight`

#### Scenario: Invalid path returns error
- **WHEN** `load_llm` is called with a non-existent file path
- **THEN** the function returns nullptr and logs an error without aborting

### Requirement: Load audio encoder weights from GGUF file
The system SHALL load all 950 tensors from `./models/Fun-ASR-Nano-2512-audio-mmproj-mtmd-F16.gguf` into the GGML CUDA backend, including encoder blocks `a.blk.{0..50}`, tp blocks `a.tp.blk.{0..19}`, norms, MLP projector `mm.a.linear1/2`, and adaptor blocks `mm.a.blk.{0..1}`.

#### Scenario: All audio tensors loaded
- **WHEN** `load_mmproj(path, backend)` is called with a valid GGUF path
- **THEN** exactly 950 tensors are loaded; `a.blk.0.self_attn.linear_qkv.weight` has shape `[560,1536]`; all other encoder block QKV weights have shape `[512,1536]`; `mm.a.linear1.weight` has shape `[512,2048]`; `mm.a.linear2.weight` has shape `[2048,1024]`

#### Scenario: CUDA backend allocation succeeds
- **WHEN** a CUDA device is available and `ggml_backend_cuda_init(0)` succeeds
- **THEN** all tensor data is allocated on GPU memory; `ggml_backend_buffer_get_size` returns nonzero

### Requirement: Fallback to CPU backend when CUDA unavailable
The system SHALL fall back to `ggml_backend_cpu_init()` when `ggml_backend_cuda_init(0)` fails.

#### Scenario: CPU fallback on CUDA failure
- **WHEN** no CUDA device is available
- **THEN** weights are loaded onto CPU memory; inference proceeds without aborting; a warning is logged
