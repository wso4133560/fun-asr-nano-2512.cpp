## ADDED Requirements

### Requirement: Build prefill embedding sequence
The system SHALL construct the prefill embedding sequence with this exact token layout:
- **Prefix**: `<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|startofspeech|>` (tokenized via BPE)
- **Audio tokens**: adaptor output embeddings `[T', 1024]` directly (NOT via `ggml_get_rows`)
- **Suffix**: `<|endofspeech|>\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n` (tokenized via BPE)

The empty `<think>\n\n</think>\n\n` in the assistant prefix **disables Qwen3 thinking mode**, forcing direct transcription output. No task text is appended after `<|endofspeech|>` (user turn contains audio only).

#### Scenario: Prefill sequence concatenation
- **WHEN** a WAV file is processed end-to-end
- **THEN** prefill sequence equals `[text_prefix_embeds || audio_tokens || text_suffix_embeds]` with total length `n_prefix + T' + n_suffix`; all parts have dim 1024

#### Scenario: Audio tokens bypass token_embd lookup
- **WHEN** audio tokens are inserted into prefill
- **THEN** audio embeddings come directly from adaptor output, NOT from `ggml_get_rows(token_embd, ...)`

### Requirement: Execute Qwen3-0.6B prefill with KV cache population
The system SHALL run all 28 Qwen3 decoder layers on the full prefill sequence, populating KV cache for all positions.

#### Scenario: KV cache populated after prefill
- **WHEN** prefill completes on sequence of length L
- **THEN** KV cache contains K,V tensors for all L positions; `kv_used` equals L

#### Scenario: Prefill respects context length
- **WHEN** total prefill length L > 40960 (`qwen3.context_length`)
- **THEN** the system logs an error and returns empty result without crashing

### Requirement: Qwen3 GQA attention with QK-Norm and RoPE
Each Qwen3 decoder layer SHALL apply: RMSNorm (no bias, eps=1e-6) → Q projection `[1024,2048]` (16 heads×128) → per-head RMSNorm on Q using `attn_q_norm [128]` → K projection `[1024,1024]` (8 heads×128) → per-head RMSNorm on K using `attn_k_norm [128]` → V projection `[1024,1024]` → RoPE on Q and K (mode=neox, head_dim=128, freq_base=1e6) → GQA flash attention (16 Q heads, 8 KV heads) → output projection `[2048,1024]`.

#### Scenario: QK-Norm applied per head
- **WHEN** Q or K is computed in any layer
- **THEN** RMSNorm with `attn_q_norm_w [128]` / `attn_k_norm_w [128]` is applied per head (reshape to `[128, n_heads, T]`, norm over dim-128, reshape back) before RoPE

#### Scenario: GQA KV head expansion
- **WHEN** flash attention is computed
- **THEN** 8 KV heads are shared across 16 Q heads (ratio 2:1); `ggml_flash_attn_ext` handles this via GQA broadcasting

#### Scenario: RoPE frequency
- **WHEN** RoPE is applied to Q or K
- **THEN** frequency base = 1,000,000.0; head_dim=128; mode = neox (interleaved)

### Requirement: SwiGLU FFN in Qwen3 decoder
Each decoder layer FFN SHALL use: RMSNorm (no bias, eps=1e-6) → `gate = silu(ffn_gate_w * x)` → `up = ffn_up_w * x` → `ffn_out = ffn_down_w * (gate ⊙ up)`; dimensions: gate/up `[1024,3072]`, down `[3072,1024]`; no bias on any FFN weight.

#### Scenario: SwiGLU computation
- **WHEN** decoder FFN is computed
- **THEN** `ggml_silu` is applied to gate branch only; gate and up branches are multiplied elementwise; no bias tensors loaded for FFN

### Requirement: Autoregressive decode loop with KV cache
The system SHALL decode tokens one at a time, using the KV cache from prefill, stopping at `eos_token_id=151645` or `max_tokens`.

#### Scenario: First decode token
- **WHEN** prefill completes
- **THEN** the first decode step uses the last token of the prefill as input and generates the first output token

#### Scenario: EOS terminates decode
- **WHEN** any generated token equals 151645
- **THEN** the decode loop terminates immediately; EOS token is NOT included in the output text

#### Scenario: max_tokens guard
- **WHEN** `max_tokens` is reached before EOS
- **THEN** decode loop terminates; warning logged; partial output returned

### Requirement: KV cache management with shift-left eviction
The system SHALL maintain a persistent KV cache of shape `[kv_dim, window, n_layers]` where `kv_dim = n_kv_heads × head_dim = 8×128 = 1024` and `window = 2048` (default, configurable). New tokens are always written at row `kv_used`; when `kv_used == window`, a shift-left eviction compacts the cache before writing.

#### Scenario: KV cache write before overflow
- **WHEN** token at position `p` is processed and `kv_used < window`
- **THEN** K and V tensors for all 28 layers are stored at cache row `kv_used`; then `kv_used++`

#### Scenario: KV cache shift-left on overflow
- **WHEN** `kv_used == window` and a new token must be stored
- **THEN** cache rows 1..window-1 are moved to rows 0..window-2 (memmove on CPU, ggml tensor copy on GPU); new token written at row window-1; `kv_used` remains at window; NO ring buffer indexing is used
