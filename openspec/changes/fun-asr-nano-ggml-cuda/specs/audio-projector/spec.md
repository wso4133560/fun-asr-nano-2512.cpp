## ADDED Requirements

### Requirement: MLP projector maps encoder output 512→1024
The system SHALL project encoder output `[T', 512]` to `[T', 1024]` using `mm.a.linear1.weight [512,2048]` + `mm.a.linear1.bias [2048]` → **SiLU activation** → `mm.a.linear2.weight [2048,1024]` + `mm.a.linear2.bias [1024]`.

#### Scenario: MLP output shape
- **WHEN** encoder output `[T', 512]` passes through MLP projector
- **THEN** output shape is `[T', 1024]`; intermediate shape after linear1 is `[T', 2048]`

#### Scenario: MLP bias applied
- **WHEN** linear1 matmul is computed
- **THEN** `mm.a.linear1.bias [2048]` is added elementwise to every row of the matmul output

### Requirement: Adaptor Transformer refines projected embeddings
The system SHALL apply 2 standard Transformer blocks (`mm.a.blk.0` and `mm.a.blk.1`) to the MLP output `[T', 1024]`, using **4 attention heads** (head_dim=256) with separate Q/K/V projections (each `[1024,1024]` with bias), output projection `[1024,1024]` with bias, LayerNorm with bias (eps≈1e-12), and FFN with intermediate dim **256** (ground truth from tensor shapes `[1024,256]`/`[256,1024]`; GGUF metadata key `ffn_dim=2048` is incorrect and SHALL be ignored).

#### Scenario: Adaptor output shape preserved
- **WHEN** adaptor processes input `[T', 1024]`
- **THEN** each of the 2 blocks outputs `[T', 1024]`; final adaptor output shape is `[T', 1024]`

#### Scenario: Adaptor FFN bottleneck
- **WHEN** adaptor block FFN is computed
- **THEN** `mm.a.blk.{i}.feed_forward.w1.weight [1024,256]` reduces to dim 256; `mm.a.blk.{i}.feed_forward.w2.weight [256,1024]` projects back to 1024

#### Scenario: Adaptor LayerNorm with bias
- **WHEN** `mm.a.blk.{i}.norm1` or `norm2` is applied
- **THEN** both weight and bias tensors (`[1024]` each) are used in LayerNorm; epsilon = `funasr.audio_adaptor.layer_norm_epsilon` ≈ 1e-12

#### Scenario: No positional encoding in adaptor
- **WHEN** adaptor attention computes Q, K, V
- **THEN** no RoPE or sinusoidal position encoding is applied; attention is global over all T' audio frames

### Requirement: Audio tokens output to LLM embedding space
The adaptor output `[T', 1024]` SHALL be the final audio token embeddings, directly concatenated into the LLM prefill sequence without any additional projection.

#### Scenario: Audio tokens dimension matches LLM embedding
- **WHEN** audio tokens are prepared for LLM prefill
- **THEN** audio token dim (1024) equals LLM embedding dim (`qwen3.embedding_length` = 1024); no shape mismatch
