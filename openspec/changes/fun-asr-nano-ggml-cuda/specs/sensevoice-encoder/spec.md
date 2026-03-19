## ADDED Requirements

### Requirement: Execute SANM encoder forward pass via GGML graph
The system SHALL build and execute a GGML compute graph for the SenseVoiceEncoderSmall encoder, processing LFR features `[T', 560]` through input block `a.blk.0`, regular blocks `a.blk.1..50`, and TP blocks `a.tp.blk.0..19`, producing encoder output `[T', 512]`.

#### Scenario: Input block handles 560→512 dimension change
- **WHEN** LFR features `[T', 560]` enter block 0
- **THEN** `a.blk.0.norm1` (LayerNorm, dim=560, with weight+bias) is applied first; `a.blk.0.self_attn.linear_qkv.weight [560,1536]` projects to QKV; output is `[T', 512]` after attention and FFN

#### Scenario: Regular and TP blocks maintain 512 dimensions
- **WHEN** blocks 1..50 and tp.blk.0..19 process input `[T', 512]`
- **THEN** each block input and output shape is `[T', 512]`; no dimension mismatch

#### Scenario: Encoder output shape
- **WHEN** the full encoder processes input with T' frames
- **THEN** `a.tp_norm` output (the final encoder output) has shape `[512, T']` in GGML column-major layout

### Requirement: SANM attention with FSMN memory convolution
Each encoder block's attention SHALL compute: (1) norm input → combined QKV from `linear_qkv [dim,3*dim]`, split into Q/K/V each `[dim/4, 4, T']` (4 heads, head_dim=dim/4); (2) FSMN memory: `m = ggml_conv_1d(normed_input, fsmn.weight [11,1,dim], stride=1, pad=5)` producing `[dim, T']`; (3) standard multi-head attention: `a = MHA(Q, K, V)`; (4) `combined = a + m` (add FSMN memory to attention output); (5) output projection: `linear_out(combined)` → `[dim, T']`; followed by residual add to block input.

#### Scenario: FSMN conv input and combination
- **WHEN** the SANM attention computes in block `i`
- **THEN** `ggml_conv_1d` receives the block's norm1-normalized input (NOT raw Q/K/V); its output `[dim,T']` is added elementwise to the MHA output BEFORE `linear_out` is applied

#### Scenario: FSMN conv kernel dimensions
- **WHEN** FSMN conv is executed for any encoder block
- **THEN** kernel `[11,1,512]`, stride=1, pad=5; output length equals input T' (same-length convolution)

#### Scenario: Residual connections preserved
- **WHEN** any encoder block processes input `x`
- **THEN** attention sub-layer output is `x + linear_out(MHA(norm1(x)) + fsmn(norm1(x)))`; FFN sub-layer output is `x_attn + w2(relu(w1(norm2(x_attn))+b1))+b2`

### Requirement: LayerNorm (not RMSNorm) in encoder
The system SHALL use LayerNorm with learnable `weight` AND `bias` (both loaded from GGUF F16 tensors) for all encoder normalization operations. RMSNorm is forbidden in the encoder.

#### Scenario: LayerNorm with bias
- **WHEN** `norm1` or `norm2` is applied in any encoder block
- **THEN** the computation uses `ggml_norm` (mean-subtraction + variance normalization) followed by elementwise scale (weight) and shift (bias); epsilon=1e-5

#### Scenario: All encoder norm tensors loaded
- **WHEN** the mmproj GGUF is loaded
- **THEN** every `a.blk.{i}.norm1.weight`, `a.blk.{i}.norm1.bias`, `a.blk.{i}.norm2.weight`, `a.blk.{i}.norm2.bias` tensor is present; bias tensors are applied in norm computation

### Requirement: FFN uses ReLU activation (not SwiGLU)
Encoder FFN SHALL use `ReLU` activation: `w2 * relu(w1 * x + b1) + b2`. Both w1 and w2 have bias terms loaded from GGUF.

#### Scenario: ReLU applied in FFN
- **WHEN** encoder FFN processes input
- **THEN** `ggml_relu` is applied after `w1` matmul; negative values are zero in intermediate output

#### Scenario: FFN dimension
- **WHEN** regular encoder blocks (blk.1..50) compute FFN
- **THEN** intermediate dim is 2048 (`feed_forward.w1.weight [512,2048]`), output dim is 512 (`feed_forward.w2.weight [2048,512]`)
