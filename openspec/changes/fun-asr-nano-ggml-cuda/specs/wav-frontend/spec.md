## ADDED Requirements

### Requirement: Read 16kHz mono PCM WAV file
The system SHALL read a WAV file, convert stereo to mono by averaging channels, and resample if necessary to 16,000 Hz. No pre-emphasis filter is applied. No global CMVN is applied.

#### Scenario: Mono 16kHz WAV loaded
- **WHEN** a 16kHz mono WAV file is provided
- **THEN** raw PCM samples are returned as `std::vector<float>` in the range approximately [-1.0, 1.0] (normalized float representation of int16 PCM)

#### Scenario: Stereo WAV averaged to mono
- **WHEN** a stereo WAV file is provided
- **THEN** each output sample equals the mean of left and right channels

### Requirement: Upscale float samples before Fbank computation
The system SHALL multiply float PCM samples by 32768 before computing the mel filterbank, because the encoder was trained with raw int16 PCM values. This implements `funasr.frontend.upscale_samples=True`.

#### Scenario: Samples upscaled
- **WHEN** a WAV with float samples in [-1,1] is loaded
- **THEN** each sample is multiplied by 32768.0 before the STFT; the resulting amplitude range is approximately [-32768, 32767]

### Requirement: Compute 80-mel Hamming-windowed filterbank features
The system SHALL compute mel filterbank features using a **Hamming window** (NOT Hann), 25ms frame length (400 samples at 16kHz), 10ms frame shift (160 samples), 512-point FFT, 80 mel bins, with `snip_edges=true` (partial edge frames discarded). No pre-emphasis and no CMVN are applied.

#### Scenario: Hamming window applied
- **WHEN** the STFT is computed on a 1-second audio signal
- **THEN** each frame uses coefficients `w[n] = 0.54 - 0.46*cos(2π*n/399)` for n in [0,399]

#### Scenario: Output shape matches expected
- **WHEN** a WAV with N samples is processed (N >= 400)
- **THEN** the number of frames equals `floor((N - 400) / 160) + 1` (snip_edges=True formula), and each frame has 80 float values

#### Scenario: No frames for very short audio
- **WHEN** audio has fewer than 400 samples
- **THEN** the frontend returns 0 frames and no crash occurs

### Requirement: Apply Low Frame Rate (LFR) stacking
The system SHALL stack `m=7` consecutive mel frames into one LFR frame, advancing by `n=6` frames per step, producing output dimension 560 (=80×7).

#### Scenario: LFR output dimension
- **WHEN** mel features with shape `[F, 80]` are processed through LFR
- **THEN** output shape is `[T', 560]` where `T' = floor((F - 7) / 6) + 1`

#### Scenario: LFR boundary handling
- **WHEN** the number of mel frames `F` is less than 7
- **THEN** the LFR output has 0 frames and no out-of-bounds access occurs

#### Scenario: LFR frame content
- **WHEN** LFR processes frame index `i`
- **THEN** output row `i` equals concatenation of mel frames `[i*6, i*6+1, ..., i*6+6]` in feature dimension order
