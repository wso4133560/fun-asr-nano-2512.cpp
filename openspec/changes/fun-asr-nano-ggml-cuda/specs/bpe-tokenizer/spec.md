## ADDED Requirements

### Requirement: Load GPT-2 BPE vocabulary from LLM GGUF metadata
The system SHALL load the vocabulary from `tokenizer.ggml.tokens` (string array, 151936 entries) in the LLM GGUF file. No external tokenizer files are required.

#### Scenario: Vocabulary loaded
- **WHEN** LLM GGUF is loaded
- **THEN** vocabulary array contains exactly 151936 entries; entry at index 0 is `"!"` (ASCII 33)

#### Scenario: Special tokens identified
- **WHEN** vocabulary is loaded
- **THEN** tokens at indices 151643 (bos `<|endoftext|>`), 151645 (eos `<|im_end|>`) are accessible; `<|startofspeech|>` and `<|endofspeech|>` tokens are located by string search

### Requirement: Encode text prompt to token IDs
The system SHALL tokenize prompt strings using GPT-2 BPE (`tokenizer.ggml.pre = qwen2`) to produce token ID sequences for prefill construction.

#### Scenario: System prompt tokenized
- **WHEN** `"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"` is tokenized
- **THEN** output token IDs are valid indices in [0, 151936); first token is `<|im_start|>` ID

#### Scenario: Special token IDs found
- **WHEN** `<|startofspeech|>` and `<|endofspeech|>` are tokenized
- **THEN** each produces exactly one token ID; these IDs are used as boundary markers in prefill

### Requirement: Decode token IDs to UTF-8 text
The system SHALL decode output token IDs to a UTF-8 string by concatenating vocabulary strings, replacing GPT-2 BPE byte-level encoding: `Ġ` (U+0120) → space, `Ċ` (U+010A) → newline.

#### Scenario: Space character decoded
- **WHEN** a token string starts with `Ġ`
- **THEN** `Ġ` is replaced with a space character in the output string

#### Scenario: EOS token excluded from output
- **WHEN** decode produces token ID 151645
- **THEN** the token is NOT added to the output string; decoding terminates

#### Scenario: Multi-token Chinese string
- **WHEN** Qwen3 generates a sequence of Chinese text tokens
- **THEN** concatenated decoded bytes form valid UTF-8; no replacement character (U+FFFD) in output
