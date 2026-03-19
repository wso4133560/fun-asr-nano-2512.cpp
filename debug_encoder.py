#!/usr/bin/env python3
"""
Dump intermediate encoder/adaptor values from the Python reference implementation
for comparison with C++ output.
Run from the fun-asr-nano-2512.cpp directory.
"""
import sys
sys.path.insert(0, '/home/tanglin/workspace2/study/gguf')
sys.path.insert(0, '/home/tanglin/workspace2/study/gguf/llama.cpp/gguf-py')

import numpy as np
import torch
import yaml
from pathlib import Path
from funasr.frontends.wav_frontend import WavFrontend
from funasr.register import tables
from funasr.utils.load_utils import extract_fbank, load_audio_text_image_video

MODEL_DIR = Path('/home/tanglin/workspace2/study/gguf/Fun-ASR-Nano-2512')
AUDIO    = MODEL_DIR / 'example/en.mp3'

config   = yaml.safe_load((MODEL_DIR / 'config.yaml').read_text())
frontend = WavFrontend(**config['frontend_conf'])
encoder  = tables.encoder_classes.get(config['audio_encoder'])(
    input_size=frontend.output_size(), **config['audio_encoder_conf'])
adaptor  = tables.adaptor_classes.get(config['audio_adaptor'])(
    **config['audio_adaptor_conf'])

state = torch.load(MODEL_DIR / 'model.pt', map_location='cpu')['state_dict']
enc_state = {k[len('audio_encoder.'):]: v for k, v in state.items() if k.startswith('audio_encoder.')}
ada_state = {k[len('audio_adaptor.'):]: v for k, v in state.items() if k.startswith('audio_adaptor.')}
encoder.load_state_dict(enc_state, strict=True)
adaptor.load_state_dict(ada_state, strict=True)
encoder.eval(); adaptor.eval()

wav = load_audio_text_image_video(str(AUDIO), fs=frontend.fs)
speech, speech_lengths = extract_fbank(wav, data_type='sound', frontend=frontend, is_final=True)
print(f'speech shape: {speech.shape}, lengths: {speech_lengths}')

# --- Hook into encoder ---
hooks = {}
handles = []

def make_hook(name):
    def hook(module, inp, out):
        if isinstance(out, tuple):
            out_t = out[0]
        else:
            out_t = out
        if isinstance(out_t, torch.Tensor):
            hooks[name] = out_t.detach().float().cpu()
    return hook

# Register hooks on encoder blocks
for i, blk in enumerate(encoder.encoders0):
    handles.append(blk.register_forward_hook(make_hook(f'enc_blk0_{i}')))
for i, blk in enumerate(encoder.encoders):
    if i < 3:
        handles.append(blk.register_forward_hook(make_hook(f'enc_blk_{i+1}')))
handles.append(encoder.after_norm.register_forward_hook(make_hook('after_norm')))
for i, blk in enumerate(encoder.tp_encoders):
    if i < 3:
        handles.append(blk.register_forward_hook(make_hook(f'tp_blk_{i}')))
handles.append(encoder.tp_norm.register_forward_hook(make_hook('tp_norm')))

with torch.no_grad():
    encoder_out, encoder_out_lens = encoder(speech, speech_lengths)
    adaptor_out, adaptor_out_lens = adaptor(encoder_out, encoder_out_lens)

for h in handles:
    h.remove()

T = int(encoder_out_lens[0])
print(f'T={T}  encoder_out[0,0,:5] = {encoder_out[0,0,:5].tolist()}')
print(f'adaptor_out[0,0,:5] = {adaptor_out[0,0,:5].tolist()}')

print('\n=== Intermediate values (first frame, first 5 dims) ===')
for name, val in sorted(hooks.items()):
    v = val[0, 0, :5] if val.dim() == 3 else val[0, :5]
    print(f'  {name}: {v.tolist()}')

# Also dump raw LFR/fbank stats
print(f'\nspeech[0,0,:5] = {speech[0,0,:5].tolist()}')
print(f'speech range: [{speech.min():.4f}, {speech.max():.4f}]')
print(f'speech mean: {speech.mean():.6f}  std: {speech.std():.6f}')
