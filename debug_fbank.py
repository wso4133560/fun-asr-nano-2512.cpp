#!/usr/bin/env python3
"""
Compare Python fbank/LFR output with C++ debug dump.
"""
import sys
sys.path.insert(0, '/home/tanglin/workspace2/study/gguf')

import numpy as np
import torch
import yaml
from pathlib import Path
from funasr.frontends.wav_frontend import WavFrontend
from funasr.utils.load_utils import extract_fbank, load_audio_text_image_video

MODEL_DIR = Path('/home/tanglin/workspace2/study/gguf/Fun-ASR-Nano-2512')
AUDIO    = MODEL_DIR / 'example/en.mp3'

config   = yaml.safe_load((MODEL_DIR / 'config.yaml').read_text())
print('frontend_conf:', config['frontend_conf'])
frontend = WavFrontend(**config['frontend_conf'])

wav = load_audio_text_image_video(str(AUDIO), fs=frontend.fs)
print(f'wav range: [{wav.min():.4f}, {wav.max():.4f}]  shape={wav.shape}')

speech, speech_lengths = extract_fbank(wav, data_type='sound', frontend=frontend, is_final=True)
print(f'speech shape: {speech.shape}  lengths: {speech_lengths}')
print(f'speech[0,0,:10] = {speech[0,0,:10].tolist()}')
print(f'speech range: [{speech.min():.4f}, {speech.max():.4f}]')
print(f'speech mean: {speech.mean():.6f}  std: {speech.std():.6f}')

# The speech tensor shape is [batch, T', 560]
# First 80 of 560 = first mel frame
print(f'\nspeech[0,0,0:5] = {speech[0,0,0:5].tolist()}  <- first LFR frame, first 5 mel bins')
print(f'speech[0,0,80:85] = {speech[0,0,80:85].tolist()}  <- second mel frame within LFR')

# Also check wav frontend internals
print('\nWavFrontend config:')
print(f'  frame_shift={frontend.frame_shift}')
print(f'  frame_length={frontend.frame_length}')
print(f'  n_mels={frontend.n_mels}')
print(f'  lfr_m={frontend.lfr_m}')
print(f'  lfr_n={frontend.lfr_n}')
print(f'  cmvn_file={getattr(frontend, "cmvn_file", None)}')
try:
    print(f'  global_cmvn_mean[:5]={frontend.global_cmvn.mean[:5]}')
    print(f'  global_cmvn_std[:5]={frontend.global_cmvn.std[:5]}')
except:
    print('  no cmvn')
