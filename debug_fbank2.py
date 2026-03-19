import sys
sys.path.insert(0, '/home/tanglin/workspace2/study/gguf')
import numpy as np, torch, yaml, wave
from pathlib import Path
from funasr.frontends.wav_frontend import WavFrontend, apply_lfr
from funasr.utils.load_utils import extract_fbank
import torchaudio.compliance.kaldi as kaldi

MODEL_DIR = Path('/home/tanglin/workspace2/study/gguf/Fun-ASR-Nano-2512')
AUDIO_WAV = '/tmp/en_test.wav'

with wave.open(AUDIO_WAV, 'rb') as wf:
    sr, n_frames = wf.getframerate(), wf.getnframes()
    audio_float = np.frombuffer(wf.readframes(n_frames), dtype=np.int16).astype(np.float32) / 32768.0

print(f'audio_float range: [{audio_float.min():.4f}, {audio_float.max():.4f}]')
config = yaml.safe_load((MODEL_DIR/'config.yaml').read_text())
print('frontend_conf:', config['frontend_conf'])
frontend = WavFrontend(**config['frontend_conf'])
print(f'upsacle_samples={frontend.upsacle_samples}')

# Python way: feed normalized float, frontend may upscale
wav_t = torch.from_numpy(audio_float)
speech, speech_lengths = extract_fbank(wav_t, data_type='sound', frontend=frontend, is_final=True)
print(f'Python speech shape: {speech.shape}  lengths: {speech_lengths}')
print(f'Python speech[0,0,:5] = {speech[0,0,:5].tolist()}')
print(f'Python speech range: [{speech.min():.4f}, {speech.max():.4f}]')

# C++ way: upscale by 32768 first, then fbank
wav_scaled = wav_t * 32768.0
mat = kaldi.fbank(wav_scaled.unsqueeze(0), num_mel_bins=80, frame_length=25,
    frame_shift=10, dither=0.0, energy_floor=0.0,
    window_type='hamming', sample_frequency=16000, snip_edges=True)
print(f'\nC++-style fbank (no LFR) shape: {mat.shape}')
print(f'fbank[0,:5] = {mat[0,:5].tolist()}')
print(f'fbank range: [{mat.min():.4f}, {mat.max():.4f}]')

mat_lfr = apply_lfr(mat, 7, 6)
print(f'\nC++-style LFR shape: {mat_lfr.shape}')
print(f'LFR[0,:5] = {mat_lfr[0,:5].tolist()}')
