#!/usr/bin/env python3
import numpy as np
import onnxruntime as ort
import wave

# Load audio
with wave.open("/home/tanglin/workspace2/kotoba-tech/whisper.cpp/samples/jfk.wav", 'rb') as wf:
    sr = wf.getframerate()
    n_frames = wf.getnframes()
    audio_bytes = wf.readframes(n_frames)
    audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
print(f"Audio shape: {audio.shape}, sr={sr}, range: [{audio.min():.4f}, {audio.max():.4f}]")

# Load ONNX encoder
sess = ort.InferenceSession("./models/model.encoder.int8.onnx")

# Run encoder
outputs = sess.run(None, {"speech": audio[np.newaxis, :].astype(np.float32), "speech_lengths": np.array([len(audio)], dtype=np.int32)})
encoder_out = outputs[0]  # [1, T', 512]

print(f"ONNX encoder_out shape: {encoder_out.shape}")
print(f"ONNX encoder_out[0, 0, :5] = {encoder_out[0, 0, :5]}")
print(f"ONNX encoder_out range: [{encoder_out.min():.4f}, {encoder_out.max():.4f}]")
print(f"ONNX encoder_out mean: {encoder_out.mean():.4f}, std: {encoder_out.std():.4f}")

# Load projector
proj_sess = ort.InferenceSession("./models/model.embed_proj.onnx")
proj_out = proj_sess.run(None, {"encoder_out": encoder_out, "encoder_out_lens": np.array([encoder_out.shape[1]], dtype=np.int32)})
audio_embeds = proj_out[0]  # [1, T', 1024]

print(f"\nONNX audio_embeds shape: {audio_embeds.shape}")
print(f"ONNX audio_embeds[0, 0, :5] = {audio_embeds[0, 0, :5]}")
print(f"ONNX audio_embeds range: [{audio_embeds.min():.4f}, {audio_embeds.max():.4f}]")
print(f"ONNX audio_embeds mean: {audio_embeds.mean():.4f}, std: {audio_embeds.std():.4f}")

# Compare with our C++ output
cpp_audio_tokens = np.array([-32.4223, 33.3369, 36.4479, -31.1332, -5.7009])
onnx_audio_tokens = audio_embeds[0, 0, :5]
ratio = cpp_audio_tokens / onnx_audio_tokens
print(f"\nC++ / ONNX ratio: {ratio}")
print(f"Mean ratio: {ratio.mean():.4f}")
