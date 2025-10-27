import json

with open('audio_test_data/reference_data_correct.json') as f:
    d = json.load(f)
    
print(f"Stats: {d['stats']}")
print(f"Shape: {d['shape']}")
print(f"First 10 values of first frame: {d['mel_spectrogram'][0][:10]}")
print(f"Min of all values: {min(min(row) for row in d['mel_spectrogram'])}")
print(f"Max of all values: {max(max(row) for row in d['mel_spectrogram'])}")
