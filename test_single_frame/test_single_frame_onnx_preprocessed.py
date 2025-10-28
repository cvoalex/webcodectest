"""
Test single frame ONNX inference using PREPROCESSED image cache + LIVE audio.
This version uses:
  - Preprocessed images from cache/ (rois_320, model_inputs)
  - Live audio processing from WAV file (on-the-fly)

This combines the best of both:
  - Images match training data exactly
  - Audio can be any WAV file for testing

Self-contained audio processing (no utils.py dependency).
"""
import argparse, os, cv2, numpy as np, onnxruntime as ort
import json
import librosa
import torch
from torch.utils.data import DataLoader
from scipy import signal

def build_args():
    p = argparse.ArgumentParser(description="Single frame test with preprocessed images + live audio")
    p.add_argument("--name", type=str, required=True, help="Dataset name (e.g., 'sanders')")
    p.add_argument("--audio_path", type=str, default="", help="Path to audio WAV file (default: dataset/<name>/aud.wav)")
    p.add_argument("--asr", type=str, default="ave", choices=["ave","hubert","wenet"])
    p.add_argument("--dataset_root", type=str, default="", help="Override dataset root (else dataset/<name>)")
    p.add_argument("--onnx", type=str, default="", help="Explicit generator ONNX path")
    p.add_argument("--audio_encoder_onnx", type=str, default="", help="Explicit audio encoder ONNX path")
    p.add_argument("--frame_index", type=int, default=8, help="Frame index to test (default 8)")
    p.add_argument("--output_json", type=str, default="", help="JSON file to save tensor data")
    return p.parse_args()

def find_onnx(dataset_dir: str, explicit_path: str = "") -> str:
    """Find generator ONNX model"""
    if explicit_path and os.path.isfile(explicit_path):
        return explicit_path
    candidates = [
        os.path.join(dataset_dir, 'models', 'generator.onnx'),
        os.path.join(dataset_dir, 'checkpoint', 'model_best.onnx'),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise SystemExit(f"[ERROR] Generator ONNX not found. Checked: {candidates}")

def find_audio_encoder(dataset_dir: str, explicit_path: str = "") -> str:
    """Find audio encoder ONNX model"""
    if explicit_path and os.path.isfile(explicit_path):
        return explicit_path
    candidates = [
        os.path.join(dataset_dir, 'models', 'audio_encoder.onnx'),
        os.path.join('model', 'checkpoints', 'audio_encoder.onnx')
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise SystemExit(f"[ERROR] Audio encoder ONNX not found. Checked: {candidates}")

# Audio processing functions (from utils.py but standalone)
from scipy import signal

def preemphasis(wav, k=0.97):
    return signal.lfilter([1, -k], [1], wav)

def _stft(y):
    return librosa.stft(y=y, n_fft=800, hop_length=200, win_length=800)

def _build_mel_basis():
    return librosa.filters.mel(sr=16000, n_fft=800, n_mels=80, fmin=55, fmax=7600)

def _linear_to_mel(spectrogram):
    _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectrogram)

def _amp_to_db(x):
    min_level = np.exp(-5 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))

def _normalize(S):
    return np.clip((2 * 4.) * ((S - -100) / (--100)) - 4., -4., 4.)

def melspectrogram(wav):
    D = _stft(preemphasis(wav, 0.97))
    S = _amp_to_db(_linear_to_mel(np.abs(D))) - 20
    return _normalize(S)

# Audio dataset class (from utils.py but standalone)
class AudDataset(object):
    def __init__(self, wavpath):
        wav = librosa.core.load(wavpath, sr=16000)[0]
        self.orig_mel = melspectrogram(wav).T
        self.data_len = int((self.orig_mel.shape[0] - 16) / 80. * float(25)) + 2
    
    def crop_audio_window(self, spec, start_frame):
        if type(start_frame) == int:
            start_frame_num = start_frame
        else:
            start_frame_num = int(os.path.basename(start_frame).split('.')[0])
        start_idx = int(80. * (start_frame_num / float(25)))
        
        end_idx = start_idx + 16
        if end_idx > spec.shape[0]:
            end_idx = spec.shape[0]
            start_idx = end_idx - 16
        
        return spec[start_idx: end_idx, :]
    
    def __len__(self):
        return self.data_len
    
    def __getitem__(self, idx):
        mel = self.crop_audio_window(self.orig_mel.copy(), idx)
        if (mel.shape[0] != 16):
            raise Exception('mel.shape[0] != 16')
        mel = torch.FloatTensor(mel.T).unsqueeze(0)
        return mel

def get_audio_features_simple(features, index):
    """
    Extract audio window for given frame index.
    No torch dependency - pure numpy.
    """
    left = index - 8
    right = index + 8
    pad_left = 0
    pad_right = 0
    
    if left < 0:
        pad_left = -left
        left = 0
    if right > features.shape[0]:
        pad_right = right - features.shape[0]
        right = features.shape[0]
    
    auds = features[left:right]
    
    if pad_left > 0:
        auds = np.concatenate([np.zeros_like(auds[:pad_left]), auds], axis=0)
    if pad_right > 0:
        auds = np.concatenate([auds, np.zeros_like(auds[:pad_right])], axis=0)
    
    return auds

def log_tensor_stats(name, tensor):
    """Print detailed tensor statistics"""
    print(f"\n{'='*60}")
    print(f"Tensor: {name}")
    print(f"{'='*60}")
    print(f"Shape: {tensor.shape}")
    print(f"Dtype: {tensor.dtype}")
    print(f"Min: {tensor.min():.8f}")
    print(f"Max: {tensor.max():.8f}")
    print(f"Mean: {tensor.mean():.8f}")
    print(f"Std: {tensor.std():.8f}")
    print(f"First 10 values (flattened): {tensor.flatten()[:10]}")
    print(f"Last 10 values (flattened): {tensor.flatten()[-10:]}")
    
    flat = tensor.flatten()
    if len(flat) > 100:
        indices = [0, len(flat)//4, len(flat)//2, 3*len(flat)//4, len(flat)-1]
        print(f"Sample values at indices {indices}:")
        for idx in indices:
            print(f"  [{idx}] = {flat[idx]:.8f}")
    
    return tensor

def main():
    args = build_args()
    
    # Resolve dataset directory
    dataset_dir = args.dataset_root if args.dataset_root else os.path.join('.', 'dataset', args.name)
    if dataset_dir.startswith('"') and dataset_dir.endswith('"'):
        dataset_dir = dataset_dir[1:-1]
    
    print(f"\n{'#'*60}")
    print(f"# Single Frame ONNX Test - Frame {args.frame_index}")
    print(f"# Using PREPROCESSED DATA from cache/")
    print(f"{'#'*60}")
    print(f"Dataset: {dataset_dir}")
    print(f"ASR Mode: {args.asr}")
    
    # Check cache directory exists
    cache_dir = os.path.join(dataset_dir, 'cache')
    if not os.path.isdir(cache_dir):
        raise SystemExit(f"[ERROR] Cache directory not found: {cache_dir}\nRun preprocessing first!")
    
    model_inputs_dir = os.path.join(cache_dir, 'model_inputs')
    rois_dir = os.path.join(cache_dir, 'rois_320')
    
    if not os.path.isdir(model_inputs_dir):
        raise SystemExit(f"[ERROR] model_inputs directory not found: {model_inputs_dir}")
    if not os.path.isdir(rois_dir):
        raise SystemExit(f"[ERROR] rois_320 directory not found: {rois_dir}")
    
    # Find ONNX model
    gen_onnx_path = find_onnx(dataset_dir, args.onnx)
    print(f"\nGenerator ONNX: {gen_onnx_path}")
    
    # ============================================================
    # Step 1: Process audio from WAV file
    # ============================================================
    print(f"\n{'='*60}")
    print("STEP 1: Process Audio from WAV File")
    print(f"{'='*60}")
    
    # Determine audio path
    if args.audio_path:
        audio_path = args.audio_path
    else:
        # Default to aud.wav in dataset directory
        audio_path = os.path.join(dataset_dir, 'aud.wav')
    
    if not os.path.isfile(audio_path):
        raise SystemExit(f"[ERROR] Audio file not found: {audio_path}")
    
    print(f"Audio file: {audio_path}")
    
    # Find audio encoder
    audio_encoder_path = find_audio_encoder(dataset_dir, args.audio_encoder_onnx)
    print(f"Audio encoder ONNX: {audio_encoder_path}")
    
    # Load audio and extract features
    aud_ds = AudDataset(audio_path)
    aud_loader = DataLoader(aud_ds, batch_size=64, shuffle=False)
    
    print(f"Audio dataset loaded: {len(aud_ds)} mel chunks")
    
    # Run audio encoder ONNX
    sess_audio = ort.InferenceSession(audio_encoder_path, providers=['CPUExecutionProvider'])
    
    emb_chunks = []
    for batch_idx, mel in enumerate(aud_loader):
        mel_np = mel.numpy()
        if batch_idx == 0:  # Log first batch
            log_tensor_stats(f"Audio Input (batch 0) - Mel Spectrogram", mel_np)
        
        out_np = sess_audio.run(None, {sess_audio.get_inputs()[0].name: mel_np})[0]
        
        if batch_idx == 0:  # Log first output
            log_tensor_stats(f"Audio Encoder Output (batch 0)", out_np)
        
        emb_chunks.append(torch.from_numpy(out_np))
    
    outputs = torch.cat(emb_chunks, dim=0)
    print(f"\nTotal audio features shape: {outputs.shape}")
    
    # Add leading/trailing frame duplication (as in original code)
    first_frame, last_frame = outputs[:1], outputs[-1:]
    audio_feats = torch.cat([first_frame, outputs, last_frame], dim=0).numpy()
    
    print(f"Audio features with padding: {audio_feats.shape}")
    log_tensor_stats("Complete Audio Features", audio_feats)
    
    # ============================================================
    # Step 2: Extract audio window for frame
    # ============================================================
    print(f"\n{'='*60}")
    print(f"STEP 2: Extract Audio Window for Frame {args.frame_index}")
    print(f"{'='*60}")
    
    a_win = get_audio_features_simple(audio_feats, args.frame_index)
    print(f"Audio window shape: {a_win.shape}")
    print(f"Audio window covers indices {args.frame_index - 8} to {args.frame_index + 8}")
    
    # Reshape based on ASR mode
    mode = args.asr
    if mode in ('hubert', 'wenet'):
        a_win_reshaped = a_win.reshape(32, 32, 32)
    else:  # ave
        a_win_reshaped = a_win.reshape(32, 16, 16)
    
    audio_np = np.expand_dims(a_win_reshaped, axis=0).astype(np.float32)
    log_tensor_stats(f"Audio Input Tensor (reshaped for {mode})", audio_np)
    
    # ============================================================
    # Step 3: Load preprocessed images
    # ============================================================
    print(f"\n{'='*60}")
    print(f"STEP 3: Load Preprocessed Images for Frame {args.frame_index}")
    print(f"{'='*60}")
    
    roi_path = os.path.join(rois_dir, f'{args.frame_index}.jpg')
    masked_path = os.path.join(model_inputs_dir, f'{args.frame_index}_masked.jpg')
    
    print(f"ROI (real): {roi_path}")
    print(f"Masked: {masked_path}")
    
    if not os.path.isfile(roi_path):
        raise SystemExit(f"[ERROR] ROI not found: {roi_path}")
    if not os.path.isfile(masked_path):
        raise SystemExit(f"[ERROR] Masked image not found: {masked_path}")
    
    # Load preprocessed images (already 320x320)
    roi_img = cv2.imread(roi_path)
    masked_img = cv2.imread(masked_path)
    
    if roi_img is None:
        raise SystemExit(f"[ERROR] Could not load ROI: {roi_path}")
    if masked_img is None:
        raise SystemExit(f"[ERROR] Could not load masked image: {masked_path}")
    
    print(f"ROI shape: {roi_img.shape}")
    print(f"Masked shape: {masked_img.shape}")
    
    # Convert to normalized CHW format
    img_real_ex = roi_img.transpose(2, 0, 1).astype(np.float32) / 255.0
    img_masked = masked_img.transpose(2, 0, 1).astype(np.float32) / 255.0
    
    log_tensor_stats("Image Real (normalized, CHW)", img_real_ex)
    log_tensor_stats("Image Masked (normalized, CHW)", img_masked)
    
    # Concatenate to 6 channels
    six = np.concatenate([img_real_ex, img_masked], axis=0)[None].astype(np.float32)
    log_tensor_stats("6-Channel Image Input", six)
    
    # ============================================================
    # Step 4: Run Generator ONNX Inference
    # ============================================================
    print(f"\n{'='*60}")
    print(f"STEP 4: Run Generator Inference")
    print(f"{'='*60}")
    
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    sess = ort.InferenceSession(gen_onnx_path, providers=providers)
    
    print(f"ONNX Runtime providers available")
    print(f"Generator inputs:")
    for inp in sess.get_inputs():
        print(f"  - {inp.name}: {inp.shape} ({inp.type})")
    print(f"Generator outputs:")
    for out in sess.get_outputs():
        print(f"  - {out.name}: {out.shape} ({out.type})")
    
    ort_inputs = {
        sess.get_inputs()[0].name: six,
        sess.get_inputs()[1].name: audio_np
    }
    
    print("\nRunning inference...")
    ort_out = sess.run(None, ort_inputs)[0][0]  # (3, 320, 320)
    
    log_tensor_stats("Generator Output (CHW)", ort_out)
    
    # ============================================================
    # Step 5: Post-process and save
    # ============================================================
    print(f"\n{'='*60}")
    print(f"STEP 5: Post-process Output")
    print(f"{'='*60}")
    
    # Convert to BGR image
    pred_bgr = (np.clip(ort_out.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
    print(f"Predicted BGR shape: {pred_bgr.shape}")
    print(f"Predicted BGR range: [{pred_bgr.min()}, {pred_bgr.max()}]")
    
    # Create full frame overlay (same as inference_328.py)
    # Load original full body image
    full_img_path = os.path.join(dataset_dir, "full_body_img", f"{args.frame_index}.jpg")
    full_img = cv2.imread(full_img_path)
    
    if full_img is not None:
        # Load landmarks to get overlay position
        lms_path = os.path.join(dataset_dir, "landmarks", f"{args.frame_index}.lms")
        lms = None
        if os.path.exists(lms_path):
            lms_list = []
            with open(lms_path, "r") as f:
                for line in f.read().splitlines():
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 2:
                            lms_list.append([float(parts[0]), float(parts[1])])
            if lms_list:
                lms = np.asarray(lms_list, dtype=np.float32)
        
        # Create 328x328 crop and insert prediction at center (4:324)
        crop_328 = cv2.resize(roi_img, (328, 328), interpolation=cv2.INTER_CUBIC)
        crop_328[4:324, 4:324] = pred_bgr
        
        # Overlay on full image using landmarks
        H, W = full_img.shape[:2]
        if lms is not None and lms.shape[0] >= 53:
            # Use landmarks to position
            xmin = int(round(lms[1][0]))
            ymin = int(round(lms[52][1]))
            xmax = int(round(lms[31][0]))
            width = max(1, xmax - xmin)
            ymax = ymin + width
            
            # Resize crop to fit
            crop_resized = cv2.resize(crop_328, (width, width), interpolation=cv2.INTER_CUBIC)
            
            # Clamp coordinates
            xmin = max(0, min(xmin, W - 1))
            xmax = max(0, min(xmax, W))
            ymin = max(0, min(ymin, H - 1))
            ymax = max(0, min(ymax, H))
            
            # Overlay
            full_img[ymin:ymax, xmin:xmax] = crop_resized[0:(ymax - ymin), 0:(xmax - xmin)]
        else:
            # No landmarks - center the face
            side = min(H, W)
            x0 = (W - side) // 2
            y0 = (H - side) // 2
            crop_resized = cv2.resize(crop_328, (side, side), interpolation=cv2.INTER_CUBIC)
            full_img[y0:y0 + side, x0:x0 + side] = crop_resized
        
        print(f"Created full frame overlay ({H}x{W})\n")
    else:
        print(f"Warning: Could not load full body image, skipping overlay\n")
        full_img = None
    
    # Save outputs
    output_dir = os.path.join(dataset_dir, 'test_output_preprocessed')
    os.makedirs(output_dir, exist_ok=True)
    
    output_roi_path = os.path.join(output_dir, f'frame_{args.frame_index}_roi_preprocessed.jpg')
    output_masked_path = os.path.join(output_dir, f'frame_{args.frame_index}_masked_preprocessed.jpg')
    output_pred_path = os.path.join(output_dir, f'frame_{args.frame_index}_pred.jpg')
    output_full_path = os.path.join(output_dir, f'frame_{args.frame_index}_full.jpg')
    
    # Save images
    cv2.imwrite(output_roi_path, roi_img)
    cv2.imwrite(output_masked_path, masked_img)
    cv2.imwrite(output_pred_path, pred_bgr)
    if full_img is not None:
        cv2.imwrite(output_full_path, full_img)
    
    print(f"\nSaved outputs:")
    print(f"  - ROI (preprocessed): {output_roi_path}")
    print(f"  - Masked (preprocessed): {output_masked_path}")
    print(f"  - Prediction: {output_pred_path}")
    if full_img is not None:
        print(f"  - Full frame overlay: {output_full_path}")
    
    # ============================================================
    # Step 6: Save JSON data for iOS verification
    # ============================================================
    json_path = args.output_json if args.output_json else os.path.join(output_dir, f'frame_{args.frame_index}_tensors.json')
    
    print(f"\n{'='*60}")
    print(f"STEP 6: Save Tensor Data to JSON")
    print(f"{'='*60}")
    
    # Prepare data (convert to lists for JSON)
    tensor_data = {
        "frame_index": args.frame_index,
        "asr_mode": args.asr,
        "dataset": args.name,
        "uses_preprocessed_images": True,
        "uses_live_audio": True,
        "audio_path": audio_path,
        "preprocessed_roi_path": roi_path,
        "preprocessed_masked_path": masked_path,
        "tensors": {
            "audio_features_full_shape": list(audio_feats.shape),
            "audio_window_shape": list(a_win.shape),
            "audio_window_reshaped_shape": list(audio_np.shape),
            "image_input_shape": list(six.shape),
            "output_shape": list(ort_out.shape),
            
            # Save first/last values for quick verification
            "audio_window_first_10": a_win.flatten()[:10].tolist(),
            "audio_window_last_10": a_win.flatten()[-10:].tolist(),
            "audio_window_stats": {
                "min": float(a_win.min()),
                "max": float(a_win.max()),
                "mean": float(a_win.mean()),
                "std": float(a_win.std())
            },
            
            "image_input_first_10": six.flatten()[:10].tolist(),
            "image_input_last_10": six.flatten()[-10:].tolist(),
            "image_input_stats": {
                "min": float(six.min()),
                "max": float(six.max()),
                "mean": float(six.mean()),
                "std": float(six.std())
            },
            
            "output_first_10": ort_out.flatten()[:10].tolist(),
            "output_last_10": ort_out.flatten()[-10:].tolist(),
            "output_stats": {
                "min": float(ort_out.min()),
                "max": float(ort_out.max()),
                "mean": float(ort_out.mean()),
                "std": float(ort_out.std())
            }
        }
    }
    
    with open(json_path, 'w') as f:
        json.dump(tensor_data, f, indent=2)
    
    print(f"Saved tensor data to: {json_path}")
    
    # Also save full tensors as .npy files for exact verification
    npy_dir = os.path.join(output_dir, 'tensors_npy')
    os.makedirs(npy_dir, exist_ok=True)
    
    np.save(os.path.join(npy_dir, f'frame_{args.frame_index}_audio_window.npy'), a_win)
    np.save(os.path.join(npy_dir, f'frame_{args.frame_index}_audio_input.npy'), audio_np)
    np.save(os.path.join(npy_dir, f'frame_{args.frame_index}_image_input.npy'), six)
    np.save(os.path.join(npy_dir, f'frame_{args.frame_index}_output.npy'), ort_out)
    
    print(f"Saved .npy tensors to: {npy_dir}")
    
    print(f"\n{'#'*60}")
    print(f"# Test Complete!")
    print(f"{'#'*60}")
    print(f"\nThis test used:")
    print(f"  - PREPROCESSED images from cache/ (exact training data)")
    print(f"  - LIVE audio processing from: {audio_path}")
    print(f"\nUse these outputs to verify your iOS inference:")
    print(f"1. Compare tensor shapes")
    print(f"2. Compare min/max/mean/std statistics")
    print(f"3. Compare first/last 10 values")
    print(f"4. Load .npy files for exact value comparison")
    print(f"\nJSON summary: {json_path}")

if __name__ == '__main__':
    main()
