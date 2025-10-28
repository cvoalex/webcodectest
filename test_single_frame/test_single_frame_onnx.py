"""
Test single frame ONNX inference with detailed tensor logging.
Processes exactly frame 8 with 640ms of audio (indices 0-16).
Use this to verify iOS inference matches Python/ONNX output exactly.
"""
import argparse, os, cv2, numpy as np, onnxruntime as ort, torch
from torch.utils.data import DataLoader
from utils import AudDataset, get_audio_features
import json

def build_args():
    p = argparse.ArgumentParser(description="Single frame test with detailed logging")
    p.add_argument("--name", type=str, required=True, help="Dataset name (e.g., 'sanders')")
    p.add_argument("--audio_path", type=str, required=True, help="Path to audio wav")
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

def load_lms(lms_path):
    """Load landmark points from .lms file"""
    pts = []
    try:
        with open(lms_path, 'r') as f:
            for line in f.read().splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                pts.append([float(parts[0]), float(parts[1])])
    except FileNotFoundError:
        return None
    if not pts:
        return None
    return np.asarray(pts, dtype=np.float32)

def bgr_crop_from_lms(img, lms):
    """Crop image based on landmarks"""
    h, w = img.shape[:2]
    if lms is None or lms.shape[0] < 53 or lms.shape[1] < 2:
        side = min(h, w)
        x0 = (w - side) // 2
        y0 = (h - side) // 2
        return img[y0:y0+side, x0:x0+side]
    xmin = int(round(lms[1][0]))
    ymin = int(round(lms[52][1]))
    xmax = int(round(lms[31][0]))
    width = max(1, xmax - xmin)
    ymax = ymin + width
    if xmin < 0 or ymin < 0 or xmax > w or ymax > h or xmax <= xmin or ymax <= ymin:
        side = min(h, w)
        x0 = (w - side) // 2
        y0 = (h - side) // 2
        return img[y0:y0+side, x0:x0+side]
    return img[ymin:ymax, xmin:xmax]

def log_tensor_stats(name, tensor, is_numpy=True):
    """Print detailed tensor statistics"""
    if is_numpy:
        arr = tensor
    else:
        arr = tensor.numpy() if isinstance(tensor, torch.Tensor) else tensor
    
    print(f"\n{'='*60}")
    print(f"Tensor: {name}")
    print(f"{'='*60}")
    print(f"Shape: {arr.shape}")
    print(f"Dtype: {arr.dtype}")
    print(f"Min: {arr.min():.8f}")
    print(f"Max: {arr.max():.8f}")
    print(f"Mean: {arr.mean():.8f}")
    print(f"Std: {arr.std():.8f}")
    print(f"First 10 values (flattened): {arr.flatten()[:10]}")
    print(f"Last 10 values (flattened): {arr.flatten()[-10:]}")
    
    # Show some sample values from different parts
    flat = arr.flatten()
    if len(flat) > 100:
        indices = [0, len(flat)//4, len(flat)//2, 3*len(flat)//4, len(flat)-1]
        print(f"Sample values at indices {indices}:")
        for idx in indices:
            print(f"  [{idx}] = {flat[idx]:.8f}")
    
    return arr

def main():
    args = build_args()
    
    # Resolve dataset directory
    dataset_dir = args.dataset_root if args.dataset_root else os.path.join('.', 'dataset', args.name)
    if dataset_dir.startswith('"') and dataset_dir.endswith('"'):
        dataset_dir = dataset_dir[1:-1]
    
    print(f"\n{'#'*60}")
    print(f"# Single Frame ONNX Test - Frame {args.frame_index}")
    print(f"{'#'*60}")
    print(f"Dataset: {dataset_dir}")
    print(f"Audio: {args.audio_path}")
    print(f"ASR Mode: {args.asr}")
    
    # Check paths
    img_dir = os.path.join(dataset_dir, 'full_body_img')
    lms_dir = os.path.join(dataset_dir, 'landmarks')
    
    if not os.path.isdir(img_dir):
        raise SystemExit(f"[ERROR] Missing full_body_img directory: {img_dir}")
    if not os.path.isdir(lms_dir):
        raise SystemExit(f"[ERROR] Missing landmarks directory: {lms_dir}")
    
    # Find ONNX models
    gen_onnx_path = find_onnx(dataset_dir, args.onnx)
    audio_enc_path = find_audio_encoder(dataset_dir, args.audio_encoder_onnx)
    
    print(f"\nGenerator ONNX: {gen_onnx_path}")
    print(f"Audio Encoder ONNX: {audio_enc_path}")
    
    # ============================================================
    # Step 1: Process audio with audio encoder
    # ============================================================
    print(f"\n{'='*60}")
    print("STEP 1: Audio Processing")
    print(f"{'='*60}")
    
    if not os.path.isfile(args.audio_path):
        raise SystemExit(f"[ERROR] Audio file not found: {args.audio_path}")
    
    # Load audio and extract features
    aud_ds = AudDataset(args.audio_path)
    aud_loader = DataLoader(aud_ds, batch_size=64, shuffle=False)
    
    print(f"Audio dataset loaded: {len(aud_ds)} mel chunks")
    
    # Run audio encoder ONNX
    sess_audio = ort.InferenceSession(audio_enc_path, providers=['CPUExecutionProvider'])
    
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
    # Step 2: Extract audio window for frame 8
    # ============================================================
    print(f"\n{'='*60}")
    print(f"STEP 2: Extract Audio Window for Frame {args.frame_index}")
    print(f"{'='*60}")
    
    # get_audio_features extracts 16 frames centered at index (index-8 to index+8)
    a_win = get_audio_features(audio_feats, args.frame_index)
    print(f"Audio window shape: {a_win.shape}")
    print(f"Audio window covers indices {args.frame_index - 8} to {args.frame_index + 8}")
    
    # Reshape based on ASR mode
    mode = args.asr
    if mode in ('hubert', 'wenet'):
        a_win_t = a_win.view(32, 32, 32)
    else:  # ave
        a_win_t = a_win.view(32, 16, 16)
    
    audio_np = a_win_t.unsqueeze(0).numpy().astype(np.float32)
    log_tensor_stats(f"Audio Input Tensor (reshaped for {mode})", audio_np)
    
    # ============================================================
    # Step 3: Load and prepare image
    # ============================================================
    print(f"\n{'='*60}")
    print(f"STEP 3: Load and Prepare Image Frame {args.frame_index}")
    print(f"{'='*60}")
    
    img_fn = f"{args.frame_index}.jpg"
    img_path = os.path.join(img_dir, img_fn)
    lms_path = os.path.join(lms_dir, f"{args.frame_index}.lms")
    
    print(f"Image: {img_path}")
    print(f"Landmarks: {lms_path}")
    
    img = cv2.imread(img_path)
    if img is None:
        raise SystemExit(f"[ERROR] Could not load image: {img_path}")
    
    print(f"Original image shape: {img.shape}")
    
    # Load landmarks
    lms = load_lms(lms_path)
    if lms is not None:
        print(f"Landmarks shape: {lms.shape}")
        print(f"Key landmarks: [1]={lms[1]}, [31]={lms[31]}, [52]={lms[52]}")
    else:
        print("No landmarks found, will use center crop")
    
    # Crop based on landmarks
    crop_img = bgr_crop_from_lms(img, lms)
    print(f"Cropped image shape: {crop_img.shape}")
    
    # Resize to 328x328
    crop_img = cv2.resize(crop_img, (328, 328), interpolation=cv2.INTER_CUBIC)
    crop_img_ori = crop_img.copy()
    
    # Extract ROI (320x320 from center)
    roi = crop_img[4:324, 4:324].copy()
    print(f"ROI shape: {roi.shape}")
    
    # Create real and masked versions
    img_real_ex = roi.transpose(2, 0, 1).astype(np.float32) / 255.0
    log_tensor_stats("Image Real (normalized, CHW)", img_real_ex)
    
    img_masked = cv2.rectangle(roi.copy(), (5, 5), (310, 305), (0, 0, 0), -1)
    img_masked = img_masked.transpose(2, 0, 1).astype(np.float32) / 255.0
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
    
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
    sess = ort.InferenceSession(gen_onnx_path, providers=providers)
    
    print(f"ONNX Runtime providers: {providers}")
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
    
    # Save outputs
    output_dir = os.path.join(dataset_dir, 'test_output')
    os.makedirs(output_dir, exist_ok=True)
    
    output_crop_path = os.path.join(output_dir, f'frame_{args.frame_index}_crop.jpg')
    output_roi_path = os.path.join(output_dir, f'frame_{args.frame_index}_roi_input.jpg')
    output_pred_path = os.path.join(output_dir, f'frame_{args.frame_index}_pred.jpg')
    output_masked_path = os.path.join(output_dir, f'frame_{args.frame_index}_masked.jpg')
    
    # Save intermediate images
    cv2.imwrite(output_crop_path, crop_img_ori)
    cv2.imwrite(output_roi_path, roi)
    cv2.imwrite(output_pred_path, pred_bgr)
    
    # Save masked input for reference
    masked_vis = (img_masked.transpose(1, 2, 0) * 255).astype(np.uint8)
    cv2.imwrite(output_masked_path, masked_vis)
    
    # Create final composite
    crop_img_ori[4:324, 4:324] = pred_bgr
    output_final_path = os.path.join(output_dir, f'frame_{args.frame_index}_final.jpg')
    cv2.imwrite(output_final_path, crop_img_ori)
    
    print(f"\nSaved outputs:")
    print(f"  - Crop: {output_crop_path}")
    print(f"  - ROI input: {output_roi_path}")
    print(f"  - Masked input: {output_masked_path}")
    print(f"  - Prediction: {output_pred_path}")
    print(f"  - Final composite: {output_final_path}")
    
    # ============================================================
    # Step 6: Save JSON data for iOS verification
    # ============================================================
    if args.output_json or True:  # Always save JSON
        json_path = args.output_json if args.output_json else os.path.join(output_dir, f'frame_{args.frame_index}_tensors.json')
        
        print(f"\n{'='*60}")
        print(f"STEP 6: Save Tensor Data to JSON")
        print(f"{'='*60}")
        
        # Prepare data (convert to lists for JSON)
        tensor_data = {
            "frame_index": args.frame_index,
            "asr_mode": args.asr,
            "dataset": args.name,
            "audio_path": args.audio_path,
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
        
        np.save(os.path.join(npy_dir, f'frame_{args.frame_index}_audio_window.npy'), a_win.numpy())
        np.save(os.path.join(npy_dir, f'frame_{args.frame_index}_audio_input.npy'), audio_np)
        np.save(os.path.join(npy_dir, f'frame_{args.frame_index}_image_input.npy'), six)
        np.save(os.path.join(npy_dir, f'frame_{args.frame_index}_output.npy'), ort_out)
        
        print(f"Saved .npy tensors to: {npy_dir}")
    
    print(f"\n{'#'*60}")
    print(f"# Test Complete!")
    print(f"{'#'*60}")
    print(f"\nUse these outputs to verify your iOS inference:")
    print(f"1. Compare tensor shapes")
    print(f"2. Compare min/max/mean/std statistics")
    print(f"3. Compare first/last 10 values")
    print(f"4. Load .npy files for exact value comparison")
    print(f"\nJSON summary: {json_path}")

if __name__ == '__main__':
    main()
