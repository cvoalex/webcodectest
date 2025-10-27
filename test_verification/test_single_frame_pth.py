"""
Self-contained single-frame PyTorch test - based directly on inference_328.py
Tests one frame with .pth model to compare with ONNX version.

Self-contained: Includes audio processing (no utils.py dependency for audio).
Only requires: unet_328.py for Model architecture
"""
import argparse, os, cv2, torch, numpy as np, json
import librosa
import librosa.filters
from scipy import signal
from torch.utils.data import DataLoader
import torch.nn as nn

# ============================================================
# Audio Processing Classes (copied from utils.py)
# ============================================================

class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, leakyReLU=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size, stride, padding),
            nn.BatchNorm2d(cout)
        )
        if leakyReLU:
            self.act = nn.LeakyReLU(0.02)
        else:
            self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)


class AudioEncoder(nn.Module):
    def __init__(self):
        super(AudioEncoder, self).__init__()

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0), )

    def forward(self, x):
        out = self.audio_encoder(x)
        out = out.squeeze(2).squeeze(2)
        return out


def load_wav(path, sr):
    return librosa.core.load(path, sr=sr)[0]


def preemphasis(wav, k):
    return signal.lfilter([1, -k], [1], wav)


def _stft(y):
    return librosa.stft(y=y, n_fft=800, hop_length=200, win_length=800)


def _linear_to_mel(spectogram):
    _mel_basis = librosa.filters.mel(sr=16000, n_fft=800, n_mels=80, fmin=55, fmax=7600)
    return np.dot(_mel_basis, spectogram)


def _amp_to_db(x):
    min_level = np.exp(-5 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))


def _normalize(S):
    return np.clip((2 * 4.) * ((S - -100) / (--100)) - 4., -4., 4.)


def melspectrogram(wav):
    D = _stft(preemphasis(wav, 0.97))
    S = _amp_to_db(_linear_to_mel(np.abs(D))) - 20
    return _normalize(S)


class AudDataset(object):
    def __init__(self, wavpath):
        wav = load_wav(wavpath, 16000)
        self.orig_mel = melspectrogram(wav).T
        self.data_len = int((self.orig_mel.shape[0] - 16) / 80. * float(25)) + 2

    def crop_audio_window(self, spec, start_frame):
        start_idx = int(80. * (start_frame / float(25)))
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


def get_audio_features(audio_encoder, dataloader, device='cpu'):
    """Process audio through encoder"""
    all_feats = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            feat = audio_encoder(batch)
            all_feats.append(feat.cpu().numpy())
    return np.concatenate(all_feats, axis=0)


def get_audio_window(features, index):
    """Get 16-frame audio window for a specific frame"""
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
    auds = torch.from_numpy(features[left:right])
    if pad_left > 0:
        auds = torch.cat([torch.zeros_like(auds[:pad_left]), auds], dim=0)
    if pad_right > 0:
        auds = torch.cat([auds, torch.zeros_like(auds[:pad_right])], dim=0)
    return auds


# ============================================================
# Import only Model architecture (still needed)
# ============================================================
from unet_328 import Model

def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", type=str, required=True)
    p.add_argument("--audio_path", type=str, default="")
    p.add_argument("--asr", type=str, default="ave", choices=["ave","hubert","wenet"])
    p.add_argument("--frame_index", type=int, default=8)
    return p.parse_args()

def main():
    args = build_args()
    
    # Setup paths
    dataset_dir = os.path.join('.', 'dataset', args.name)
    audio_path = args.audio_path if args.audio_path else os.path.join(dataset_dir, "aud.wav")
    checkpoint_path = os.path.join(dataset_dir, "checkpoint", "best_trainloss.pth")
    cache_dir = os.path.join(dataset_dir, "cache")
    output_dir = os.path.join(dataset_dir, "test_output_pth")
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"PyTorch Single Frame Test")
    print(f"Dataset: {dataset_dir}")
    print(f"Audio: {audio_path}")
    print(f"Frame: {args.frame_index}")
    print(f"Device: {device}\n")
    
    # Load audio encoder (same as inference_328.py)
    av_ckpt_path = os.path.join("model", "checkpoints", "audio_visual_encoder.pth")
    audio_enc = AudioEncoder().to(device).eval()
    av_state = torch.load(av_ckpt_path, map_location=device)
    if all(not k.startswith("audio_encoder.") for k in av_state.keys()):
        av_state = {f"audio_encoder.{k}": v for k, v in av_state.items()}
    audio_enc.load_state_dict(av_state, strict=False)
    print("Audio encoder loaded")
    
    # Process audio (same as inference_328.py)
    aud_ds = AudDataset(audio_path)
    aud_loader = DataLoader(aud_ds, batch_size=64, shuffle=False)
    emb_chunks = []
    with torch.no_grad():
        for mel in aud_loader:
            mel = mel.to(device)
            out = audio_enc(mel)
            emb_chunks.append(out)
    outputs = torch.cat(emb_chunks, dim=0).cpu()
    first_frame, last_frame = outputs[:1], outputs[-1:]
    audio_feats = torch.cat([first_frame, outputs, last_frame], dim=0).numpy()
    print(f"Audio features shape: {audio_feats.shape}")
    print(f"Audio features mean: {audio_feats.mean():.6f}\n")
    
    # Load generator
    net = Model(6, args.asr).to(device).eval()
    state = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(state, strict=False)
    print(f"Generator loaded from: {checkpoint_path}\n")
    
    # Load preprocessed images for the frame
    roi_path = os.path.join(cache_dir, "rois_320", f"{args.frame_index}.jpg")
    masked_path = os.path.join(cache_dir, "model_inputs", f"{args.frame_index}_masked.jpg")
    
    roi_img = cv2.imread(roi_path)
    masked_img = cv2.imread(masked_path)
    
    print(f"Loaded ROI: {roi_path}")
    print(f"Loaded masked: {masked_path}")
    
    # Prepare image input (same as inference_328.py)
    img_real_ex = roi_img.transpose(2, 0, 1).astype(np.float32) / 255.0
    img_masked = masked_img.transpose(2, 0, 1).astype(np.float32) / 255.0
    img_concat_T = torch.from_numpy(np.concatenate([img_real_ex, img_masked], axis=0))[None].to(device)
    
    print(f"Image input shape: {img_concat_T.shape}")
    print(f"Image input mean: {img_concat_T.mean():.6f}\n")
    
    # Get audio window for frame (same as inference_328.py)
    i = args.frame_index
    a_win = get_audio_window(audio_feats, i)
    # get_audio_window returns tensor
    a_win_t = a_win.float()
    
    if args.asr == "hubert":
        a_win_t = a_win_t.view(32, 32, 32)
    elif args.asr == "wenet":
        a_win_t = a_win_t.view(32, 32, 32)
    else:  # ave
        a_win_t = a_win_t.view(32, 16, 16)
    
    audio_T = a_win_t.unsqueeze(0).to(device)
    
    print(f"Audio input shape: {audio_T.shape}")
    print(f"Audio input mean: {audio_T.mean():.6f}\n")
    
    # Run inference (same as inference_328.py)
    with torch.no_grad():
        pred = net(img_concat_T, audio_T)[0]
    
    # Post-process (same as inference_328.py)
    pred = (pred.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    
    print(f"Output shape: {pred.shape}")
    print(f"Output range: [{pred.min()}, {pred.max()}]")
    print(f"Output mean per channel: {pred.mean(axis=(0,1))}\n")
    
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
        crop_328[4:324, 4:324] = pred
        
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
    cv2.imwrite(os.path.join(output_dir, f"frame_{args.frame_index}_roi.jpg"), roi_img)
    cv2.imwrite(os.path.join(output_dir, f"frame_{args.frame_index}_masked.jpg"), masked_img)
    cv2.imwrite(os.path.join(output_dir, f"frame_{args.frame_index}_pred.jpg"), pred)
    if full_img is not None:
        cv2.imwrite(os.path.join(output_dir, f"frame_{args.frame_index}_full.jpg"), full_img)
    
    # Save tensors
    npy_dir = os.path.join(output_dir, 'tensors_npy')
    os.makedirs(npy_dir, exist_ok=True)
    
    np.save(os.path.join(npy_dir, f'frame_{args.frame_index}_audio_input.npy'), audio_T.cpu().numpy())
    np.save(os.path.join(npy_dir, f'frame_{args.frame_index}_image_input.npy'), img_concat_T.cpu().numpy())
    np.save(os.path.join(npy_dir, f'frame_{args.frame_index}_output.npy'), pred.transpose(2, 0, 1).astype(np.float32) / 255.0)
    
    # Save JSON
    json_data = {
        "frame_index": args.frame_index,
        "asr_mode": args.asr,
        "dataset": args.name,
        "model_type": "pytorch_pth",
        "audio_features_mean": float(audio_feats.mean()),
        "audio_input_mean": float(audio_T.mean()),
        "image_input_mean": float(img_concat_T.mean()),
        "output_mean": float(pred.mean()),
        "output_shape": list(pred.shape),
        "output_mean_per_channel": pred.mean(axis=(0,1)).tolist()
    }
    
    with open(os.path.join(output_dir, f"frame_{args.frame_index}_tensors.json"), 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"Saved outputs to: {output_dir}")
    print(f"✓ Test complete!")

if __name__ == '__main__':
    main()
