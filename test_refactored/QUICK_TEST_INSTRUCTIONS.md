# Quick Test Instructions for Refactored Server

## Step-by-Step Testing

### Step 1: Start the Refactored Server

Open a **NEW PowerShell terminal** and run:

```powershell
cd d:\Projects\webcodecstest\go-monolithic-server-refactored
go run cmd/server/main.go
```

**Wait for this message:**
```
✅ Ready to accept connections!
```

**Leave this terminal open!** The server must keep running.

---

### Step 2: Run the Performance Test

Open **ANOTHER PowerShell terminal** and run:

```powershell
cd d:\Projects\webcodecstest\test_refactored
go run test_refactored.go
```

---

## Expected Test Output

The test will run **3 warmup iterations** + **3 timed iterations** and show:

```
🎬 BATCH 8 REAL DATA TEST - Monolithic Server
======================================================================
🔌 Connecting to monolithic server at localhost:50053...
✅ Connected successfully

📊 Checking server health...
✅ Server Status: Healthy (Models: 1/40, GPUs: [0])

🎵 Loading REAL audio file: ../go-monolithic-server/testing/aud.wav
✅ Loaded XXXXX samples (X.XX seconds at 16000 Hz)

📁 Output directory: output_refactored

🔥 Running 3 warmup iterations...
⏱️  Running 3 timed test iterations...
   Run 1: XX.XX FPS (Audio: XX.XXms, Infer: XXX.XXms, Composite: X.XXms, Total: XXX.XXms)
   Run 2: XX.XX FPS (Audio: XX.XXms, Infer: XXX.XXms, Composite: X.XXms, Total: XXX.XXms)
   Run 3: XX.XX FPS (Audio: XX.XXms, Infer: XXX.XXms, Composite: X.XXms, Total: XXX.XXms)

======================================================================
📊 REFACTORED SERVER - FINAL RESULTS (Average over 3 runs)
======================================================================
⚡ Average FPS: XX.XX
🎵 Average Audio Processing: XX.XX ms
🧠 Average Inference: XXX.XX ms
🎨 Average Compositing: X.XX ms
⏱️  Average Total: XXX.XX ms
======================================================================
```

---

## Success Criteria

The refactored server should achieve:

- **≥42 FPS** (batch size 8)
- **Audio Processing:** ~23ms
- **Inference:** ~165ms
- **Compositing:** ~4ms
- **Total:** ~192ms

If you see these numbers, the refactoring was **successful**! ✅

---

## Output Files

After the test completes, check:

```powershell
ls output_refactored\
```

You should see:
- `frame_0_refactored.jpg` - Sample output frame

---

## Troubleshooting

### Error: "connection refused"
- Make sure Step 1 is complete (server running)
- Check server terminal shows "✅ Ready to accept connections!"

### Error: "package not found"
- Already fixed with `go mod tidy`
- If still an issue, run: `cd test_refactored; go mod tidy`

### Server crashes immediately
- Check if port 50053 is already in use
- Try: `netstat -ano | findstr :50053`
- If port is busy, kill the process or change port in config.yaml

---

## Alternative: Use Existing Test

You can also use the original test from go-monolithic-server:

```powershell
# Terminal 1: Start refactored server
cd d:\Projects\webcodecstest\go-monolithic-server-refactored
go run cmd/server/main.go

# Terminal 2: Run existing test
cd d:\Projects\webcodecstest\go-monolithic-server\testing
go run test_batch_8_real.go
```

This test is already proven to work and will show the same metrics.

---

## Manual Comparison

If you want to compare original vs refactored:

### Test Original Server

```powershell
# Terminal 1
cd d:\Projects\webcodecstest\go-monolithic-server
go run cmd/server/main.go

# Terminal 2
cd d:\Projects\webcodecstest\go-monolithic-server\testing
go run test_batch_8_real.go
# Note the FPS and timing
```

### Test Refactored Server

```powershell
# Terminal 1 (stop original, start refactored)
cd d:\Projects\webcodecstest\go-monolithic-server-refactored
go run cmd/server/main.go

# Terminal 2 (run same test)
cd d:\Projects\webcodecstest\go-monolithic-server\testing
go run test_batch_8_real.go
# Compare FPS and timing to original
```

The numbers should be **nearly identical** (within ±1 FPS).

---

**Ready?** Start with Step 1 above! 🚀
