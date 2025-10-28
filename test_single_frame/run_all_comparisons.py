#!/usr/bin/env python3
"""
Compare single frame output between:
1. Go Monolithic Server (ONNX)
2. PyTorch (.pth) version
3. ONNX standalone version
"""
import subprocess
import os
import sys

def run_go_server_test():
    """Run single frame test on Go monolithic server"""
    print("\n" + "="*70)
    print("1️⃣  TESTING GO MONOLITHIC SERVER (ONNX)")
    print("="*70)
    
    os.chdir("../go-monolithic-server")
    
    # Run test_batch_25_full but modify to save just frame 0
    result = subprocess.run(
        ["./testing/test_batch_25_full.exe"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        # Copy frame 0 to comparison directory
        src = "test_output/batch_25_full/frame_000000.jpg"
        dst = "../test_single_frame/output_go_monolithic_onnx_frame0.jpg"
        if os.path.exists(src):
            import shutil
            shutil.copy(src, dst)
            print(f"✅ Saved: {dst}")
            return True
    else:
        print(f"❌ Failed: {result.stderr}")
        return False

def run_pth_test():
    """Run PyTorch version test"""
    print("\n" + "="*70)
    print("2️⃣  TESTING PYTORCH (.pth) VERSION")
    print("="*70)
    
    os.chdir("../test_single_frame")
    
    # Activate venv and run
    if sys.platform == "win32":
        python_exe = r"D:\Projects\.venv312\Scripts\python.exe"
    else:
        python_exe = "python"
    
    result = subprocess.run(
        [python_exe, "test_single_frame_pth.py"],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.returncode != 0:
        print(f"❌ Error: {result.stderr}")
        return False
    
    # Should have created output_pth_frame0.jpg or similar
    print("✅ PyTorch test complete")
    return True

def run_onnx_test():
    """Run standalone ONNX version test"""
    print("\n" + "="*70)
    print("3️⃣  TESTING STANDALONE ONNX VERSION")
    print("="*70)
    
    os.chdir("../test_single_frame")
    
    if sys.platform == "win32":
        python_exe = r"D:\Projects\.venv312\Scripts\python.exe"
    else:
        python_exe = "python"
    
    result = subprocess.run(
        [python_exe, "test_single_frame_onnx.py"],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.returncode != 0:
        print(f"❌ Error: {result.stderr}")
        return False
    
    print("✅ ONNX test complete")
    return True

def compare_outputs():
    """Compare the output images"""
    print("\n" + "="*70)
    print("📊 COMPARISON SUMMARY")
    print("="*70)
    
    files = [
        "output_go_monolithic_onnx_frame0.jpg",
        # Look for whatever the pth/onnx scripts create
    ]
    
    os.chdir("../test_single_frame")
    
    print("\n📁 Output files created:")
    for f in os.listdir("."):
        if f.startswith("output_") and f.endswith(".jpg"):
            size = os.path.getsize(f)
            print(f"   {f:50s} ({size:,} bytes)")
    
    print("\n💡 Compare images visually:")
    print("   explorer.exe .")

if __name__ == "__main__":
    original_dir = os.getcwd()
    
    try:
        # Run all tests
        run_go_server_test()
        run_pth_test()
        run_onnx_test()
        compare_outputs()
        
    finally:
        os.chdir(original_dir)
    
    print("\n✅ All tests complete!")
