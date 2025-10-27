# Installing MinGW-w64 for Go CGO

Go needs a C compiler (GCC) to use CGO, which is required for ONNX Runtime bindings.

## Option 1: TDM-GCC (Easiest)

### Download and Install
1. Download TDM-GCC from: https://jmeubank.github.io/tdm-gcc/
2. Choose "TDM-GCC 64-bit" 
3. Download the installer (tdm64-gcc-*.exe)
4. Run installer with default options
5. **Important**: Check "Add to PATH" during installation

### Verify Installation
Open a NEW terminal and run:
```powershell
gcc --version
```

You should see something like:
```
gcc.exe (tdm64-1) 10.3.0
```

## Option 2: MSYS2 MinGW

### Install MSYS2
1. Download from: https://www.msys2.org/
2. Run the installer
3. Open MSYS2 terminal and run:
```bash
pacman -S mingw-w64-x86_64-gcc
```

### Add to PATH
Add `C:\msys64\mingw64\bin` to your PATH environment variable.

## Option 3: WinLibs

### Download
1. Go to: https://winlibs.com/
2. Download the latest "GCC 13.x + MinGW-w64 11.x (UCRT)" 
3. Extract to `C:\mingw64\`

### Add to PATH
```powershell
$env:PATH = "C:\mingw64\bin;$env:PATH"
[System.Environment]::SetEnvironmentVariable('PATH', "C:\mingw64\bin;$([System.Environment]::GetEnvironmentVariable('PATH', 'User'))", 'User')
```

## After Installation

1. Close ALL terminal windows
2. Open a new terminal
3. Verify: `gcc --version`
4. Go back to `go-onnx-inference` directory
5. Try building:

```powershell
cd d:\Projects\webcodecstest\go-onnx-inference
go build ./cmd/simple-test/main.go
```

If successful, you should get `simple-test.exe`

## Quick Test Without MinGW

If you just want to see if it would work, you can try using an alternative ONNX library that doesn't require CGO, but it will be slower. For now, I recommend installing MinGW for the best performance.
