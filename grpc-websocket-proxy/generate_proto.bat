@echo off
REM Generate Go code from Protocol Buffers

echo Generating Go code from proto files...

REM Create pb directory if it doesn't exist
if not exist "pb" mkdir pb

REM Generate Go code
protoc --go_out=pb --go_opt=paths=source_relative ^
    --go-grpc_out=pb --go-grpc_opt=paths=source_relative ^
    optimized_lipsyncsrv.proto

if %ERRORLEVEL% EQU 0 (
    echo ✅ Proto generation successful!
    echo Generated files:
    echo   - pb/optimized_lipsyncsrv.pb.go
    echo   - pb/optimized_lipsyncsrv_grpc.pb.go
) else (
    echo ❌ Proto generation failed!
    echo Make sure protoc and Go plugins are installed:
    echo   go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
    echo   go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
    exit /b 1
)
