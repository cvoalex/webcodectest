@echo off
echo 🔧 Setting up Go WebSocket Frame Generator...

echo 📦 Installing Go dependencies...
go mod tidy

echo 🏗️ Creating pb directory...
if not exist "pb" mkdir pb

echo 📋 Generating protobuf Go files...
protoc --go_out=. --go_opt=paths=source_relative --go-grpc_out=. --go-grpc_opt=paths=source_relative proto/lipsync.proto

echo ✅ Setup complete!
echo.
echo To run the integrated server:
echo go run main_integrated.go
echo.
echo The server will include:
echo - Token generation on :3000/token
echo - Model video serving on :3000/api/model-video/
echo - WebSocket frame generator on :3000/ws
echo.
echo This replaces both the Python frame generator (port 8080) and the original Go server (port 3000)
