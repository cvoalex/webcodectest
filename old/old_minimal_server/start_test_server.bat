@echo off
cd /d D:\Projects\webcodecstest
call .venv312\Scripts\activate.bat
cd minimal_server
python optimized_grpc_server.py --port 50052
pause
