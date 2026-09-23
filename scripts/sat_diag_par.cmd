@echo off
cd /d D:\Unervisity\Project09
set TORCH_THREADS=1

rem w0: 8v8 random, w1: 8v8 fixed, w2: 8v8 greedy, w3: 8v8 memory
start "sat0" /b cmd /c ".venv\Scripts\python.exe C:\Users\Taha\AppData\Local\Temp\opencode\diag_saturation.py --regimes MATE-8v8-9-v0 --seeds 0..7 --policies random > results\campaign_chao8v8\sat_w0.out 2>&1"
start "sat1" /b cmd /c ".venv\Scripts\python.exe C:\Users\Taha\AppData\Local\Temp\opencode\diag_saturation.py --regimes MATE-8v8-9-v0 --seeds 0..7 --policies fixed > results\campaign_chao8v8\sat_w1.out 2>&1"
start "sat2" /b cmd /c ".venv\Scripts\python.exe C:\Users\Taha\AppData\Local\Temp\opencode\diag_saturation.py --regimes MATE-8v8-9-v0 --seeds 0..7 --policies greedy > results\campaign_chao8v8\sat_w2.out 2>&1"
start "sat3" /b cmd /c ".venv\Scripts\python.exe C:\Users\Taha\AppData\Local\Temp\opencode\diag_saturation.py --regimes MATE-8v8-9-v0 --seeds 0..7 --policies memory > results\campaign_chao8v8\sat_w3.out 2>&1"
rem w4: 4v8 random, w5: 4v8 fixed, w6: 4v8 greedy, w7: 4v8 memory
start "sat4" /b cmd /c ".venv\Scripts\python.exe C:\Users\Taha\AppData\Local\Temp\opencode\diag_saturation.py --regimes MATE-4v8-9-v0 --seeds 0..7 --policies random > results\campaign_chao4v8win\sat_w4.out 2>&1"
start "sat5" /b cmd /c ".venv\Scripts\python.exe C:\Users\Taha\AppData\Local\Temp\opencode\diag_saturation.py --regimes MATE-4v8-9-v0 --seeds 0..7 --policies fixed > results\campaign_chao4v8win\sat_w5.out 2>&1"
start "sat6" /b cmd /c ".venv\Scripts\python.exe C:\Users\Taha\AppData\Local\Temp\opencode\diag_saturation.py --regimes MATE-4v8-9-v0 --seeds 0..7 --policies greedy > results\campaign_chao4v8win\sat_w6.out 2>&1"
start "sat7" /b cmd /c ".venv\Scripts\python.exe C:\Users\Taha\AppData\Local\Temp\opencode\diag_saturation.py --regimes MATE-4v8-9-v0 --seeds 0..7 --policies memory > results\campaign_chao4v8win\sat_w7.out 2>&1"
echo ALL_SAT_WORKERS_STARTED