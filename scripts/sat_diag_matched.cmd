@echo off
cd /d D:\Unervisity\Project09
set TORCH_THREADS=1

rem 8v8 confirmatory seeds 16..23
start "sat0" /b cmd /c ".venv\Scripts\python.exe C:\Users\Taha\AppData\Local\Temp\opencode\diag_saturation.py --regimes MATE-8v8-9-v0 --seeds 16..23 --policies random > results\campaign_chao8v8\satm_w0.out 2>&1"
start "sat1" /b cmd /c ".venv\Scripts\python.exe C:\Users\Taha\AppData\Local\Temp\opencode\diag_saturation.py --regimes MATE-8v8-9-v0 --seeds 16..23 --policies fixed > results\campaign_chao8v8\satm_w1.out 2>&1"
start "sat2" /b cmd /c ".venv\Scripts\python.exe C:\Users\Taha\AppData\Local\Temp\opencode\diag_saturation.py --regimes MATE-8v8-9-v0 --seeds 16..23 --policies greedy > results\campaign_chao8v8\satm_w2.out 2>&1"
start "sat3" /b cmd /c ".venv\Scripts\python.exe C:\Users\Taha\AppData\Local\Temp\opencode\diag_saturation.py --regimes MATE-8v8-9-v0 --seeds 16..23 --policies memory > results\campaign_chao8v8\satm_w3.out 2>&1"
rem 4v8 seeds 0..3
start "sat4" /b cmd /c ".venv\Scripts\python.exe C:\Users\Taha\AppData\Local\Temp\opencode\diag_saturation.py --regimes MATE-4v8-9-v0 --seeds 0..3 --policies random > results\campaign_chao4v8win\satm_w4.out 2>&1"
start "sat5" /b cmd /c ".venv\Scripts\python.exe C:\Users\Taha\AppData\Local\Temp\opencode\diag_saturation.py --regimes MATE-4v8-9-v0 --seeds 0..3 --policies fixed > results\campaign_chao4v8win\satm_w5.out 2>&1"
start "sat6" /b cmd /c ".venv\Scripts\python.exe C:\Users\Taha\AppData\Local\Temp\opencode\diag_saturation.py --regimes MATE-4v8-9-v0 --seeds 0..3 --policies greedy > results\campaign_chao4v8win\satm_w6.out 2>&1"
start "sat7" /b cmd /c ".venv\Scripts\python.exe C:\Users\Taha\AppData\Local\Temp\opencode\diag_saturation.py --regimes MATE-4v8-9-v0 --seeds 0..3 --policies memory > results\campaign_chao4v8win\satm_w7.out 2>&1"
echo ALL_SAT_MATCHED_STARTED