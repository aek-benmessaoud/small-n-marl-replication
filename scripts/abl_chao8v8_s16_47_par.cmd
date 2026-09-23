@echo off
cd /d D:\Unervisity\Project09
set TORCH_THREADS=1

rem worker 0: seeds 16-19
start "abl0" /b cmd /c ".venv\Scripts\python.exe C:\Users\Taha\AppData\Local\Temp\opencode\sens_seqloc.py --mode ablate --seeds 16..19 --arm intrinsic --episodes 5 > results\campaign_chao8v8\abl_w0.out 2>&1"
rem worker 1: seeds 20-23
start "abl1" /b cmd /c ".venv\Scripts\python.exe C:\Users\Taha\AppData\Local\Temp\opencode\sens_seqloc.py --mode ablate --seeds 20..23 --arm intrinsic --episodes 5 > results\campaign_chao8v8\abl_w1.out 2>&1"
rem worker 2: seeds 24-27
start "abl2" /b cmd /c ".venv\Scripts\python.exe C:\Users\Taha\AppData\Local\Temp\opencode\sens_seqloc.py --mode ablate --seeds 24..27 --arm intrinsic --episodes 5 > results\campaign_chao8v8\abl_w2.out 2>&1"
rem worker 3: seeds 28-31
start "abl3" /b cmd /c ".venv\Scripts\python.exe C:\Users\Taha\AppData\Local\Temp\opencode\sens_seqloc.py --mode ablate --seeds 28..31 --arm intrinsic --episodes 5 > results\campaign_chao8v8\abl_w3.out 2>&1"
rem worker 4: seeds 32-35
start "abl4" /b cmd /c ".venv\Scripts\python.exe C:\Users\Taha\AppData\Local\Temp\opencode\sens_seqloc.py --mode ablate --seeds 32..35 --arm intrinsic --episodes 5 > results\campaign_chao8v8\abl_w4.out 2>&1"
rem worker 5: seeds 36-39
start "abl5" /b cmd /c ".venv\Scripts\python.exe C:\Users\Taha\AppData\Local\Temp\opencode\sens_seqloc.py --mode ablate --seeds 36..39 --arm intrinsic --episodes 5 > results\campaign_chao8v8\abl_w5.out 2>&1"
rem worker 6: seeds 40-43
start "abl6" /b cmd /c ".venv\Scripts\python.exe C:\Users\Taha\AppData\Local\Temp\opencode\sens_seqloc.py --mode ablate --seeds 40..43 --arm intrinsic --episodes 5 > results\campaign_chao8v8\abl_w6.out 2>&1"
rem worker 7: seeds 44-47
start "abl7" /b cmd /c ".venv\Scripts\python.exe C:\Users\Taha\AppData\Local\Temp\opencode\sens_seqloc.py --mode ablate --seeds 44..47 --arm intrinsic --episodes 5 > results\campaign_chao8v8\abl_w7.out 2>&1"
echo ALL_ABL_WORKERS_STARTED