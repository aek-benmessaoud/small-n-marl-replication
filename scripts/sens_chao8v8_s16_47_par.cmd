@echo off
cd /d D:\Unervisity\Project09
set TORCH_THREADS=1

rem worker 0: seeds 16-19
start "sens0" /b cmd /c ".venv\Scripts\python.exe C:\Users\Taha\AppData\Local\Temp\opencode\sens_seqloc.py --mode random --seeds 16..19 --episodes 5 > results\campaign_chao8v8\sens_w0.out 2>&1 & .venv\Scripts\python.exe C:\Users\Taha\AppData\Local\Temp\opencode\sens_seqloc.py --mode ablate --seeds 16..19 --arm intrinsic --episodes 5 >> results\campaign_chao8v8\sens_w0.out 2>&1"
rem worker 1: seeds 20-23
start "sens1" /b cmd /c ".venv\Scripts\python.exe C:\Users\Taha\AppData\Local\Temp\opencode\sens_seqloc.py --mode random --seeds 20..23 --episodes 5 > results\campaign_chao8v8\sens_w1.out 2>&1 & .venv\Scripts\python.exe C:\Users\Taha\AppData\Local\Temp\opencode\sens_seqloc.py --mode ablate --seeds 20..23 --arm intrinsic --episodes 5 >> results\campaign_chao8v8\sens_w1.out 2>&1"
rem worker 2: seeds 24-27
start "sens2" /b cmd /c ".venv\Scripts\python.exe C:\Users\Taha\AppData\Local\Temp\opencode\sens_seqloc.py --mode random --seeds 24..27 --episodes 5 > results\campaign_chao8v8\sens_w2.out 2>&1 & .venv\Scripts\python.exe C:\Users\Taha\AppData\Local\Temp\opencode\sens_seqloc.py --mode ablate --seeds 24..27 --arm intrinsic --episodes 5 >> results\campaign_chao8v8\sens_w2.out 2>&1"
rem worker 3: seeds 28-31
start "sens3" /b cmd /c ".venv\Scripts\python.exe C:\Users\Taha\AppData\Local\Temp\opencode\sens_seqloc.py --mode random --seeds 28..31 --episodes 5 > results\campaign_chao8v8\sens_w3.out 2>&1 & .venv\Scripts\python.exe C:\Users\Taha\AppData\Local\Temp\opencode\sens_seqloc.py --mode ablate --seeds 28..31 --arm intrinsic --episodes 5 >> results\campaign_chao8v8\sens_w3.out 2>&1"
rem worker 4: seeds 32-35
start "sens4" /b cmd /c ".venv\Scripts\python.exe C:\Users\Taha\AppData\Local\Temp\opencode\sens_seqloc.py --mode random --seeds 32..35 --episodes 5 > results\campaign_chao8v8\sens_w4.out 2>&1 & .venv\Scripts\python.exe C:\Users\Taha\AppData\Local\Temp\opencode\sens_seqloc.py --mode ablate --seeds 32..35 --arm intrinsic --episodes 5 >> results\campaign_chao8v8\sens_w4.out 2>&1"
rem worker 5: seeds 36-39
start "sens5" /b cmd /c ".venv\Scripts\python.exe C:\Users\Taha\AppData\Local\Temp\opencode\sens_seqloc.py --mode random --seeds 36..39 --episodes 5 > results\campaign_chao8v8\sens_w5.out 2>&1 & .venv\Scripts\python.exe C:\Users\Taha\AppData\Local\Temp\opencode\sens_seqloc.py --mode ablate --seeds 36..39 --arm intrinsic --episodes 5 >> results\campaign_chao8v8\sens_w5.out 2>&1"
rem worker 6: seeds 40-43
start "sens6" /b cmd /c ".venv\Scripts\python.exe C:\Users\Taha\AppData\Local\Temp\opencode\sens_seqloc.py --mode random --seeds 40..43 --episodes 5 > results\campaign_chao8v8\sens_w6.out 2>&1 & .venv\Scripts\python.exe C:\Users\Taha\AppData\Local\Temp\opencode\sens_seqloc.py --mode ablate --seeds 40..43 --arm intrinsic --episodes 5 >> results\campaign_chao8v8\sens_w6.out 2>&1"
rem worker 7: seeds 44-47
start "sens7" /b cmd /c ".venv\Scripts\python.exe C:\Users\Taha\AppData\Local\Temp\opencode\sens_seqloc.py --mode random --seeds 44..47 --episodes 5 > results\campaign_chao8v8\sens_w7.out 2>&1 & .venv\Scripts\python.exe C:\Users\Taha\AppData\Local\Temp\opencode\sens_seqloc.py --mode ablate --seeds 44..47 --arm intrinsic --episodes 5 >> results\campaign_chao8v8\sens_w7.out 2>&1"
echo ALL_SENS_WORKERS_STARTED