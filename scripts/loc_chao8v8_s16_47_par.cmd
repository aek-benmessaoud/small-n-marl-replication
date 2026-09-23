@echo off
cd /d D:\Unervisity\Project09
set TORCH_THREADS=1

rem worker 0: seeds 16-19
start "locw0" /b cmd /c ".venv\Scripts\python.exe scripts\analysis\eval_localization.py --tag chao8v8 --regime MATE-8v8-9-v0 --seeds 16..19 --episodes 5 --mode simult --out-suffix _p0 > results\campaign_chao8v8\loc_w0.out 2>&1 & .venv\Scripts\python.exe scripts\analysis\eval_localization.py --tag chao8v8 --regime MATE-8v8-9-v0 --seeds 16..19 --episodes 5 --mode seq --out-suffix _p0 >> results\campaign_chao8v8\loc_w0.out 2>&1"
rem worker 1: seeds 20-23
start "locw1" /b cmd /c ".venv\Scripts\python.exe scripts\analysis\eval_localization.py --tag chao8v8 --regime MATE-8v8-9-v0 --seeds 20..23 --episodes 5 --mode simult --out-suffix _p1 > results\campaign_chao8v8\loc_w1.out 2>&1 & .venv\Scripts\python.exe scripts\analysis\eval_localization.py --tag chao8v8 --regime MATE-8v8-9-v0 --seeds 20..23 --episodes 5 --mode seq --out-suffix _p1 >> results\campaign_chao8v8\loc_w1.out 2>&1"
rem worker 2: seeds 24-27
start "locw2" /b cmd /c ".venv\Scripts\python.exe scripts\analysis\eval_localization.py --tag chao8v8 --regime MATE-8v8-9-v0 --seeds 24..27 --episodes 5 --mode simult --out-suffix _p2 > results\campaign_chao8v8\loc_w2.out 2>&1 & .venv\Scripts\python.exe scripts\analysis\eval_localization.py --tag chao8v8 --regime MATE-8v8-9-v0 --seeds 24..27 --episodes 5 --mode seq --out-suffix _p2 >> results\campaign_chao8v8\loc_w2.out 2>&1"
rem worker 3: seeds 28-31
start "locw3" /b cmd /c ".venv\Scripts\python.exe scripts\analysis\eval_localization.py --tag chao8v8 --regime MATE-8v8-9-v0 --seeds 28..31 --episodes 5 --mode simult --out-suffix _p3 > results\campaign_chao8v8\loc_w3.out 2>&1 & .venv\Scripts\python.exe scripts\analysis\eval_localization.py --tag chao8v8 --regime MATE-8v8-9-v0 --seeds 28..31 --episodes 5 --mode seq --out-suffix _p3 >> results\campaign_chao8v8\loc_w3.out 2>&1"
rem worker 4: seeds 32-35
start "locw4" /b cmd /c ".venv\Scripts\python.exe scripts\analysis\eval_localization.py --tag chao8v8 --regime MATE-8v8-9-v0 --seeds 32..35 --episodes 5 --mode simult --out-suffix _p4 > results\campaign_chao8v8\loc_w4.out 2>&1 & .venv\Scripts\python.exe scripts\analysis\eval_localization.py --tag chao8v8 --regime MATE-8v8-9-v0 --seeds 32..35 --episodes 5 --mode seq --out-suffix _p4 >> results\campaign_chao8v8\loc_w4.out 2>&1"
rem worker 5: seeds 36-39
start "locw5" /b cmd /c ".venv\Scripts\python.exe scripts\analysis\eval_localization.py --tag chao8v8 --regime MATE-8v8-9-v0 --seeds 36..39 --episodes 5 --mode simult --out-suffix _p5 > results\campaign_chao8v8\loc_w5.out 2>&1 & .venv\Scripts\python.exe scripts\analysis\eval_localization.py --tag chao8v8 --regime MATE-8v8-9-v0 --seeds 36..39 --episodes 5 --mode seq --out-suffix _p5 >> results\campaign_chao8v8\loc_w5.out 2>&1"
rem worker 6: seeds 40-43
start "locw6" /b cmd /c ".venv\Scripts\python.exe scripts\analysis\eval_localization.py --tag chao8v8 --regime MATE-8v8-9-v0 --seeds 40..43 --episodes 5 --mode simult --out-suffix _p6 > results\campaign_chao8v8\loc_w6.out 2>&1 & .venv\Scripts\python.exe scripts\analysis\eval_localization.py --tag chao8v8 --regime MATE-8v8-9-v0 --seeds 40..43 --episodes 5 --mode seq --out-suffix _p6 >> results\campaign_chao8v8\loc_w6.out 2>&1"
rem worker 7: seeds 44-47
start "locw7" /b cmd /c ".venv\Scripts\python.exe scripts\analysis\eval_localization.py --tag chao8v8 --regime MATE-8v8-9-v0 --seeds 44..47 --episodes 5 --mode simult --out-suffix _p7 > results\campaign_chao8v8\loc_w7.out 2>&1 & .venv\Scripts\python.exe scripts\analysis\eval_localization.py --tag chao8v8 --regime MATE-8v8-9-v0 --seeds 44..47 --episodes 5 --mode seq --out-suffix _p7 >> results\campaign_chao8v8\loc_w7.out 2>&1"
echo ALL_LOC_WORKERS_STARTED