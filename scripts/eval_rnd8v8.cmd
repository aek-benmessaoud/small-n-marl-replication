@echo off
cd /d D:\Unervisity\Project09
set TORCH_THREADS=1
.venv\Scripts\python.exe scripts\analysis\eval_final.py --tag rnd8v8 --regime MATE-8v8-9-v0 --seeds 0..3 --episodes 5 --window 20 --arms rnd,no_intrinsic > results\campaign_rnd8v8\eval_task.out 2>&1
.venv\Scripts\python.exe scripts\analysis\eval_localization.py --tag rnd8v8 --regime MATE-8v8-9-v0 --seeds 0..3 --episodes 3 --mode simult --arms rnd,no_intrinsic >> results\campaign_rnd8v8\eval_task.out 2>&1
.venv\Scripts\python.exe scripts\analysis\eval_localization.py --tag rnd8v8 --regime MATE-8v8-9-v0 --seeds 0..3 --episodes 3 --mode seq --arms rnd,no_intrinsic >> results\campaign_rnd8v8\eval_task.out 2>&1
