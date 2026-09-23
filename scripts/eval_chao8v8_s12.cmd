@echo off
cd /d D:\Unervisity\Project09
set TORCH_THREADS=1
.venv\Scripts\python.exe scripts\analysis\eval_final.py --tag chao8v8 --regime MATE-8v8-9-v0 --seeds 12..15 --episodes 5 --window 20 > results\campaign_chao8v8\eval_task_s12.out 2>&1
.venv\Scripts\python.exe scripts\analysis\eval_localization.py --tag chao8v8 --regime MATE-8v8-9-v0 --seeds 12..15 --episodes 5 --mode simult >> results\campaign_chao8v8\eval_task_s12.out 2>&1
.venv\Scripts\python.exe scripts\analysis\eval_localization.py --tag chao8v8 --regime MATE-8v8-9-v0 --seeds 12..15 --episodes 5 --mode seq >> results\campaign_chao8v8\eval_task_s12.out 2>&1