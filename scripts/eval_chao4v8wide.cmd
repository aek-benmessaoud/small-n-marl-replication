@echo off
cd /d D:\Unervisity\Project09
set TORCH_THREADS=1
echo === eval_final === > results\campaign_chao4v8wide\eval_task.out
.venv\Scripts\python.exe scripts\analysis\eval_final.py --tag chao4v8wide --regime MATE-4v8-9-wide90-v0 --seeds 0..3 --episodes 5 --window 20 >> results\campaign_chao4v8wide\eval_task.out 2>&1
echo === eval_simult === >> results\campaign_chao4v8wide\eval_task.out
.venv\Scripts\python.exe scripts\analysis\eval_localization.py --tag chao4v8wide --regime MATE-4v8-9-wide90-v0 --seeds 0..3 --episodes 5 --mode simult >> results\campaign_chao4v8wide\eval_task.out 2>&1
echo === eval_seq === >> results\campaign_chao4v8wide\eval_task.out
.venv\Scripts\python.exe scripts\analysis\eval_localization.py --tag chao4v8wide --regime MATE-4v8-9-wide90-v0 --seeds 0..3 --episodes 5 --mode seq >> results\campaign_chao4v8wide\eval_task.out 2>&1
echo === DONE === >> results\campaign_chao4v8wide\eval_task.out
