@echo off
cd /d D:\Unervisity\Project09
set TORCH_THREADS=1
.venv\Scripts\python.exe scripts\launch\launch_campaign.py --workers 4 --tag chao4v8wide --seeds 0..3 --regime MATE-4v8-9-wide90-v0 --steps 30000 --eval-episodes 3 --eval-every 10000 --reward chao --lambda-target-ratio 1.0 --env-reward-clip 8 --reward-window 20 --no-baselines > results\campaign_chao4v8wide\launch_task.out 2>&1
