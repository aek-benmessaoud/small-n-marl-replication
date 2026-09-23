@echo off
cd /d D:\Unervisity\Project09
set TORCH_THREADS=1
.venv\Scripts\python.exe scripts\launch_campaign.py --workers 4 --tag chao8v8 --seeds 8..11 --regime MATE-8v8-9-v0 --steps 30000 --eval-episodes 3 --eval-every 10000 --reward chao --lambda-target-ratio 1.0 --env-reward-clip 8 --reward-window 20 --no-baselines > results\campaign_chao8v8\launch_task_s8.out 2>&1
