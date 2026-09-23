@echo off
cd /d D:\Unervisity\Project09
set TORCH_THREADS=1
.venv\Scripts\python.exe scripts\launch\launch_campaign.py --workers 4 --tag rnd8v8 --seeds 0..3 --regime MATE-8v8-9-v0 --steps 30000 --eval-episodes 3 --eval-every 10000 --reward rnd --lambda-target-ratio 1.0 --env-reward-clip 8 --skip-no-intrinsic --no-baselines > results\campaign_rnd8v8\launch_task.out 2>&1
