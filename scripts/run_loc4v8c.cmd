@echo off
rem Step 4 (frozen roadmap): 4v8 confirmatory null, Loc objective arm.
rem Seeds 16..31 = confirmatory block, never seen. Same locked config.
cd /d D:\Unervisity\Project09
set TORCH_THREADS=1
.venv\Scripts\python.exe scripts\launch\launch_campaign.py --workers 8 --tag loc4v8c --seeds 16..31 --regime MATE-4v8-9-v0 --steps 30000 --eval-episodes 3 --eval-every 10000 --reward loc --lambda-target-ratio 1.0 --env-reward-clip 8 --reward-window 20 --no-baselines > results\campaign_loc4v8c\launch_task.out 2>&1