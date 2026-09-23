@echo off
rem Step 4 (frozen roadmap): 4v8 confirmatory null, Chao windowed arm.
rem Seeds 16..31 = confirmatory block, never seen (exploratory 0..3 excluded).
rem Config locked identically to the 8v8 confirmatory batch.
cd /d D:\Unervisity\Project09
set TORCH_THREADS=1
.venv\Scripts\python.exe scripts\launch_campaign.py --workers 8 --tag chao4v8c --seeds 16..31 --regime MATE-4v8-9-v0 --steps 30000 --eval-episodes 3 --eval-every 10000 --reward chao --lambda-target-ratio 1.0 --env-reward-clip 8 --reward-window 20 --no-baselines > results\campaign_chao4v8c\launch_task.out 2>&1