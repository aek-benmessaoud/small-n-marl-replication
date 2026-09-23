@echo off
REM Step 5: lambda sweep (lambda/2, lambda, 2lambda) on 8v8 seeds 0-3.
REM Frozen roadmap: "5 | lambda sweep (lambda/2, lambda, 2lambda) on 8v8 seeds
REM 0-3 (24 runs x 30k) | 0.72M | ~3h".
REM Per-seed auto-lambda calibration with target ratio 0.5 / 1.0 / 2.0.
cd /d D:\Unervisity\Project09

set PY=.venv\Scripts\python.exe

echo [%date% %time%] step5 lambda/2 (ratio 0.5) >> results\step5_launch.log
%PY% scripts\launch\launch_campaign.py --workers 8 --seeds 0..3 --tag chao8v8_l0_5 --regime MATE-8v8-9-v0 --steps 30000 --eval-episodes 3 --eval-every 10000 --reward chao --lambda-target-ratio 0.5 --reward-window 20 --env-reward-clip 8 --no-baselines
if errorlevel 1 goto fail

echo [%date% %time%] step5 lambda x1 (ratio 1.0) >> results\step5_launch.log
%PY% scripts\launch\launch_campaign.py --workers 8 --seeds 0..3 --tag chao8v8_l1_0 --regime MATE-8v8-9-v0 --steps 30000 --eval-episodes 3 --eval-every 10000 --reward chao --lambda-target-ratio 1.0 --reward-window 20 --env-reward-clip 8 --no-baselines
if errorlevel 1 goto fail

echo [%date% %time%] step5 lambda x2 (ratio 2.0) >> results\step5_launch.log
%PY% scripts\launch\launch_campaign.py --workers 8 --seeds 0..3 --tag chao8v8_l2_0 --regime MATE-8v8-9-v0 --steps 30000 --eval-episodes 3 --eval-every 10000 --reward chao --lambda-target-ratio 2.0 --reward-window 20 --env-reward-clip 8 --no-baselines
if errorlevel 1 goto fail

echo [%date% %time%] STEP5 ALL DONE >> results\step5_launch.log
exit /b 0
:fail
echo [%date% %time%] STEP5 FAILED >> results\step5_launch.log
exit /b 1