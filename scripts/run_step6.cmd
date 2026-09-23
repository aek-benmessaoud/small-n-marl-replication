@echo off
REM Step 6: bin-sensitivity (8/12/16/24 angular bins) on 8v8 seeds 0-3.
REM Frozen roadmap: "6 | Bin-sensitivity (8/12/16/24 bins, seeds 0-3, x2 arms
REM x 30k) | 0.96M | ~3-4h".
REM Interprets bins as equal-angular discretization of the full circle;
REM config_count per target = number of DISTINCT occupied bins (replaces the
REM greedy 15-deg clustering). Greedy ANG_TOL matches 24 bins (15 deg/bin).
REM Mirror config of step 5 (windowed chao, seeded auto-lambda, clip 8)
REM to keep every step comparable.
cd /d D:\Unervisity\Project09

set PY=.venv\Scripts\python.exe

echo [%date% %time%] step6 bins=8 >> results\step6_launch.log
%PY% scripts\launch_campaign.py --workers 8 --seeds 0..3 --tag chao8v8_b08 --regime MATE-8v8-9-v0 --steps 30000 --eval-episodes 3 --eval-every 10000 --reward chao --reward-window 20 --env-reward-clip 8 --num-bins 8 --no-baselines
if errorlevel 1 goto fail

echo [%date% %time%] step6 bins=12 >> results\step6_launch.log
%PY% scripts\launch_campaign.py --workers 8 --seeds 0..3 --tag chao8v8_b12 --regime MATE-8v8-9-v0 --steps 30000 --eval-episodes 3 --eval-every 10000 --reward chao --reward-window 20 --env-reward-clip 8 --num-bins 12 --no-baselines
if errorlevel 1 goto fail

echo [%date% %time%] step6 bins=16 >> results\step6_launch.log
%PY% scripts\launch_campaign.py --workers 8 --seeds 0..3 --tag chao8v8_b16 --regime MATE-8v8-9-v0 --steps 30000 --eval-episodes 3 --eval-every 10000 --reward chao --reward-window 20 --env-reward-clip 8 --num-bins 16 --no-baselines
if errorlevel 1 goto fail

echo [%date% %time%] step6 bins=24 >> results\step6_launch.log
%PY% scripts\launch_campaign.py --workers 8 --seeds 0..3 --tag chao8v8_b24 --regime MATE-8v8-9-v0 --steps 30000 --eval-episodes 3 --eval-every 10000 --reward chao --reward-window 20 --env-reward-clip 8 --num-bins 24 --no-baselines
if errorlevel 1 goto fail

echo [%date% %time%] STEP6 ALL DONE >> results\step6_launch.log
exit /b 0
:fail
echo [%date% %time%] STEP6 FAILED >> results\step6_launch.log
exit /b 1