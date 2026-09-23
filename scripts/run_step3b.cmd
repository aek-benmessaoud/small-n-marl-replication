@echo off
REM Step 3b: RND intrinsic in 4v8 n=16 (seeds 16-31), matching chao4v8c config.
REM Decides whether the 4v8 Chao gain (+0.0179, p=0.009) is specific to the
REM Chao-U richness signal or an artifact of ANY intrinsic novelty bonus.
cd /d D:\Unervisity\Project09
set PY=.venv\Scripts\python.exe
echo [%date% %time%] step3b rnd4v8c >> results\step3b_launch.log
%PY% scripts\launch\launch_campaign.py --workers 8 --seeds 16..31 --tag rnd4v8c --regime MATE-4v8-9-v0 --steps 30000 --eval-episodes 3 --eval-every 10000 --reward rnd --reward-window 20 --env-reward-clip 8 --no-baselines
if errorlevel 1 goto fail
echo [%date% %time%] STEP3B ALL DONE >> results\step3b_launch.log
exit /b 0
:fail
echo [%date% %time%] STEP3B FAILED >> results\step3b_launch.log
exit /b 1