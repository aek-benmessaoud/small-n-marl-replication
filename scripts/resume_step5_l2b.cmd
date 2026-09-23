@echo off
cd /d D:\Unervisity\Project09
echo [%date% %time%] RESUME l2_0 seeds 1-3 no_intrinsic (2nd attempt) >> results\step5_launch.log
for %%s in (1 2 3) do (
  start "l2res_s0%%s" /min cmd /c ".venv\Scripts\python.exe scripts\run_mappo.py --seed %%s --tag chao8v8_l2_0 --regime MATE-8v8-9-v0 --steps 30000 --eval-episodes 3 --eval-every 10000 --reward chao --reward-window 20 --env-reward-clip 8 --timestamp 20260921_162233 --arm no_intrinsic > results\campaign_chao8v8_l2_0\resume_l2_s0%%s.log 2>&1"
)
echo STEP5_RATIO2_RESUME2_LAUNCHED >> results\step5_launch.log