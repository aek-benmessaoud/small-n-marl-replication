@echo off
rem Step 4 (frozen roadmap): 4v8 confirmatory null, seeds 16..31.
rem Run the two reward arms SEQUENTIALLY (8 workers each) to avoid
rem over-subscription on the 4P+4T host.
cd /d D:\Unervisity\Project09
call scripts\run_chao4v8c.cmd
if errorlevel 1 (
  echo CHAO4V8C_FAILED >> results\campaign_chao4v8c\launch_task.out
  exit /b 1
)
call scripts\run_loc4v8c.cmd
if errorlevel 1 (
  echo LOC4V8C_FAILED >> results\campaign_loc4v8c\launch_task.out
  exit /b 1
)
echo STEP4_ALL_DONE