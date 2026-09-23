@echo off
cd /d D:\Unervisity\Project09

echo [%date% %time%] ORCHESTRATOR: waiting for step5-resume + loc-eval, then launching step6 >> orphan.log

:wait_resume
set /a L2DONE=0
for %%s in (0 1 2 3) do (
  for /f %%d in ('dir /b /s results\campaign_chao8v8_l2_0_s0%%s_no_intrinsic\DONE 2^>nul ^| find /c "DONE"') do set /a L2DONE+=%%d
)
if %L2DONE% lss 4 (
  timeout /t 120 /nobreak >nul
  goto wait_resume
)
echo [%date% %time%] step5 l2_0 complete (%L2DONE% DONE) >> orphan.log

:wait_loc
set LOCJSON=results\campaign_loc4v8c\eval_localization\MATE-4v8-9-v0_seq_ep5_s16_31.json
if not exist "%LOCJSON%" (
  timeout /t 120 /nobreak >nul
  goto wait_loc
)
echo [%date% %time%] loc seq eval JSON present >> orphan.log

if exist scripts\run_step6.cmd (
  call scripts\run_step6.cmd
  echo [%date% %time%] STEP6 LAUNCHED, rc=%errorlevel% >> orphan.log
)

:done
echo [%date% %time%] ORCHESTRATOR DONE >> orphan.log