@echo off
REM eval_slow_split.cmd — split chao8v8 slow env evals by seed range.
REM usage: eval_slow_split.cmd <RANGE> <MODE>
setlocal
cd /d D:\Unervisity\Project09
set RANGE=%~1
set MODE=%~2
set TAG=chao8v8
set REGIME=MATE-8v8-9-slow-v0
set EPISODES=5
set CAP=1000
set SUFFIX=_%RANGE:-=_%
set LOG=results\campaign_%TAG%\eval_localization\slow_%MODE%_%SUFFIX%.out
echo [%date% %time%] eval_localization %TAG% %REGIME% mode=%MODE% range=%RANGE% >> %LOG% 2>&1
.venv\Scripts\python.exe scripts\analysis\eval_localization.py --tag %TAG% --regime %REGIME% --seeds %RANGE% --episodes %EPISODES% --mode %MODE% --cap %CAP% --out-suffix %SUFFIX% >> %LOG% 2>&1
echo [%date% %time%] DONE_%MODE%_%SUFFIX% >> %LOG% 2>&1