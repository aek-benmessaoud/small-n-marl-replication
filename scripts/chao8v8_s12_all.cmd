@echo off
cd /d D:\Unervisity\Project09
echo [%date% %time%] chao8v8 EXTENSION n=16 (seeds 12-15) >> results\campaign_chao8v8\extension_n16.log
echo Pre-committed: stop at n=16, no peeking at n=14. >> results\campaign_chao8v8\extension_n16.log
call scripts\run_chao8v8_s12.cmd
call scripts\eval_chao8v8_s12.cmd
echo [%date% %time%] ALL DONE >> results\campaign_chao8v8\extension_n16.log