@echo off
cd /d D:\Unervisity\Project09
> results\campaign_chao4v8c\seq_eval_s16_31.log 2>&1 (
  .venv\Scripts\python.exe scripts\eval_localization.py --tag chao4v8c --regime MATE-4v8-9-v0 --seeds 16..31 --episodes 5 --mode seq --arms intrinsic,no_intrinsic --out-suffix _s16_31
  echo CHAO_EXITCODE=%errorlevel%
)
> results\campaign_loc4v8c\seq_eval_s16_31.log 2>&1 (
  .venv\Scripts\python.exe scripts\eval_localization.py --tag loc4v8c --regime MATE-4v8-9-v0 --seeds 16..31 --episodes 5 --mode seq --arms loc,no_intrinsic --out-suffix _s16_31
  echo LOC_EXITCODE=%errorlevel%
)
echo SEQ_EVAL_CHAIN_DONE