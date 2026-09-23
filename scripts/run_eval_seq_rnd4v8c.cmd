@echo off
cd /d D:\Unervisity\Project09
> results\campaign_rnd4v8c\seq_eval_s16_31.log 2>&1 (
  .venv\Scripts\python.exe scripts\eval_localization.py --tag rnd4v8c --regime MATE-4v8-9-v0 --seeds 16..31 --episodes 5 --mode seq --arms rnd,no_intrinsic --out-suffix _s16_31
  echo RND_EXITCODE=%errorlevel%
)
echo SEQ_EVAL_RND_DONE