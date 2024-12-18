export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
python ../../rollout_rl.py "$@" \
    --exp_name=connector_insert \
    --checkpoint_path=PATH/TO/CHECKPOINT \
    --eval_checkpoint_step STEP \
    --eval_n_trajs N_TRAJS