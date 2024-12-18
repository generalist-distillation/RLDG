export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
python ../../train_rlpd.py "$@" \
    --exp_name=connector_insert \
    --actor \
    --checkpoint_path=/PATH/TO/SAVE/CHECKPOINT \
