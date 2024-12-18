export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.4 && \
python ../../train_rlpd.py "$@" \
    --exp_name=connector_insert \
    --learner \
    --checkpoint_path=/PATH/TO/SAVE/CHECKPOINT \
    --demo_path=/PATH/TO/DEMO/DATA \
    --online_demo_path=/PATH/TO/DEMO/DATA \