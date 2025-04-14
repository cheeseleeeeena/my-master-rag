#!/bin/bash

# set tmux session name
SESH="expB5"

# set project dir
PROJECT_DIR="/dataspace/P76125041/my-master-rag/experiments/B4"

# set huggingface cache dir
HF_HOME_DIR="/dataspace/P76125041/.cache/"

# set conda environment name
ENV_NAME="vllm5"

# check if session exists
tmux has-session -t $SESH 2>/dev/null

# couple things to do when creating new session on new machine
if [ $? != 0 ]; then
    # create new detached session with new window "nvtop"
    tmux new-session -d -s $SESH -n "nvtop"
    tmux send-keys -t $SESH:nvtop "nvtop" C-m
    
    # create new window for MAIN program
    tmux new-window -t $SESH -n "main"
    tmux send-keys -t $SESH:main "export HF_HOME=$HF_HOME_DIR" C-m
    tmux send-keys -t $SESH:main "cd $PROJECT_DIR" C-m
    tmux send-keys -t $SESH:main "conda activate $ENV_NAME" C-m

    # create new window for INDEXER program
    tmux new-window -t $SESH -n "indexer"
    tmux send-keys -t $SESH:indexer "export HF_HOME=$HF_HOME_DIR" C-m
    tmux send-keys -t $SESH:indexer "cd $PROJECT_DIR" C-m
    tmux send-keys -t $SESH:indexer "conda activate $ENV_NAME" C-m

    # create new window for PREDICTOR program
    tmux new-window -t $SESH -n "predict"
    tmux send-keys -t $SESH:predict "export HF_HOME=$HF_HOME_DIR" C-m
    tmux send-keys -t $SESH:predict "cd $PROJECT_DIR" C-m
    tmux send-keys -t $SESH:predict "conda activate $ENV_NAME" C-m
    
    # create new window for EVALUATOR program
    tmux new-window -t $SESH -n "eval"
    tmux send-keys -t $SESH:eval "export HF_HOME=$HF_HOME_DIR" C-m
    tmux send-keys -t $SESH:eval "cd $PROJECT_DIR" C-m
    tmux send-keys -t $SESH:eval "conda activate $ENV_NAME" C-m

    # create new window for SCP ops
    tmux new-window -t $SESH -n "scp"
    tmux send-keys -t $SESH:scp "export HF_HOME=$HF_HOME_DIR" C-m
    tmux send-keys -t $SESH:scp "cd /workspace/P76125041/my-master-rag/" C-m
    # tmux send-keys -t $SESH:scp "tree ." C-m
    tmux send-keys -t $SESH:scp "conda activate $ENV_NAME" C-m

    # create new window for misc ops
    tmux new-window -t $SESH -n "misc"
    tmux send-keys -t $SESH:misc "export HF_HOME=$HF_HOME_DIR" C-m
    tmux send-keys -t $SESH:misc "cd $PROJECT_DIR" C-m
    # tmux send-keys -t $SESH:misc "tree ." C-m
    tmux send-keys -t $SESH:misc "conda activate $ENV_NAME" C-m


    # Switch back to the main window
    tmux select-window -t $SESH:main
fi

# Attach to the session
tmux attach-session -t $SESH