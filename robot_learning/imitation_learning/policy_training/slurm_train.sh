#!/bin/bash
#SBATCH --job-name=policy-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=1-00:00
#SBATCH --signal=B:TERM@600
#SBATCH --output=%x-%j.out
# slurm_train.sh — train.py on a Slurm cluster, self-resubmitting until the run is done. 10 min before the wall
# limit a STOP file makes train.py save `last` and exit; the job resubmits itself (only if it made progress).
# Add your site's --account / --partition on the sbatch command line (or in SBATCH_ACCOUNT / SBATCH_PARTITION).
#
#   sbatch --export=ALL,DATASET=<name>,VENV=/path/to/venv slurm_train.sh
#   other policy / run:  sbatch --export=ALL,POLICY=act,NAME=act_v1,DATASET=<name> slurm_train.sh
#   π0.5 full fine-tune (VLM + vision + expert, fp32 AdamW; needs an 80 GB GPU):
#       sbatch --export=ALL,NAME=pi05_full,DATASET=<name>,SET="pi05.train_expert_only=false pi05.adam_8bit=false batch_size=16" slurm_train.sh
#   4 GPUs (torchrun data parallel; batch_size is per GPU):
#       sbatch --gres=gpu:4 --cpus-per-task=48 --mem=384G --export=ALL,GPUS=4,NAME=...,DATASET=... slurm_train.sh
# GPU nodes are often offline: fetch lerobot/pi05_base into $HF_HOME and run one `prepare` on a login node first.
set -uo pipefail
POLICY=${POLICY:-pi05}
NAME=${NAME:-${POLICY}_run}
DATASET=${DATASET:?set DATASET=<name or path>}
ROBOT=${ROBOT:-nova}
GPUS=${GPUS:-1}
SET=${SET:-}
HERE=${HERE:-$SLURM_SUBMIT_DIR}             # the policy_training folder (sbatch from inside it)
cd "$HERE"
[[ -n ${VENV:-} ]] && source "$VENV/bin/activate"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} USE_TF=0 USE_FLAX=0 TOKENIZERS_PARALLELISM=false \
       TRANSFORMERS_VERBOSITY=error PYTORCH_ALLOC_CONF=expandable_segments:True
RUNS=${POLICY_RUNS_ROOT:-$HERE/runs}
RESUME=(); [[ -f $RUNS/$NAME/config.json ]] && RESUME=(--resume)
step() { python -c "import json; print(json.load(open('$RUNS/$NAME/status.json')).get('step', 0))" 2>/dev/null || echo 0; }
S0=$(step)
echo "=== $(date) job=$SLURM_JOB_ID run=$NAME ${RESUME[*]}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
LAUNCH=(python -u); [[ $GPUS -gt 1 ]] && LAUNCH=(torchrun --standalone --nproc_per_node=$GPUS)
# shellcheck disable=SC2086
"${LAUNCH[@]}" train.py train --policy "$POLICY" --name "$NAME" --robot "$ROBOT" --set dataset="$DATASET" $SET "${RESUME[@]}" &
PID=$!
# a STOP file, not a signal: every rank sees it at the same step, rank 0 checkpoints, all exit
trap 'echo "=== wall in 10 min: STOP"; touch "$RUNS/$NAME/STOP"' TERM
wait $PID; wait $PID
STATE=$(python -c "import json; print(json.load(open('$RUNS/$NAME/status.json'))['state'])" 2>/dev/null || echo unknown)
S1=$(step)
echo "=== $(date) run state: $STATE, step $S0 -> $S1"
# resubmit only if this job made progress (a crash loop must not refill the queue)
if [[ $STATE != done && $STATE != evaluating && $S1 -gt $S0 ]]; then
    sbatch --gres=gpu:$GPUS --cpus-per-task=$SLURM_CPUS_PER_TASK --mem=$((SLURM_MEM_PER_NODE / 1024))G \
        --export=ALL,POLICY="$POLICY",NAME="$NAME",DATASET="$DATASET",ROBOT="$ROBOT",SET="$SET",GPUS=$GPUS,HERE="$HERE" \
        "$HERE/slurm_train.sh"
fi
