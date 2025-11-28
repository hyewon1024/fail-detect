#!/bin/bash
# Run the same RND-DAgger experiment sequentially for multiple seeds.

set -e

# Base hyperparameters (same as run.sh)
K_ITER=200
STEPS_PER_ITER=2000
INIT_STEPS=2000
EVAL_ITERS=0
LAMBDA=0.01
MIN_DEMO=70
HIST=0
EPOCH=10

# Key parameters
SEEDS=(2 13 42)
ALPHAS=(0.5 0.7 0.9)
ALG="Safe_Dagger"    # Pure_Dagger, R_Dagger, Balanced_Dagger, Safe_Dagger

# BETA is only used in Pure_Dagger
if [ "$ALG" = "Pure_Dagger" ]; then
  BETA=0.95
else
  BETA=""
fi

if [ -z "$BETA" ]; then
  BETA_ARG=""
else
  BETA_ARG="--beta $BETA"
fi

echo "=============================="
echo " Running sequential seeds: ${SEEDS[*]}"
echo " Alphas: ${ALPHAS[*]}"
echo "=============================="

for ALPHA in "${ALPHAS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "------------------------------"
    echo " SEED               : $SEED"
    echo " ALG                : $ALG"
    echo " ALPHA              : $ALPHA"
    echo " K_ITER             : $K_ITER"
    echo " STEPS_PER_ITER     : $STEPS_PER_ITER"
    echo " INIT_STEPS         : $INIT_STEPS"
    echo " EVAL_ITERS         : $EVAL_ITERS"
    echo " LAMBDA_THRESHOLD   : $LAMBDA"
    echo " MIN_DEMO_TIME      : $MIN_DEMO"
    echo " HISTORIC_CONTEXT   : $HIST"
    echo " EPOCH              : $EPOCH"
    echo " BETA               : ${BETA:-None}"
    echo "------------------------------"

    python run_RND.py --kit_args "--alg $ALG \
--alpha $ALPHA \
$BETA_ARG \
--seed $SEED \
--k_iter $K_ITER \
--steps_per_iter $STEPS_PER_ITER \
--init_steps $INIT_STEPS \
--eval_iters $EVAL_ITERS \
--lambda_th $LAMBDA \
--min_demo $MIN_DEMO \
--hist $HIST \
--epoch $EPOCH"
  done
done
