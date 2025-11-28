#!/bin/bash
# Run RND-DAgger experiments sequentially for multiple seeds (and alphas for non-Pure modes).

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
ALG="Pure_Dagger"    # Pure_Dagger, R_Dagger, Balanced_Dagger, Safe_Dagger

echo "=============================="
echo " Running sequential seeds: ${SEEDS[*]}"
echo " Alphas: ${ALPHAS[*]}"
echo " ALG: $ALG"
echo "=============================="

if [ "$ALG" = "Pure_Dagger" ]; then
  # Pure DAgger: beta는 0.95 한 번만, alpha sweep 없이 고정 값만 사용
  BETA=0.95
  BETA_ARG="--beta $BETA"
  FIXED_ALPHA=${ALPHAS[0]}

  for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "------------------------------"
    echo " SEED               : $SEED"
    echo " ALG                : $ALG (Pure)"
    echo " ALPHA (fixed)      : $FIXED_ALPHA"
    echo " K_ITER             : $K_ITER"
    echo " STEPS_PER_ITER     : $STEPS_PER_ITER"
    echo " INIT_STEPS         : $INIT_STEPS"
    echo " EVAL_ITERS         : $EVAL_ITERS"
    echo " LAMBDA_THRESHOLD   : $LAMBDA"
    echo " MIN_DEMO_TIME      : $MIN_DEMO"
    echo " HISTORIC_CONTEXT   : $HIST"
    echo " EPOCH              : $EPOCH"
    echo " BETA               : $BETA"
    echo "------------------------------"

    python run_RND.py --kit_args "--alg $ALG \
--alpha $FIXED_ALPHA \
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
else
  # Non-Pure DAgger: beta 미사용, alpha sweep 수행
  BETA_ARG=""

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
      echo " BETA               : None"
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
fi
