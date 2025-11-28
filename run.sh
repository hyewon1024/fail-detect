#!/bin/bash
# chmod +x run.sh

K_ITER=200
STEPS_PER_ITER=2000
INIT_STEPS=2000
EVAL_ITERS=5          #change steps to iters
LAMBDA=0.01
MIN_DEMO=70
HIST=0
EPOCH=10

# =========================================
# ✏️ 여기만 수정해서 실험 세팅하세요
# =========================================

ALG="Balanced_Dagger"       # Pure_Dagger, R_Dagger, Balanced_Dagger, Safe_Dagger
ALPHA=0.5
SEED=42
BETA=""        # Pure_Dagger 등에서만 사용 ("" or float)

# =========================================
# 실행
# =========================================
if [ -z "$BETA" ]; then
  BETA_ARG=""
else
  BETA_ARG="--beta $BETA"
fi

echo "=============================="
echo " Running RND-DAgger"
echo "=============================="
echo "ALG                : $ALG"
echo "ALPHA              : $ALPHA"
echo "SEED               : $SEED"
echo "K_ITER             : $K_ITER"
echo "STEPS_PER_ITER     : $STEPS_PER_ITER"
echo "INIT_STEPS         : $INIT_STEPS"
echo "EVAL_STEPS         : $EVAL_STEPS"
echo "LAMBDA_THRESHOLD   : $LAMBDA"
echo "MIN_DEMO_TIME      : $MIN_DEMO"
echo "HISTORIC_CONTEXT   : $HIST"
echo "EPOCH              : $EPOCH"
echo "BETA               : ${BETA:-None}"
echo "=============================="

python run_RND.py --kit_args "--alg $ALG \
--alpha $ALPHA \
$BETA_ARG \
--seed $SEED \
--k_iter $K_ITER \
--steps_per_iter $STEPS_PER_ITER \
--init_steps $INIT_STEPS \
--eval_steps $EVAL_STEPS \
--lambda_th $LAMBDA \
--min_demo $MIN_DEMO \
--hist $HIST \
--epoch $EPOCH"