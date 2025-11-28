#!/bin/bash
# chmod +x backup.sh
####################################
# 1. 실행 모드 선택
####################################
MODE="resume"          # train or resume
K_ITER=200
STEPS_PER_ITER=2000
INIT_STEPS=2000
EVAL_ITERS=100
LAMBDA=0.01
MIN_DEMO=70
HIST=0
EPOCH=10
# =========================================
# :연필2: 여기만 수정해서 실험 세팅하세요
# =========================================
ALG="Balanced_Dagger"       # Pure_Dagger, R_Dagger, Balanced_Dagger, Safe_Dagger
ALPHA=0.5
SEED=42
BETA=""        # Pure_Dagger 등에서만 사용 ("" or float)
####################################
# resume 전용 설정(if RESUME)
####################################
CKPT_FOLDER="/AILAB-summer-school-2025/RND/checkpoints/[Balanced_Dagger]rnd_iter200_balance_True_epoch_10_alpha_0.5_minDemo70_H0_FTrue_seed2"
ITER_NUM=60  # 끊어진 iteration 번호
####################################
# 실행
####################################
echo "=============================="
echo " MODE = $MODE "
echo "=============================="
if [ "$MODE" = "train" ]; then
    echo ":로켓: Starting training"
    python run_RND.py --kit_args "--alg $ALG \
    --alpha $ALPHA \
    --seed $SEED \
    --k_iter $K_ITER \
    --steps_per_iter $STEPS_PER_ITER \
    --init_steps $INIT_STEPS \
    --eval_iters $EVAL_ITERS \
    --lambda_th $LAMBDA \
    --min_demo $MIN_DEMO \
    --hist $HIST \
    --epoch $EPOCH \
    $BETA_ARG"
elif [ "$MODE" = "resume" ]; then
    echo ""
    echo ":재활용: Resuming from checkpoint"
    echo "--------------------------------"
    echo " Folder : $CKPT_FOLDER"
    echo " Iter   : $ITER_NUM"
    echo "--------------------------------"
    if [ ! -d "$CKPT_FOLDER" ]; then
        echo ":x: Checkpoint folder not found: $CKPT_FOLDER"
        exit 1
    fi
    KIT_ARGS="--ckpt_folder $CKPT_FOLDER --iter_num $ITER_NUM" \
    python run_RND_f.py
else
    echo ":x: MODE must be 'train' or 'resume'"
    exit 1
fi
