
####################################
# resume 전용 설정(if RESUME)
####################################
CKPT_FOLDER="/fail-detect/checkpoints/[Safe_Dagger]rnd_iter200_balance_False_epoch_10_alpha_0.5_minDemo70_H0_FTrue_seed42"
ITER_NUM=5  # 끊어진 iteration 번호
####################################
# 실행
####################################
echo "=============================="
echo " Folder = $CKPT_FOLDER "
echo " Iter_number = $ITER_NUM "

python run_RND_f.py --kit_args\
        "--folder $CKPT_FOLDER \
        --Iter_NUM $ITER_NUM"
