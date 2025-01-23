resolution=128
SAMPLES_FOR_TEST=50000
BATCH_SIZE=16

spatial_scale=1

configs_of_training_lists=(./output/auto_expanding_codebook/)

for cfg_dir in "${configs_of_training_lists[@]}"
do
    configs_of_training=$cfg_dir"config.yaml"
    checkpoint_dir=$cfg_dir"models/"
    
    echo "Start testing on $configs_of_training"
    checkpoint_arr=()
    for (( i=100000; i<=100000; i+=1 ))
    do
        checkpoint="eval_${i}.pt"
        # checkpoint="checkpoint_${i}.pt"
        if [ ! -f "${checkpoint_dir}${checkpoint}" ]; then
            break
        fi
        checkpoint_arr+=("$checkpoint")
    done

    for checkpoint in "${checkpoint_arr[@]}"
    do
        echo "-----------------------"
        echo "Testing on $checkpoint"
        NUMBER_OF_GPUS=4
        torchrun --nnodes=1 --nproc_per_node=$NUMBER_OF_GPUS ./evaluation/eval_tokenizer.py --config_file=$configs_of_training \
        --checkpoint_path=$checkpoint_dir$checkpoint --num_samples=$(($SAMPLES_FOR_TEST/$NUMBER_OF_GPUS)) --batch_size=$BATCH_SIZE \
        --resolution=$resolution --spatial_scale=$spatial_scale \
        opts evaluation.metrics=[mse,fid,is,lpips,psnr,ssim]
        # opts evaluation.metrics=[mse,fid,is,sfid,fdd,lpips,psnr,ssim]
    done

    echo "=====> Done"
    echo "@@@@@@@@@@@@@@@@@@@"
done

