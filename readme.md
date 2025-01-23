# Auto Expanding Codebook for Vector Quantization
This is the repository for "Auto Expanding Codebook for Vector Quantization".

# How to train
```
torchrun --nnodes=1 --nproc_per_node=4 ./scripts/train_autoencoder.py --config-file=configs/auto_expanding_codebook.yaml \
--output_dir=./output/auto_expanding_codebook opts train.num_train_steps=100 train_batch_size=16
```


# How to evaluate
```
sh evaluation/run_tokenizer_eval.sh
```


