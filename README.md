# Aligned Transformer

|       Model        | IWSLT14 De-En | IWSLT14 En-De |
| :----------------: | :-----------: | :-----------: |
|    Transformer     |   **34.66**   |     28.33     |
|   Ours. No align   |     34.64     |     28.63     |
|   Ours. Aligned    |     34.58     |   **28.91**   |
| Ours. Aligned + lm |     34.33     |     28.64     |

## Installation

- PyTorch version >= 1.5.0
- Python version >= 3.6
- To install fairseq and develop locally:

```shell
git clone https://github.com/zml24/fairseq
cd fairseq
pip install --editable ./

# on MacOS:
# CFLAGS="-stdlib=libc++" pip install --editable ./
```
- For faster training install NVIDIA's apex library:
```shell
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir \
    --global-option="--cpp_ext" --global-option="--cuda_ext" \
    --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
    --global-option="--fast_multihead_attn" ./
```

## IWSLT14 De-En

To get the binary dataset, follow [fairseq's example](https://github.com/pytorch/fairseq/tree/master/examples/translation)

```shell
cd examples/translation/
bash prepare-iwslt14.sh
cd ../..

TEXT=examples/translation/iwslt14.tokenized.de-en
fairseq-preprocess \
    --source-lang de --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/iwslt14.tokenized.de-en \
    --workers 20
```

To reproduce a forward Transformer baseline (assume running on a P100 GPU)
```shell
export DATA=iwslt14.tokenized.de-en
export CKT_DIR=checkpoints/forward

fairseq-train \
    data-bin/$DATA \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 5.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 3584 --save-dir $CKT_DIR \
    --keep-last-epochs 10 --patience 10 --log-interval 1

fairseq-generate \
    data-bin/$DATA \
    --path $CKT_DIR/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > $CKT_DIR/gen.out

bash scripts/compound_split_bleu.sh $CKT_DIR/gen.out
```

To reproduce a backward Transformer baseline
```shell
export DATA=iwslt14.tokenized.de-en
export CKT_DIR=checkpoints/backward

fairseq-train \
    data-bin/$DATA -s en -t de \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 5.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 3584 --save-dir $CKT_DIR \
    --keep-last-epochs 10 --patience 10 --log-interval 1

fairseq-generate \
    data-bin/$DATA -s en -t de \
    --path $CKT_DIR/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > $CKT_DIR/gen.out

bash scripts/compound_split_bleu.sh $CKT_DIR/gen.out
```

To reproduce a bidirectional Transformer
```shell
export DATA=iwslt14.tokenized.de-en
export CKT_DIR=checkpoints/bidirection

fairseq-train --task aligned_translation \
    data-bin/$DATA \
    --arch aligned_transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 5.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion bidirectional_label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 3584 --save-dir $CKT_DIR \
    --keep-last-epochs 10 --patience 10 --log-interval 1

fairseq-generate --task aligned_translation --direction s2t \
    data-bin/$DATA \
    --path $CKT_DIR/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > $CKT_DIR/gen_s2t.out

fairseq-generate --task aligned_translation --direction t2s \
    data-bin/$DATA \
    --path $CKT_DIR/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > $CKT_DIR/gen_t2s.out

bash scripts/compound_split_bleu.sh $CKT_DIR/gen_s2t.out
bash scripts/compound_split_bleu.sh $CKT_DIR/gen_t2s.out
```

To reproduce a bidirectional Transformer with alignment
```shell
export DATA=iwslt14.tokenized.de-en
export CKT_DIR=checkpoints/alignment

fairseq-train --task aligned_translation \
    data-bin/$DATA \
    --arch aligned_transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 5.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion bidirectional_label_smoothed_cross_entropy_with_mse --label-smoothing 0.1 \
    --max-tokens 3584 --save-dir $CKT_DIR \
    --keep-last-epochs 10 --patience 10 --log-interval 1

fairseq-generate --task aligned_translation --direction s2t \
    data-bin/$DATA \
    --path $CKT_DIR/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > $CKT_DIR/gen_s2t.out

fairseq-generate --task aligned_translation --direction t2s \
    data-bin/$DATA \
    --path $CKT_DIR/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > $CKT_DIR/gen_t2s.out

bash scripts/compound_split_bleu.sh $CKT_DIR/gen_s2t.out
bash scripts/compound_split_bleu.sh $CKT_DIR/gen_t2s.out
```

To reproduce a bidirectional Transformer with alignment and better decoder
```shell
export DATA=iwslt14.tokenized.de-en
export CKT_DIR=checkpoints/alignment
export LM_CKT_DIR=checkpoints/alignment_lm
export MSE_CKT_DIR=checkpoints/alignment_mse

fairseq-train --task aligned_translation \
    data-bin/$DATA \
    --arch aligned_transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 5.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion bidirectional_label_smoothed_cross_entropy_with_mse --label-smoothing 0.1 \
    --max-tokens 3584 --save-dir $CKT_DIR \
    --keep-last-epochs 10 --patience 10 --log-interval 1

mkdir $LM_CKT_DIR
cp -r $CKT_DIR/checkpoint_last.pt $LM_CKT_DIR

fairseq-train --task aligned_translation \
    data-bin/$DATA --freeze-encoder \
    --arch aligned_transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 5.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion bidirectional_label_smoothed_cross_entropy_lm --label-smoothing 0.1 \
    --max-tokens 3584 --save-dir $LM_CKT_DIR \
    --reset-dataloader --reset-lr-scheduler --reset-meters --reset-optimizer \
    --keep-last-epochs 10 --patience 10 --max-epoch 100 --log-interval 1

mkdir $MSE_CKT_DIR
cp -r $LM_CKT_DIR/checkpoint_last.pt $MSE_CKT_DIR

fairseq-train --task aligned_translation \
    data-bin/$DATA \
    --arch aligned_transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 5.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion bidirectional_label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 3584 --save-dir $MSE_CKT_DIR \
    --reset-dataloader --reset-lr-scheduler --reset-meters --reset-optimizer \
    --keep-last-epochs 10 --patience 10 --log-interval 1

fairseq-generate --task aligned_translation --direction s2t \
    data-bin/$DATA \
    --path $MSE_CKT_DIR/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > $MSE_CKT_DIR/gen_s2t.out

fairseq-generate --task aligned_translation --direction t2s \
    data-bin/$DATA \
    --path $MSE_CKT_DIR/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > $MSE_CKT_DIR/gen_t2s.out

bash scripts/compound_split_bleu.sh $MSE_CKT_DIR/gen_s2t.out
bash scripts/compound_split_bleu.sh $MSE_CKT_DIR/gen_t2s.out
```
