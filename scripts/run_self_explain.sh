#!/bin/bash
export TOKENIZERS_PARALLELISM=false
python model/run.py --dataset_basedir data/roberta-sst2 \
                         --lr 2e-5  --max_epochs 5 \
                         --gpus 1 \
                         --model_name kssteven/ibert-roberta-base\
                         --concept_store data/roberta-sst2/concept_store.pt \
                         --accelerator cuda \
                         --gamma 0.1 \
                         --lamda 0.1 \
                         --topk 2

# for RoBERTa
# python model/run.py --dataset_basedir data/RoBERTa-SUBJ \
#                          --lr 2e-5  --max_epochs 5 \
#                          --gpus 1 \
#                          --concept_store data/RoBERTa-SUBJ/concept_store.pt \
#                          --accelerator ddp \
#                          --model_name roberta-base \
#                          --topk 5 \
#                          --gamma 0.1 \
#                          --lamda 0.1
