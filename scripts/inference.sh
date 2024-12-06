 python model/infer_model.py \
      --ckpt lightning_logs/version_34/checkpoints/epoch=4-step=1476-val_acc_epoch=0.9682.ckpt \
      --concept_map data/xlnet-trec/concept_idx.json \
      --paths_output_loc result/result_xlnettr_ft.tsv \
      --dev_file  data/xlnet-trec/dev_with_parse.json \
      --batch_size 16