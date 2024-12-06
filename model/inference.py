import json
from operator import itemgetter

import torch
import numpy as np
from pytorch_lightning import Trainer
from tqdm import tqdm
import pandas as pd
import resource
from argparse import ArgumentParser

from SE_XLNet import SEXLNet
from data import ClassificationData

from constituency_parse import ParseTree
from transformers import AutoTokenizer


def load_model(ckpt, batch_size):
    model = SEXLNet.load_from_checkpoint(ckpt)
    model.eval()
    trainer = Trainer(gpus=1)
    dm = ClassificationData(basedir=model.hparams.dataset_basedir, tokenizer_name=model.hparams.model_name,
                            batch_size=batch_size)
    return model, trainer, dm


def load_dev_examples(file_name):
    dev_samples = []
    with open(file_name, 'r') as open_file:
        for line in open_file:
            dev_samples.append(json.loads(line))
    return dev_samples


def eval(model, dataloader, concept_map, text):
    parser = ParseTree(tokenizer_name="xlnet-base-cased")

    parse_tree, nt_idx_matrix = parser.get_parse_tree_for_raw_sent(raw_sent=text)

    datapoint_dict = [{'sentence': text,'parse_tree': parse_tree,
                                            'label': 0,
                                            'nt_idx_matrix': nt_idx_matrix}]

    total_evaluated = 0.
    total_correct = 0.
    i = 0
    predicted_labels, true_labels, gil_overall, lil_overall = [], [], [], []
    accs = []
    with torch.no_grad():
        row = datapoint_dict[0]
        labels = torch.tensor(int(row["label"]))
        sent = row["sentence"]
        nt_idx_matrix = torch.tensor([row["nt_idx_matrix"]]).long()
        tokenizer = AutoTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)
        encoded_input = tokenizer(sent)
        input_tokens = torch.tensor([encoded_input["input_ids"]])
        if "token_type_ids" in encoded_input:
            token_type_ids = torch.tensor([encoded_input["token_type_ids"]]
        else:
            token_type_ids = torch.tensor([[0] * len(s) for s in encoded_input["input_ids"]])
        
        batch = (input_tokens, token_type_ids, nt_idx_matrix, labels)
        # input_tokens, token_type_ids, nt_idx_matrix, labels = batch
        print(batch)
        logits, acc, interpret_dict_list = model(batch)
        gil_interpretations = gil_interpret(concept_map=concept_map,
                                            list_of_interpret_dict=interpret_dict_list)
        lil_interpretations = lil_interpret(logits=logits,
                                            list_of_interpret_dict=interpret_dict_list,
                                            dev_samples=datapoint_dict,
                                            current_idx=i)
        batch_predicted_labels = torch.argmax(logits, -1)
        predicted_labels.extend(batch_predicted_labels.tolist())

        true_labels.extend(labels.tolist())
        gil_overall.extend(gil_interpretations)
        lil_overall.extend(lil_interpretations)

    df =  pd.DataFrame({"predicted_labels": predicted_labels, "true_labels": true_labels, "lil_interpretations": lil_overall,
                  "gil_interpretations": gil_overall})
    print(df)


def gil_interpret(concept_map, list_of_interpret_dict):
    batch_concepts = []
    for topk_concepts in list_of_interpret_dict["topk_indices"]:
        concepts = [concept_map[x] for x in topk_concepts.tolist()][:10]
        batch_concepts.append(concepts)
    return batch_concepts


def lil_interpret(logits, list_of_interpret_dict, dev_samples, current_idx):
    sf_logits = torch.softmax(logits, dim=1).tolist()
    print(torch.softmax(logits, dim=1).shape)
    
    lil_sf_logits = torch.softmax(list_of_interpret_dict["lil_logits"], dim=-1).tolist()

    lil_outputs = []
    for idx, (sf_item, lil_sf_item) in enumerate(zip(sf_logits, lil_sf_logits)):
        dev_sample = dev_samples[current_idx + idx]
        for phrase_idx, phrase in enumerate(dev_sample["parse_tree"]):
            print(phrase_idx, )
        print(dev_sample)
        lil_dict = {}
        argmax_sf, _ = max(enumerate(sf_item), key=itemgetter(1))
        for phrase_idx, phrase in enumerate(dev_sample["parse_tree"]):
            print(lil_sf_logits[idx])
            phrase_logits = lil_sf_logits[idx][phrase_idx]
            relevance_score = phrase_logits[argmax_sf] - sf_item[argmax_sf]
            if phrase_idx != 0:
                lil_dict[phrase["phrase"]] = relevance_score
        lil_dict = sorted(lil_dict.items(), key=lambda item: item[1], reverse=True)[:5]
        lil_outputs.append(lil_dict)
    return lil_outputs


def load_concept_map(concept_map_path):
    concept_map = {}
    with open(concept_map_path, 'r') as open_file:
        concept_map_str = json.loads(open_file.read())
    for key, value in concept_map_str.items():
        concept_map[int(key)] = value
    return concept_map


if __name__ == "__main__":
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

    parser = ArgumentParser()
    ckpt = "lightning_logs/version_33/checkpoints/epoch=4-step=19290-val_acc_epoch=0.9289.ckpt"
    concept_map ="data/XLNet-SST-2/concept_idx.json"
    dev_file ="How flower is today?"
    batch_size =1
    
    model, trainer, dm = load_model(ckpt,
                                    batch_size=batch_size)
    concept_map = load_concept_map(concept_map)
    eval(model,
         dm.val_dataloader(),
         concept_map=concept_map, text=dev_file)
