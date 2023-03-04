import json
import argparse
import os
import sys
import random
import logging
import time
from tqdm import tqdm
import numpy as np

from shared.const import ner_labels, role_labels, get_labelmap
from role.utils import convert_dataset_to_samples_sent_level, batchify
from role.models import RoleModel

from transformers import AdamW, get_linear_schedule_with_warmup
import torch

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger('root')

def setseed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_model(model, model_save_path):
    """
    Save the model to the output directory
    """
    logger.info('Saving model to %s...'%(model_save_path))
    model_to_save = model.bert_model.module if hasattr(model.bert_model, 'module') else model.bert_model
    model_to_save.save_pretrained(model_save_path)
    model.tokenizer.save_pretrained(model_save_path)
    
def output_role_predictions(model, batches, output_file):
    """
    Save the prediction as a json file
    """
    for i in range(len(batches)):
        output_dict = model.run_batch(batches[i], use_gold=False, training=False)
        pred_role = output_dict['pred_role']
        for sample, preds in zip(batches[i], pred_role):
            spans = []
            spans_ner_label = []
            spans_role_label = []
            pred_spans = []
            pred_spans_ner_label = []
            sample['pred_spans_role_label'] = []
            
            for span, span_ner, span_role in zip(sample['spans'], sample['spans_ner_label'], sample['spans_role_label']):
                spans.append([span[0], span[1]])
                spans_ner_label.append(ner_id2label[span_ner])
                spans_role_label.append(role_id2label[span_role])
            
            for p_role, pred_span_info, pred_span, pred_ner in zip(preds, sample['pred_spans_info'], sample['pred_spans'], sample['pred_spans_ner_label']):
                sample['pred_spans_role_label'].append(role_id2label[p_role])
                pred_span_info[3] = role_id2label[p_role]
                pred_spans.append([pred_span[0], pred_span[1]])
                pred_spans_ner_label.append(ner_id2label[pred_ner])
            
            sample['spans'] = spans
            sample['spans_ner_label'] = spans_ner_label
            sample['spans_role_label'] = spans_role_label
            sample['pred_spans'] = pred_spans
            sample['pred_spans_ner_label'] = pred_spans_ner_label

    logger.info('Output predictions to %s..'%(output_file))
    with open(output_file, 'w') as f:
        f.write('\n'.join(json.dumps(sample) for batch in batches for sample in batch))

def evaluate(model, batches, use_gold=True):
    logger.info('Evaluating...')
    c_time = time.time()
    
    l_tot = 0    # totoal predict labels, include 0
    tot_gold = 0
    tot_pred = 0
    l_tot_role = dict.fromkeys(range(6), 0)
    tot_gold_role = dict.fromkeys(range(6), 0)
    tot_pred_role = dict.fromkeys(range(6), 0)
    
    l_cor_loosely = 0
    l_cor_strict = 0
    cor_loosely = 0
    cor_strict = 0
    l_cor_role_loosely = dict.fromkeys(range(6), 0)
    l_cor_role_strict = dict.fromkeys(range(6), 0)
    cor_role_loosely = dict.fromkeys(range(6), 0)
    cor_role_strict = dict.fromkeys(range(6), 0)

    for i in range(len(batches)):
        output_dict = model.run_batch(batches[i], use_gold=use_gold, training=False)    # when eval on dev set, wheather to use gold.
        pred_role = output_dict['pred_role']
        if use_gold:    # when using gold, span must be correct
            for sample, preds in zip(batches[i], pred_role):
                for gold_ner, gold_role, pred_ner, p_role in zip(sample['spans_ner_label'], sample['spans_role_label'], sample['pred_spans_ner_label'], preds):
                    l_tot += 1
                    if p_role == gold_role:
                        l_cor_loosely += 1
                        l_cor_role_loosely[p_role] += 1
                    if gold_role != 0:
                        tot_gold += 1
                        tot_gold_role[gold_role] += 1
                    if p_role != 0 and gold_role != 0 and p_role == gold_role:
                        cor_loosely += 1
                        cor_role_loosely[p_role] += 1
                    if p_role != 0:
                        tot_pred += 1
                        tot_pred_role[p_role] += 1

                    if pred_ner == gold_ner:    # ner is also correct
                        if p_role == gold_role:
                            l_cor_strict += 1
                            l_cor_role_strict[p_role] += 1
                        if p_role != 0 and gold_role != 0 and p_role == gold_role:
                            cor_strict += 1
                            cor_role_strict[p_role] += 1
        else:
            for sample, preds in zip(batches[i], pred_role):
                for gold_role in sample['spans_role_label']:
                    if gold_role != 0:
                        tot_gold += 1
                        tot_gold_role[gold_role] += 1
                
                for i, pred_span in enumerate(sample['pred_spans']):
                    l_tot += 1
                    if pred_span in sample['spans']:    # pred span correct
                        j = sample['spans'].index(pred_span)
                        gold_role = sample['spans_role_label'][j]
                        gold_ner = sample['spans_ner_label'][j]
                        p_role = preds[i]
                        pred_ner = sample['pred_spans_ner_label'][i]
                        l_tot_role[p_role] += 1
                        if p_role == gold_role:
                            l_cor_loosely += 1
                            l_cor_role_loosely[p_role] += 1
                        if p_role != 0 and gold_role != 0 and p_role == gold_role:
                            cor_loosely += 1
                            cor_role_loosely[p_role] += 1
                        if p_role != 0:
                            tot_pred += 1
                            tot_pred_role[p_role] += 1

                        if pred_ner == gold_ner:    # ner is also correct
                            if p_role == gold_role:
                                l_cor_strict += 1
                                l_cor_role_strict[p_role] += 1
                            if p_role != 0 and gold_role != 0 and p_role == gold_role:
                                cor_strict += 1
                                cor_role_strict[p_role] += 1
                            
    acc_loosely = l_cor_loosely / l_tot
    acc_strict = l_cor_strict / l_tot
    logger.info('Accuracy: loosely %5f, strict %5f'%(acc_loosely, acc_strict))
    logger.info('loosely Cor: %d, Pred TOT: %d, Gold TOT: %d'%(cor_loosely, tot_pred, tot_gold))
    logger.info('strict Cor: %d, Pred TOT: %d, Gold TOT: %d'%(cor_strict, tot_pred, tot_gold))
    
    p_loosely = cor_loosely / tot_pred if cor_loosely > 0 else 0.0
    r_loosely = cor_loosely / tot_gold if cor_loosely > 0 else 0.0
    f1_loosely = 2 * (p_loosely * r_loosely) / (p_loosely + r_loosely) if cor_loosely > 0 else 0.0
    
    p_strict = cor_strict / tot_pred if cor_strict > 0 else 0.0
    r_strict = cor_strict / tot_gold if cor_strict > 0 else 0.0
    f1_strict = 2 * (p_strict * r_strict) / (p_strict + r_strict) if cor_strict > 0 else 0.0
    logger.info('loosely P: %.5f, R: %.5f, F1: %.5f'%(p_loosely, r_loosely, f1_loosely))
    logger.info('strict P: %.5f, R: %.5f, F1: %.5f'%(p_strict, r_strict, f1_strict))
    
    for idx in range(1, len(role_labels)):
        p_loosely = cor_role_loosely[idx] / tot_pred_role[idx] if cor_role_loosely[idx] > 0 else 0.0
        r_loosely = cor_role_loosely[idx] / tot_gold_role[idx] if cor_role_loosely[idx] > 0 else 0.0
        f1_loosely = 2 * (p_loosely * r_loosely) / (p_loosely + r_loosely) if cor_role_loosely[idx] > 0 else 0.0
        logger.info('loosely %s P: %.5f, R: %.5f, F1: %.5f, Cor: %d, Pred TOT: %d, Gold TOT: %d'%(role_labels[idx], p_loosely, r_loosely, f1_loosely, cor_role_loosely[idx], tot_pred_role[idx], tot_gold_role[idx]))
        
        p_strict = cor_role_strict[idx] / tot_pred_role[idx] if cor_role_strict[idx] > 0 else 0.0
        r_strict = cor_role_strict[idx] / tot_gold_role[idx] if cor_role_strict[idx] > 0 else 0.0
        f1_strict = 2 * (p_strict * r_strict) / (p_strict + r_strict) if cor_role_strict[idx] > 0 else 0.0
        logger.info('strict %s P: %.5f, R: %.5f, F1: %.5f, Cor: %d, Pred TOT: %d, Gold TOT: %d'%(role_labels[idx], p_strict, r_strict, f1_strict, cor_role_strict[idx], tot_pred_role[idx], tot_gold_role[idx]))
    logger.info('Used time: %f'%(time.time()-c_time))
    return f1_strict


if __name__ == '__main__':
    data_dir = '/your/dataset/'
    output_dir = '/your/output_dir/'
    model_path = '/your/model/path/'
    model_save_path = '/your/model_save_path/'
    test_pred_filename = 'test_predict_role.json'
    dev_pred_filename = 'dev_predict_role.json'
    
    eval_batch_size = 16
    train_batch_size = 16
    task_learning_rate = 5e-4
    learning_rate = 1e-5
    num_epoch = 5
    warmup_proportion = 0.1
    eval_per_epoch = 2
    train_shuffle = True
    print_loss_step = 300
    max_span_length = 12
    use_gold = True
    do_train = True
    do_eval = True
    eval_test = True
    
    train_data = os.path.join(data_dir, 'train.json')
    dev_data = os.path.join(data_dir, 'dev.json')
    test_data = '/your/path/to/test_predict_ner.json'

    setseed(37)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if do_train:
        logger.addHandler(logging.FileHandler(os.path.join(output_dir, "train.log"), 'w'))
    else:
        logger.addHandler(logging.FileHandler(os.path.join(output_dir, "eval.log"), 'w'))
    
    ner_label2id, ner_id2label = get_labelmap(ner_labels)
    role_label2id, role_id2label = get_labelmap(role_labels)
    
    num_role_labels = len(role_labels)
    model = RoleModel(model_path, num_role_labels=num_role_labels)

    dev_samples = convert_dataset_to_samples_sent_level(dev_data, use_gold=use_gold, ner_label2id=ner_label2id,  role_label2id=role_label2id)
    dev_batches = batchify(dev_samples, eval_batch_size)

    if do_train:
        train_samples = convert_dataset_to_samples_sent_level(train_data,  use_gold=use_gold, ner_label2id=ner_label2id,  role_label2id=role_label2id)
        train_batches = batchify(train_samples, train_batch_size)
        best_result = 0.0

        param_optimizer = list(model.bert_model.named_parameters())
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer
                if 'bert' in n]},
            {'params': [p for n, p in param_optimizer
                if 'bert' not in n], 'lr': task_learning_rate}]
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
        t_total = len(train_batches) * num_epoch
        scheduler = get_linear_schedule_with_warmup(optimizer, int(t_total*warmup_proportion), t_total)
        
        tr_loss = 0
        tr_examples = 0
        global_step = 0
        eval_step = len(train_batches) // eval_per_epoch
        for _ in tqdm(range(num_epoch)):
            if train_shuffle:
                random.shuffle(train_batches)
            for i in tqdm(range(len(train_batches))):
                output_dict = model.run_batch(train_batches[i], training=True)    # when training, use_gold is True
                loss = output_dict['role_loss']
                loss.backward()

                tr_loss += loss.item()
                tr_examples += len(train_batches[i])
                global_step += 1

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if global_step % print_loss_step == 0:
                    logger.info('Epoch=%d, iter=%d, loss=%.5f'%(_, i, tr_loss / tr_examples))
                    tr_loss = 0
                    tr_examples = 0

                if global_step % eval_step == 0:
                    f1 = evaluate(model, dev_batches, use_gold=use_gold)
                    if f1 > best_result:
                        best_result = f1
                        logger.info('!!! Best valid (epoch=%d): %.2f' % (_, f1*100))
                        save_model(model, model_save_path)

    if do_eval:
        model = RoleModel(model_save_path, num_role_labels=num_role_labels)
        if eval_test:
            test_data = test_data
            prediction_file = os.path.join(output_dir, test_pred_filename)
            use_gold = False    # when prediting, use_gold is False
        else:
            test_data = dev_data
            prediction_file = os.path.join(output_dir, dev_pred_filename)
        test_samples = convert_dataset_to_samples_sent_level(test_data,  use_gold=use_gold, ner_label2id=ner_label2id, role_label2id=role_label2id)
        test_batches = batchify(test_samples, eval_batch_size)
        evaluate(model, test_batches, use_gold=use_gold)
        output_role_predictions(model, test_batches, output_file=prediction_file)
