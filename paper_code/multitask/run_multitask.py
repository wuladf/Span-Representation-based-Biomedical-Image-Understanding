import json
import argparse
import os
import sys
import random
import logging
import time
from tqdm import tqdm
import numpy as np

from const import ner_labels, role_labels, get_labelmap
from utils import convert_dataset_to_samples_sent_level,batchify
from models import MultiTaskModel

from transformers import AdamW, get_linear_schedule_with_warmup
import torch

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger('root')

def save_model(model, model_save_path):
    """
    Save the model to the output directory
    """
    logger.info('Saving model to %s...'%(model_save_path))
    model_to_save = model.bert_model.module if hasattr(model.bert_model, 'module') else model.bert_model
    model_to_save.save_pretrained(model_save_path)
    model.tokenizer.save_pretrained(model_save_path)

def output_predictions(model, batches, output_file):
    """
    Save the prediction as a json file
    """
    for i in range(len(batches)):
        output_dict = model.run_batch(batches[i], training=False)
        pred_ner = output_dict['pred_ner']
        pred_role = output_dict['pred_role']
        for sample, ner_preds, role_preds in zip(batches[i], pred_ner, pred_role):
            sample['pred_spans_info'] = []
            sample['pred_spans'] = []
            sample['pred_spans_ner_label'] = []
            sample['pred_spans_role_label'] = []
            for span, ner_pred, role_pred in zip(sample['spans'], ner_preds, role_preds):
                if ner_pred == 0:
                    continue
                mention = sample['tokens'][span[0]:span[1]+1]
                mention = ' '.join(token for token in mention)
                sample['pred_spans_info'].append([span[0], span[1], ner_id2label[ner_pred], role_id2label[role_pred], 'normalized', mention])
                sample['pred_spans'].append([span[0], span[1]])
                sample['pred_spans_ner_label'].append(ner_id2label[ner_pred])
                sample['pred_spans_role_label'].append(role_id2label[role_pred])
            
            spans = []
            spans_ner_label = []
            spans_role_label = []
            for span_info in sample['spans_info']:
#                 if span_ner_label != 0:
#                     mention = sample['tokens'][span[0]:span[1]+1]
#                     mention = ' '.join(token for token in mention)
#                     spans.append(span)
                spans.append([span_info[0], span_info[1]])
                spans_ner_label.append(span_info[2])
                spans_role_label.append(span_info[3])
            sample['spans'] = spans
            sample['spans_ner_label'] = spans_ner_label
            sample['spans_role_label'] = spans_role_label
#             sample['spans_role_label'] = [role_id2label[span_role_label] for span_role_label in sample['spans_role_label']]
            
    logger.info('Output predictions to %s..'%(output_file))
    with open(output_file, 'w') as f:
        f.write('\n'.join(json.dumps(sample) for batch in batches for sample in batch))

def evaluate(model, batches):
    """
    Evaluate the entity model
    """
    logger.info('Evaluating...')
    c_time = time.time()
    ner_cor = 0
    ner_tot_gold = 0
    ner_tot_pred = 0
    ner_l_cor = 0
    l_tot = 0
    
    cor_type = dict.fromkeys(range(8), 0)
    tot_gold_type = dict.fromkeys(range(8), 0)
    tot_pred_type = dict.fromkeys(range(8), 0)
    l_cor_type = dict.fromkeys(range(8), 0)
    l_tot_type = dict.fromkeys(range(8), 0)
    
    role_tot_gold = 0
    role_tot_pred = 0
    l_tot_role = dict.fromkeys(range(7), 0)
    tot_gold_role = dict.fromkeys(range(7), 0)
    tot_pred_role = dict.fromkeys(range(7), 0)
    
    role_l_cor_loosely = 0
    role_l_cor_strict = 0
    role_cor_loosely = 0
    role_cor_strict = 0
    l_cor_role_loosely = dict.fromkeys(range(7), 0)
    l_cor_role_strict = dict.fromkeys(range(7), 0)
    cor_role_loosely = dict.fromkeys(range(7), 0)
    cor_role_strict = dict.fromkeys(range(7), 0)

    for i in range(len(batches)):
        output_dict = model.run_batch(batches[i], training=False)
        pred_ner = output_dict['pred_ner']
        pred_role = output_dict['pred_role']
        for sample, ner_preds, role_preds in zip(batches[i], pred_ner, pred_role):
            for ner_gold, ner_pred, role_gold, role_pred in zip(sample['spans_ner_label'], ner_preds, sample['spans_role_label'], role_preds):
                l_tot += 1
                if ner_gold != 0:
                    ner_tot_gold += 1
                    tot_gold_type[ner_gold] += 1
                if ner_pred == ner_gold:
                    ner_l_cor += 1
                    l_cor_type[ner_pred] += 1
                if ner_pred != 0 and ner_gold != 0 and ner_pred == ner_gold:
                    ner_cor += 1
                    cor_type[ner_pred] += 1
                if ner_pred != 0:
                    ner_tot_pred += 1
                    tot_pred_type[ner_pred] += 1
                    
                if role_pred == role_gold:
                    role_l_cor_loosely += 1
                    l_cor_role_loosely[role_pred] += 1
                if role_gold != 0:
                    role_tot_gold += 1
                    tot_gold_role[role_gold] += 1
                if role_pred != 0 and role_gold != 0 and role_pred == role_gold:
                    role_cor_loosely += 1
                    cor_role_loosely[role_pred] += 1
                if role_pred != 0:
                    role_tot_pred += 1
                    tot_pred_role[role_pred] += 1

                if ner_pred!=0 and ner_pred == ner_gold:
                    if role_pred == role_gold:
                        role_l_cor_strict += 1
                        l_cor_role_strict[role_pred] += 1
                    if role_pred != 0 and role_gold != 0 and role_pred == role_gold:
                        role_cor_strict += 1
                        cor_role_strict[role_pred] += 1
                   
    ner_acc = ner_l_cor / l_tot
    logger.info('Accuracy: %5f'%ner_acc)
    logger.info('Cor: %d, Pred TOT: %d, Gold TOT: %d'%(ner_cor, ner_tot_pred, ner_tot_gold))
    ner_p = ner_cor / ner_tot_pred if ner_cor > 0 else 0.0
    ner_r = ner_cor / ner_tot_gold if ner_cor > 0 else 0.0
    ner_f1 = 2 * (ner_p * ner_r) / (ner_p + ner_r) if ner_cor > 0 else 0.0
    logger.info('P: %.5f, R: %.5f, F1: %.5f'%(ner_p, ner_r, ner_f1))
    for idx in range(1, len(ner_labels)):
        ner_p = cor_type[idx] / tot_pred_type[idx] if cor_type[idx] > 0 else 0.0
        ner_r = cor_type[idx] / tot_gold_type[idx] if cor_type[idx] > 0 else 0.0
        ner_f1 = 2 * (ner_p * ner_r) / (ner_p + ner_r) if cor_type[idx] > 0 else 0.0
        logger.info('%s P: %.5f, R: %.5f, F1: %.5f, Cor: %d, Pred TOT: %d, Gold TOT: %d'%(ner_labels[idx], ner_p, ner_r, ner_f1, cor_type[idx], tot_pred_type[idx], tot_gold_type[idx]))
        
    role_acc_loosely = role_l_cor_loosely / l_tot
    role_acc_strict = role_l_cor_strict / l_tot
    logger.info('Accuracy: loosely %5f, strict %5f'%(role_acc_loosely, role_acc_strict))
    logger.info('loosely Cor: %d, Pred TOT: %d, Gold TOT: %d'%(role_cor_loosely, role_tot_pred, role_tot_gold))
    logger.info('strict Cor: %d, Pred TOT: %d, Gold TOT: %d'%(role_cor_strict, role_tot_pred, role_tot_gold))
    
    role_p_loosely = role_cor_loosely / role_tot_pred if role_cor_loosely > 0 else 0.0
    role_r_loosely = role_cor_loosely / role_tot_gold if role_cor_loosely > 0 else 0.0
    role_f1_loosely = 2 * (role_p_loosely * role_r_loosely) / (role_p_loosely + role_r_loosely) if role_cor_loosely > 0 else 0.0
    
    role_p_strict = role_cor_strict / role_tot_pred if role_cor_strict > 0 else 0.0
    role_r_strict = role_cor_strict / role_tot_gold if role_cor_strict > 0 else 0.0
    role_f1_strict = 2 * (role_p_strict * role_r_strict) / (role_p_strict + role_r_strict) if role_cor_strict > 0 else 0.0
    logger.info('loosely P: %.5f, R: %.5f, F1: %.5f'%(role_p_loosely, role_r_loosely, role_f1_loosely))
    logger.info('strict P: %.5f, R: %.5f, F1: %.5f'%(role_p_strict, role_r_strict, role_f1_strict))
    
    for idx in range(1, len(role_labels)):
        role_p_loosely = cor_role_loosely[idx] / tot_pred_role[idx] if cor_role_loosely[idx] > 0 else 0.0
        role_r_loosely = cor_role_loosely[idx] / tot_gold_role[idx] if cor_role_loosely[idx] > 0 else 0.0
        role_f1_loosely = 2 * (role_p_loosely * role_r_loosely) / (role_p_loosely + role_r_loosely) if cor_role_loosely[idx] > 0 else 0.0
        logger.info('loosely %s P: %.5f, R: %.5f, F1: %.5f, Cor: %d, Pred TOT: %d, Gold TOT: %d'%(role_labels[idx], role_p_loosely, role_r_loosely, role_f1_loosely, cor_role_loosely[idx], tot_pred_role[idx], tot_gold_role[idx]))
        
        role_p_strict = cor_role_strict[idx] / tot_pred_role[idx] if cor_role_strict[idx] > 0 else 0.0
        role_r_strict = cor_role_strict[idx] / tot_gold_role[idx] if cor_role_strict[idx] > 0 else 0.0
        role_f1_strict = 2 * (role_p_strict * role_r_strict) / (role_p_strict + role_r_strict) if cor_role_strict[idx] > 0 else 0.0
        logger.info('strict %s P: %.5f, R: %.5f, F1: %.5f, Cor: %d, Pred TOT: %d, Gold TOT: %d'%(role_labels[idx], role_p_strict, role_r_strict, role_f1_strict, cor_role_strict[idx], tot_pred_role[idx], tot_gold_role[idx]))
    logger.info('Used time: %f'%(time.time()-c_time))
    return (ner_f1+role_f1_strict)/2

def setseed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    models_path = '/root/autodl-nas/'
    for model_name in os.listdir(models_path):
        used_list = {'.ipynb_checkpoints','scibert', 'biobert'}
        if model_name in used_list:
            continue
        model_path = os.path.join(models_path, model_name)
#         output_dir = '/root/predict/' + 'panel_' + model_name
#         model_save_path = '/root/model_saved/' + 'panel_' + model_name
#         data_dir = '/root/paper_dataset/'
    #     output_dir = '/root/predict/'
    #     model_path = '/root/autodl-nas/PubMedBERT/'
    #     model_path = '/root/autodl-nas/scibert/'
    #     model_path = '/root/autodl-nas/biobert/'
    #     model_save_path = '/root/model_saved/'
#         data_dir = '/root/test_paper_dataset_full/'
#         output_dir = '/root/test_predict_multitask/'
        data_dir = '/root/paper_dataset_full/'
        output_dir = '/root/multitask_predict/'
        model_save_path = '/root/multitask_PubMed_model_saved/'
#         test_pred_filename = 'test_predict.json'
#         dev_pred_filename = 'dev_predict.json'
#         test_pred_filename = 'test_predict_ner.json'
#         dev_pred_filename = 'dev_predict_ner.json'
        test_pred_filename = 'test_predict_multitask.json'
        dev_pred_filename = 'dev_predict_multitask.json'
        

#         data_dir = '/root/paper_test_dataset/'
    #     output_dir = '/root/test_predict/'
    #     model_path = '/root/autodl-nas/PubMedBERT/'
    #     model_save_path = '/root/test_model_saved/'
    #     test_pred_filename = 'test_predict.json'
    #     dev_pred_filename = 'dev_predict.json'

        eval_batch_size = 16
        train_batch_size = 16
        task_learning_rate = 5e-4
        learning_rate = 1e-5
        num_epoch = 5
        warmup_proportion = 0.1
        eval_per_epoch = 2
        train_shuffle = True
        print_loss_step = 300
#         print_loss_step = 30
        max_span_length = 8
        do_train = True
#         do_train = False
        do_eval = True
        eval_test = True

        train_data = os.path.join(data_dir, 'train.json')
        dev_data = os.path.join(data_dir, 'dev.json')
        test_data = os.path.join(data_dir, 'test.json')

        setseed(37)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        if do_train:
            logger.addHandler(logging.FileHandler(os.path.join(output_dir, "train.log"), 'w'))
        else:
            logger.addHandler(logging.FileHandler(os.path.join(output_dir, "eval.log"), 'w'))

        ner_label2id, ner_id2label = get_labelmap(ner_labels)
        role_label2id, role_id2label = get_labelmap(role_labels)

        num_ner_labels = len(ner_labels)
        num_role_labels = len(role_labels)
        model = MultiTaskModel(model_path, num_ner_labels=num_ner_labels, num_role_labels=num_role_labels)

        dev_samples = convert_dataset_to_samples_sent_level(dev_data, max_span_length, ner_label2id=ner_label2id,  role_label2id=role_label2id)
#         dev_samples = convert_dataset_to_samples_panel_level(dev_data, max_span_length, model_path, ner_label2id=ner_label2id, role_label2id=role_label2id)
        dev_batches = batchify(dev_samples, eval_batch_size)

        if do_train:
            train_samples = convert_dataset_to_samples_sent_level(train_data, max_span_length, ner_label2id=ner_label2id,  role_label2id=role_label2id)
#             train_samples = convert_dataset_to_samples_panel_level(train_data, max_span_length, model_path, ner_label2id=ner_label2id, role_label2id=role_label2id)
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
                    output_dict = model.run_batch(train_batches[i], training=True)
                    loss = output_dict['multi_loss']
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
                        f1 = evaluate(model, dev_batches)
                        if f1 > best_result:
                            best_result = f1
                            logger.info('!!! Best valid (epoch=%d): %.2f' % (_, f1*100))
                            save_model(model, model_save_path)
                            
#             test_samples = convert_dataset_to_samples_sent_level(test_data, max_span_length, ner_label2id=ner_label2id, role_label2id=role_label2id)
#             test_batches = batchify(test_samples, eval_batch_size)
#             evaluate(model, test_batches)
#             output_predictions(model, test_batches, output_file=os.path.join(output_dir, test_pred_filename))

        if do_eval:
            model = MultiTaskModel(model_save_path, num_ner_labels=num_ner_labels, num_role_labels=num_role_labels)
            if eval_test:
                test_data = test_data
                prediction_file = os.path.join(output_dir, test_pred_filename)
            else:
                test_data = dev_data
                prediction_file = os.path.join(output_dir, dev_pred_filename)
            test_samples = convert_dataset_to_samples_sent_level(test_data, max_span_length, ner_label2id=ner_label2id,  role_label2id=role_label2id)
#             test_samples = convert_dataset_to_samples_panel_level(test_data, max_span_length, model_path, ner_label2id=ner_label2id, role_label2id=role_label2id)
            test_batches = batchify(test_samples, eval_batch_size)
            evaluate(model, test_batches)
            output_predictions(model, test_batches, output_file=prediction_file)