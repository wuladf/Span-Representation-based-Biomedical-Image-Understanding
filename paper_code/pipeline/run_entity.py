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
from entity.utils_full import convert_dataset_to_samples_sent_level,batchify
from entity.models import EntityModel

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

def output_ner_predictions(model, batches, output_file):
    """
    Save the prediction as a json file
    """
    for i in range(len(batches)):
        output_dict = model.run_batch(batches[i], training=False)
        pred_ner = output_dict['pred_ner']
        for sample, preds in zip(batches[i], pred_ner):
            sample['pred_spans_info'] = []
            sample['pred_spans'] = []
            sample['pred_spans_ner_label'] = []
            for span, pred in zip(sample['spans'], preds):
                if pred == 0:
                    continue
                mention = sample['tokens'][span[0]:span[1]+1]
                mention = ' '.join(token for token in mention)
                sample['pred_spans_info'].append([span[0], span[1], ner_id2label[pred], 'role', 'normalized', mention])
                sample['pred_spans'].append([span[0], span[1]])
                sample['pred_spans_ner_label'].append(ner_id2label[pred])
            
            spans = []
            spans_ner_label = []
            for span, span_ner_label in zip(sample['spans'], sample['spans_ner_label']):
                if span_ner_label != 0:
                    spans.append([span[0], span[1]])
                    spans_ner_label.append(ner_id2label[span_ner_label])
            sample['spans'] = spans
            sample['spans_ner_label'] = spans_ner_label
            sample['spans_role_label'] = [role_id2label[span_role_label] for span_role_label in sample['spans_role_label']]
            
    logger.info('Output predictions to %s..'%(output_file))
    with open(output_file, 'w') as f:
        f.write('\n'.join(json.dumps(sample) for batch in batches for sample in batch))

def evaluate(model, batches):
    """
    Evaluate the entity model
    """
    logger.info('Evaluating...')
    c_time = time.time()
    cor = 0
    tot_gold = 0
    tot_pred = 0
    l_cor = 0
    l_tot = 0
    
    cor_type = dict.fromkeys(range(8), 0)
    tot_gold_type = dict.fromkeys(range(8), 0)
    tot_pred_type = dict.fromkeys(range(8), 0)
    l_cor_type = dict.fromkeys(range(8), 0)
    l_tot_type = dict.fromkeys(range(8), 0)

    for i in range(len(batches)):
        output_dict = model.run_batch(batches[i], training=False)
        pred_ner = output_dict['pred_ner']
        for sample, preds in zip(batches[i], pred_ner):
            for gold, pred in zip(sample['spans_ner_label'], preds):
                l_tot += 1
                if gold != 0:
                    tot_gold += 1
                    tot_gold_type[gold] += 1
                if pred == gold:
                    l_cor += 1
                    l_cor_type[pred] += 1
                if pred != 0 and gold != 0 and pred == gold:
                    cor += 1
                    cor_type[pred] += 1
                if pred != 0:
                    tot_pred += 1
                    tot_pred_type[pred] += 1
                   
    acc = l_cor / l_tot
    logger.info('Accuracy: %5f'%acc)
    logger.info('Cor: %d, Pred TOT: %d, Gold TOT: %d'%(cor, tot_pred, tot_gold))
    p = cor / tot_pred if cor > 0 else 0.0
    r = cor / tot_gold if cor > 0 else 0.0
    f1 = 2 * (p * r) / (p + r) if cor > 0 else 0.0
    logger.info('P: %.5f, R: %.5f, F1: %.5f'%(p, r, f1))
    for idx in range(1, len(ner_labels)):
        p = cor_type[idx] / tot_pred_type[idx] if cor_type[idx] > 0 else 0.0
        r = cor_type[idx] / tot_gold_type[idx] if cor_type[idx] > 0 else 0.0
        f1 = 2 * (p * r) / (p + r) if cor_type[idx] > 0 else 0.0
        logger.info('%s P: %.5f, R: %.5f, F1: %.5f, Cor: %d, Pred TOT: %d, Gold TOT: %d'%(ner_labels[idx], p, r, f1, cor_type[idx], tot_pred_type[idx], tot_gold_type[idx]))
    logger.info('Used time: %f'%(time.time()-c_time))
    return f1

def setseed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    models_path = '/your/model/path/'
    for model_name in os.listdir(models_path):
        model_path = os.path.join(models_path, model_name)
        data_dir = '/your/dataset/'
        output_dir = '/your/output/'
        model_save_path = '/your/model/save_path/'
        test_pred_filename = 'test_predict_ner.json'
        dev_pred_filename = 'dev_predict_ner.json'

        eval_batch_size = 16
        train_batch_size = 16
        task_learning_rate = 5e-4
        learning_rate = 1e-5
        num_epoch = 5
        warmup_proportion = 0.1
        eval_per_epoch = 2
        train_shuffle = True
        print_loss_step = 300
        max_span_length = 8
        do_train = True
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
        model = EntityModel(model_path, num_ner_labels=num_ner_labels)

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
                    loss = output_dict['ner_loss']
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

        if do_eval:
            model = EntityModel(model_save_path, num_ner_labels=num_ner_labels)
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
            output_ner_predictions(model, test_batches, output_file=prediction_file)
