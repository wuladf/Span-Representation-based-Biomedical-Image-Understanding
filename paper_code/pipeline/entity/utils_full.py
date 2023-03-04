import numpy as np
import json
import logging

logger = logging.getLogger('root')

def batchify(samples, batch_size):
    """
    Batchfy samples with a batch size
    """
    num_samples = len(samples)

    list_samples_batches = []
    
    # if a sentence is too long, make itself a batch to avoid GPU OOM
    to_single_batch = []
    for i in range(0, len(samples)):
        if len(samples[i]['tokens']) > 350:
            to_single_batch.append(i)
    
    for i in to_single_batch:
        list_samples_batches.append([samples[i]])
    samples = [sample for i, sample in enumerate(samples) if i not in to_single_batch]

    for i in range(0, len(samples), batch_size):
        list_samples_batches.append(samples[i:i+batch_size])

    assert(sum([len(batch) for batch in list_samples_batches]) == num_samples)

    return list_samples_batches

def convert_dataset_to_samples_sent_level(dataset_file, max_span_length, ner_label2id=None, role_label2id=None):
    f_w = open(dataset_file, 'r', encoding='utf8')
    samples = []
    for line in f_w:
        panel = json.loads(line)
        assert(len(panel['sentences']) == len(panel['spans']))
        for i, sent in enumerate(panel['sentences']):
            sample = {}
            # those fileds are for recover from sents to panels, 
            # even we shuffle the dataset during training and predicting, 
            # as long as the panels' sents are not split into different dataset part
            sample['panel_id'] = panel['panel_id']
            sample['panel_length'] = panel['panel_length']
            sample['sentence_ix'] = i
            
            # this field is unnessary, since we are already in sentence.
            # however, we can keep some redundant info.
            sample['sent_length'] = len(sent)
            
            sample['tokens'] = sent
            sample['spans_info'] = panel['spans_info'][i]
            sample['spans'] = []
            sample['spans_ner_label'] = []
#             sample['spans_role_label'] = panel['spans_role_label'][i]
            sample['spans_role_label'] = []
            
            spans = panel['spans'][i]
            ners = panel['spans_ner_label'][i]
            roles = panel['spans_role_label'][i]
#             for span in panel['spans'][i]:
#                 spans.append((span[0], span[1]))
#                 ners.append(span[2])
#                 roles.append(span[3])
            
            for j in range(len(sent)):
                for k in range(j, min(len(sent), j+max_span_length)):
                    sample['spans'].append((j, k, k-j+1))
                    if [j, k] not in spans:
                        sample['spans_ner_label'].append(0)
#                         sample['spans_role_label'].append(0)
                    else:
                        idx = spans.index([j, k])
                        sample['spans_ner_label'].append(ner_label2id[ners[idx]])
                        sample['spans_role_label'].append(role_label2id[roles[idx]])
            samples.append(sample)
    
    avg_length = sum([len(sample['tokens']) for sample in samples]) / len(samples)
    max_length = max([len(sample['tokens']) for sample in samples])
    logger.info('Extracted %d samples, %.3f avg input length, %d max length'%(len(samples), avg_length, max_length))
            
    return samples


def convert_dataset_to_samples_panel_level(dataset_file, max_span_length, ner_label2id=None, role_label2id=None):
    f_w = open(dataset_file, 'r', encoding='utf8')
    samples = []
    for line in f_w:
        panel = json.loads(line)
        sample = {}
        
        # those two fileds may be unnessary, since we are already in panel.
        # however, we can keep some redundant info.
        sample['panel_id'] = panel['panel_id']
#         sample['sents_length'] = [len(sent) for sent in panel['sentences']]
        sample['sents_length'] = panel['sents_length']
        
        sample['tokens'] = []
        sample['spans_info'] = []
        sample['spans'] = []
        sample['spans_ner_label'] = []
        sample['spans_role_label'] = []

    #     sample['tokens'] = [token for sent in panel['sentences'] for token in sent]
        cnt = 0
        for i, sent in enumerate(panel['sentences']):
            spans = []
            ners = []
            roles = []
            sample['tokens'].extend(sent)
            sample['spans_info'].extend(panel['spans_info'][i])
            
            for span in panel['spans'][i]:
                spans.append((span[0]+cnt, span[1]+cnt))
                ners.append(span[2])
                roles.append(span[3])

            for j in range(len(sent)):
                for k in range(j, min(len(sent), j+max_span_length)):
                    sample['spans'].append((j+cnt, k+cnt, k-j+1))
                    if (j+cnt, k+cnt) not in spans:
                        sample['spans_ner_label'].append(0)
#                         sample['spans_role_label'].append(0)
                    else:
                        idx = spans.index((j+cnt, k+cnt))
                        sample['spans_ner_label'].append(ner_label2id[ners[idx]])
                        sample['spans_role_label'].append(role_label2id[roles[idx]])

            cnt += len(sent)

        samples.append(sample)
        
    avg_length = sum([len(sample['tokens']) for sample in samples]) / len(samples)
    max_length = max([len(sample['tokens']) for sample in samples])
    logger.info('Extracted %d samples, %.3f avg input length, %d max length'%(len(samples), avg_length, max_length))

    return samples

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
#             spans_role_label = []
            for span, span_ner_label in zip(sample['spans'], sample['spans_ner_label']):
                if span_ner_label != 0:
#                     mention = sample['tokens'][span[0]:span[1]+1]
#                     mention = ' '.join(token for token in mention)
#                     spans.append(span)
                    spans.append([span[0], span[1]])
                    spans_ner_label.append(ner_id2label[span_ner_label])
#                     spans_role_label.append(span_role_label)
            sample['spans'] = spans
            sample['spans_ner_label'] = spans_ner_label
            sample['spans_role_label'] = [role_id2label[span_role_label] for span_role_label in sample['spans_role_label']]
            
    logger.info('Output predictions to %s..'%(output_file))
    with open(output_file, 'w') as f:
        f.write('\n'.join(json.dumps(sample) for batch in batches for sample in batch))

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

