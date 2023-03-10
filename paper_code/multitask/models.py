import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from allennlp.nn.util import batched_index_select
from allennlp.nn import util, Activation
from allennlp.modules import FeedForward

from transformers import BertPreTrainedModel, BertModel, BertTokenizer

import os
import json
import logging

logger = logging.getLogger('root')

class MultiTaskBert(BertPreTrainedModel):
    def __init__(self, config, num_ner_labels=8, num_role_labels=7, 
                 head_hidden_dim=150, width_embedding_dim=150, max_span_length=8, task_num=2):
        super().__init__(config)
        
        self.num_ner_labels = num_ner_labels
        self.num_role_labels = num_role_labels
        self.task_num = task_num
        
        self.log_vars = nn.Parameter(torch.zeros(self.task_num))
        
        self.bert = BertModel(config)
        self.hidden_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.width_embedding = nn.Embedding(max_span_length+1, width_embedding_dim)
    
        self.ner_classifier = nn.Sequential(
            FeedForward(input_dim=config.hidden_size*2+width_embedding_dim,
                       num_layers=2,
                       hidden_dims=head_hidden_dim,
                       activations=nn.ReLU(),
                       dropout=0.2),
            nn.Linear(head_hidden_dim, num_ner_labels)
        )
        
        self.role_classifier = nn.Sequential(
            FeedForward(input_dim=config.hidden_size*2,
                       num_layers=2,
                       hidden_dims=head_hidden_dim,
                       activations=nn.ReLU(),
                       dropout=0.2),
            nn.Linear(head_hidden_dim, num_role_labels)
        )
        
        self.init_weights() 
    
    def _get_span_embeddings(self, input_ids, spans, token_type_ids=None, attention_mask=None):
        sequence_output, pooled_output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output = self.hidden_dropout(sequence_output)
        """
        spans: [batch_size, num_spans, 3]; 0: left_ned, 1: right_end, 2: width
        """
        spans_start = spans[:, :, 0].view(spans.size(0), -1)
        spans_start_embedding = batched_index_select(sequence_output, spans_start)
        spans_end = spans[:, :, 1].view(spans.size(0), -1)
        spans_end_embedding = batched_index_select(sequence_output, spans_end)

        spans_width = spans[:, :, 2].view(spans.size(0), -1)
        spans_width_embedding = self.width_embedding(spans_width)
#         logger.info(spans_width)

        # Concatenate embeddings of left/right points and the width embedding
        ner_spans_embedding = torch.cat((spans_start_embedding, spans_end_embedding, spans_width_embedding), dim=-1)
        role_spans_embedding = torch.cat((spans_start_embedding, spans_end_embedding), dim=-1)
        """
        spans_embedding: (batch_size, num_spans, hidden_size*2+embedding_dim)
        """
        return ner_spans_embedding, role_spans_embedding
    
    def forward(self, input_ids, spans=None, spans_mask=None, spans_ner_label=None, spans_role_label=None, token_type_ids=None, attention_mask=None):
        ner_spans_embedding, role_spans_embedding = self._get_span_embeddings(input_ids, spans, token_type_ids=token_type_ids, attention_mask=attention_mask)
        ner_ffnn_hidden = []
        role_ffnn_hidden = []
        hidden = ner_spans_embedding
        for layer in self.ner_classifier:
            hidden = layer(hidden)
            ner_ffnn_hidden.append(hidden)
        ner_logits = ner_ffnn_hidden[-1]

        hidden = role_spans_embedding
        for layer in self.role_classifier:
            hidden = layer(hidden)
            role_ffnn_hidden.append(hidden)
        role_logits = role_ffnn_hidden[-1]

        if spans_ner_label is not None and spans_role_label is not None:  # jointly training
            loss_func_ner = CrossEntropyLoss(reduce='sum')
            loss_func_role = CrossEntropyLoss(reduce='sum')

            if attention_mask is not None:
                active_loss = spans_mask.view(-1) == 1
                active_ner_logits = ner_logits.view(-1, ner_logits.shape[-1])
                active_ner_labels = torch.where(
                    active_loss, spans_ner_label.view(-1), torch.tensor(loss_func_ner.ignore_index).type_as(spans_ner_label)
                )
                loss_ner = loss_func_ner(active_ner_logits, active_ner_labels)

                active_role_logits = role_logits.view(-1, role_logits.shape[-1])
                active_role_labels = torch.where(
                    active_loss, spans_role_label.view(-1), torch.tensor(loss_func_role.ignore_index).type_as(spans_role_label)
                )
                loss_role = loss_func_role(active_role_logits, active_role_labels)
            else:
                loss_ner = loss_func_ner(ner_logits.view(-1, self.num_ner_labels, spans_ner_label.view(-1)))
                loss_role = loss_func_role(role_logits.view(-1, self.num_role_labels, spans_role_label.view(-1)))

            # a re-implemention of cvpr2018 paper "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"
            precision_role = torch.exp(-self.log_vars[0])
            loss0 = precision_role*loss_role + self.log_vars[0]

            precision_entity = torch.exp(-self.log_vars[1])
            loss1 = precision_entity*loss_ner + self.log_vars[1]

            return loss0+loss1

        else:
            return ner_logits, role_logits, ner_spans_embedding

        
class MultiTaskModel():

    def __init__(self, model_path, num_ner_labels, num_role_labels):
        super().__init__()

        logger.info('Loading BERT model from {}'.format(model_path))

        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.bert_model = MultiTaskBert.from_pretrained(model_path, num_ner_labels=num_ner_labels, num_role_labels=num_role_labels, max_span_length=8, return_dict=False)

        self._model_device = 'cpu'
        self.move_model_to_cuda()

    def move_model_to_cuda(self):
        if not torch.cuda.is_available():
            logger.error('No CUDA found!')
            exit(-1)
        logger.info('Moving to CUDA...')
        self._model_device = 'cuda'
        self.bert_model.cuda()
        logger.info('# GPUs = %d'%(torch.cuda.device_count()))
        if torch.cuda.device_count() > 1:
            self.bert_model = torch.nn.DataParallel(self.bert_model)

    def _get_input_tensors(self, tokens, spans, spans_ner_label, spans_role_label):
        start2idx = []
        end2idx = []
        
        bert_tokens = []
        bert_tokens.append(self.tokenizer.cls_token)
        for token in tokens:
            start2idx.append(len(bert_tokens))
            sub_tokens = self.tokenizer.tokenize(token)
            bert_tokens += sub_tokens
            end2idx.append(len(bert_tokens)-1)
        bert_tokens.append(self.tokenizer.sep_token)

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(bert_tokens)
        tokens_tensor = torch.tensor([indexed_tokens])

        bert_spans = [[start2idx[span[0]], end2idx[span[1]], span[2]] for span in spans]
        bert_spans_tensor = torch.tensor([bert_spans])

        spans_ner_label_tensor = torch.tensor([spans_ner_label])
        spans_role_label_tensor = torch.tensor([spans_role_label])

        return tokens_tensor, bert_spans_tensor, spans_ner_label_tensor, spans_role_label_tensor

    def _get_input_tensors_batch(self, samples_list):
        tokens_tensor_list = []
        bert_spans_tensor_list = []
        spans_ner_label_tensor_list = []
        spans_role_label_tensor_list = []
        sentence_length = []

        max_tokens = 0
        max_spans = 0
        for sample in samples_list:
            tokens = sample['tokens']
            spans = sample['spans']
            spans_ner_label = sample['spans_ner_label']
            spans_role_label = sample['spans_role_label']

            tokens_tensor, bert_spans_tensor, spans_ner_label_tensor, spans_role_label_tensor = self._get_input_tensors(tokens, spans, spans_ner_label, spans_role_label)
            tokens_tensor_list.append(tokens_tensor)
            bert_spans_tensor_list.append(bert_spans_tensor)
            spans_ner_label_tensor_list.append(spans_ner_label_tensor)
            spans_role_label_tensor_list.append(spans_role_label_tensor)
            assert(bert_spans_tensor.shape[1] == spans_role_label_tensor.shape[1])
            if (tokens_tensor.shape[1] > max_tokens):
                max_tokens = tokens_tensor.shape[1]
            if (bert_spans_tensor.shape[1] > max_spans):
                max_spans = bert_spans_tensor.shape[1]
            sentence_length.append(len(sample['tokens']))
        sentence_length = torch.Tensor(sentence_length)

        # apply padding and concatenate tensors
        final_tokens_tensor = None
        final_attention_mask = None
        final_bert_spans_tensor = None
        final_spans_ner_label_tensor = None
        final_spans_role_label_tensor = None
        final_spans_mask_tensor = None
        for tokens_tensor, bert_spans_tensor, spans_ner_label_tensor, spans_role_label_tensor in zip(tokens_tensor_list, bert_spans_tensor_list, spans_ner_label_tensor_list, spans_role_label_tensor_list):
            # padding for tokens
            num_tokens = tokens_tensor.shape[1]
            tokens_pad_length = max_tokens - num_tokens
            attention_tensor = torch.full([1,num_tokens], 1, dtype=torch.long)
            if tokens_pad_length>0:
                pad = torch.full([1,tokens_pad_length], self.tokenizer.pad_token_id, dtype=torch.long)
                tokens_tensor = torch.cat((tokens_tensor, pad), dim=1)
                attention_pad = torch.full([1,tokens_pad_length], 0, dtype=torch.long)
                attention_tensor = torch.cat((attention_tensor, attention_pad), dim=1)

            # padding for spans
            num_spans = bert_spans_tensor.shape[1]
            spans_pad_length = max_spans - num_spans
            spans_mask_tensor = torch.full([1,num_spans], 1, dtype=torch.long)
            if spans_pad_length>0:
                pad = torch.full([1,spans_pad_length,bert_spans_tensor.shape[2]], 0, dtype=torch.long)
                bert_spans_tensor = torch.cat((bert_spans_tensor, pad), dim=1)
                mask_pad = torch.full([1,spans_pad_length], 0, dtype=torch.long)
                spans_mask_tensor = torch.cat((spans_mask_tensor, mask_pad), dim=1)
                spans_ner_label_tensor = torch.cat((spans_ner_label_tensor, mask_pad), dim=1)
                spans_role_label_tensor = torch.cat((spans_role_label_tensor, mask_pad), dim=1)

            # update final outputs
            if final_tokens_tensor is None:
                final_tokens_tensor = tokens_tensor
                final_attention_mask = attention_tensor
                final_bert_spans_tensor = bert_spans_tensor
                final_spans_ner_label_tensor = spans_ner_label_tensor
                final_spans_role_label_tensor = spans_role_label_tensor
                final_spans_mask_tensor = spans_mask_tensor
            else:
                final_tokens_tensor = torch.cat((final_tokens_tensor,tokens_tensor), dim=0)
                final_attention_mask = torch.cat((final_attention_mask, attention_tensor), dim=0)
                final_bert_spans_tensor = torch.cat((final_bert_spans_tensor, bert_spans_tensor), dim=0)
                final_spans_ner_label_tensor = torch.cat((final_spans_ner_label_tensor, spans_ner_label_tensor), dim=0)
                final_spans_role_label_tensor = torch.cat((final_spans_role_label_tensor, spans_role_label_tensor), dim=0)
                final_spans_mask_tensor = torch.cat((final_spans_mask_tensor, spans_mask_tensor), dim=0)
        #logger.info(final_tokens_tensor)
        #logger.info(final_attention_mask)
        #logger.info(final_bert_spans_tensor)
        #logger.info(final_bert_spans_tensor.shape)
        #logger.info(final_spans_mask_tensor.shape)
        #logger.info(final_spans_role_label_tensor.shape)
        return final_tokens_tensor, final_attention_mask, final_bert_spans_tensor, final_spans_mask_tensor, final_spans_ner_label_tensor, final_spans_role_label_tensor, sentence_length

    def run_batch(self, samples_list, training=True):
        # convert samples to input tensors
        tokens_tensor, attention_mask_tensor, bert_spans_tensor, spans_mask_tensor, spans_ner_label_tensor, spans_role_label_tensor, sentence_length = self._get_input_tensors_batch(samples_list)

        output_dict = {
            'multi_loss': 0,
        }

        if training:
            self.bert_model.train()
            multi_loss = self.bert_model(
                input_ids = tokens_tensor.to(self._model_device),
                spans = bert_spans_tensor.to(self._model_device),
                spans_mask = spans_mask_tensor.to(self._model_device),
                spans_ner_label = spans_ner_label_tensor.to(self._model_device),
                spans_role_label = spans_role_label_tensor.to(self._model_device),
                attention_mask = attention_mask_tensor.to(self._model_device),
            )
            output_dict['multi_loss'] = multi_loss.sum()
        else:
            self.bert_model.eval()
            with torch.no_grad():
                ner_logits, role_logits, last_hidden = self.bert_model(
                    input_ids = tokens_tensor.to(self._model_device),
                    spans = bert_spans_tensor.to(self._model_device),
                    spans_mask = spans_mask_tensor.to(self._model_device),
                    spans_ner_label = None,
                    spans_role_label = None,
                    attention_mask = attention_mask_tensor.to(self._model_device),
                )
                
            _, predicted_ner_label = ner_logits.max(2)
            predicted_ner_label = predicted_ner_label.cpu().numpy()
            
            _, predicted_role_label = role_logits.max(2)
            predicted_role_label = predicted_role_label.cpu().numpy()
            last_hidden = last_hidden.cpu().numpy()
            
            predicted_ner = []
            pred_ner_prob = []
            predicted_role = []
            pred_role_prob = []
            hidden = []
            for i, sample in enumerate(samples_list):
                ner = []
                role = []
                ner_prob = []
                role_prob = []
                lh = []
                for j in range(len(sample['spans'])):
                    ner.append(predicted_ner_label[i][j])
                    role.append(predicted_role_label[i][j])
                    # prob.append(F.softmax(role_logits[i][j], dim=-1).cpu().numpy())
                    ner_prob.append(ner_logits[i][j].cpu().numpy())
                    role_prob.append(role_logits[i][j].cpu().numpy())
                    lh.append(last_hidden[i][j])
                predicted_ner.append(ner)
                pred_ner_prob.append(ner_prob)
                predicted_role.append(role)
                pred_role_prob.append(role_prob)
                hidden.append(lh)
            output_dict['pred_ner'] = predicted_ner
            output_dict['ner_probs'] = pred_ner_prob
            output_dict['pred_role'] = predicted_role
            output_dict['role_probs'] = pred_role_prob
            output_dict['last_hidden'] = hidden

        return output_dict
