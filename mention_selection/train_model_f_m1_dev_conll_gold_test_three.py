# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 10:51:41 2019

@author: chenm
"""

import numpy
import time
import torch
import torch.nn as NN
import torch.optim as OPT
import torch.nn.functional as F
from torch.autograd import Variable
from pytorchtools import EarlyStopping
from pytorch_pretrained_bert import BertTokenizer, BertModel

import argparse
import pickle
from os import listdir
from os.path import isfile, join

class MentionInfo(object):
    def __init__(self, text, index_in_sentence, sentence_text, sentence_num):
        self._text = text
        self._index_in_sentence = index_in_sentence
        self._sentence_text = sentence_text
        self._sentence_num = sentence_num
        
class RawSequenceData(object):
    def __init__(self, 
                 is_identified_mention,
                 is_annotated_mention,
                 is_identified_verb,
                 is_annotated_verb,
                 start_pos,
                 identified_original_focus_mention,
                 annotated_original_focus_mention,
                 identified_converted_focus_mention,
                 annotated_converted_focus_mention,
                 identified_original_verb,
                 annotated_original_verb,
                 identified_converted_verb,
                 annotated_converted_verb,
                 original_focus_mention,
                 converted_focus_mention,
                 focus_mention_cluster_id,
                 focus_mention_sentence_num,
                 focus_mention_index_in_sentence,
                 is_subject,
                 is_object,
                 related_mentions,
                 related_mention_cluster_ids,
                 related_original_mention_text_list,
                 related_converted_mention_text_list,
                 raw_sequence,
                 postag_sequence,
                 related_sentence_num_to_sentence_text,
                 start_token_index_in_sentence,
                 end_token_index_in_sentence,
                 original_pre_mention_sequence,
                 converted_pre_mention_sequence,
                 pre_mention_cluster_id_sequence,
                 pre_mention_distance_sequence,
                 pre_mention_info_list,
                 original_post_mention_sequence,
                 converted_post_mention_sequence,
                 post_mention_cluster_id_sequence,
                 post_mention_distance_sequence,
                 post_mention_info_list):
        self._is_identified_mention = is_identified_mention
        self._is_annotated_mention = is_annotated_mention
        self._is_identified_verb = is_identified_verb
        self._is_annotated_verb = is_annotated_verb
        self._start_pos = start_pos
        self._identified_original_focus_mention = identified_original_focus_mention
        self._annotated_original_focus_mention = annotated_original_focus_mention
        self._identified_converted_focus_mention = identified_converted_focus_mention
        self._annotated_converted_focus_mention = annotated_converted_focus_mention
        self._identified_original_verb = identified_original_verb
        self._annotated_original_verb = annotated_original_verb
        self._identified_converted_verb = identified_converted_verb
        self._annotated_converted_verb = annotated_converted_verb
        self._original_focus_mention = original_focus_mention
        self._converted_focus_mention = converted_focus_mention
        self._focus_mention_cluster_id = focus_mention_cluster_id
        self._focus_mention_sentence_num = focus_mention_sentence_num
        self._focus_mention_index_in_sentence = focus_mention_index_in_sentence
        self._is_subject = is_subject
        self._is_object = is_object
        self._related_mentions = related_mentions
        self._related_mention_cluster_ids = related_mention_cluster_ids
        self._related_original_mention_text_list = related_original_mention_text_list
        self._related_converted_mention_text_list = related_converted_mention_text_list
        self._raw_sequence = raw_sequence
        self._postag_sequence = postag_sequence
        self._related_sentence_num_to_sentence_text = related_sentence_num_to_sentence_text
        self._start_token_index_in_sentence = start_token_index_in_sentence
        self._end_token_index_in_sentence = end_token_index_in_sentence
        self._original_pre_mention_sequence = original_pre_mention_sequence
        self._converted_pre_mention_sequence = converted_pre_mention_sequence
        self._pre_mention_cluster_id_sequence = pre_mention_cluster_id_sequence
        self._pre_mention_distance_sequence = pre_mention_distance_sequence
        self._pre_mention_info_list = pre_mention_info_list
        self._original_post_mention_sequence = original_post_mention_sequence
        self._converted_post_mention_sequence = converted_post_mention_sequence
        self._post_mention_cluster_id_sequence = post_mention_cluster_id_sequence
        self._post_mention_distance_sequence = post_mention_distance_sequence
        self._post_mention_info_list = post_mention_info_list

class RawClusterData(object):
    def __init__(self, cluster_id, common_mention, max_count, total_count, mention_list, postags_list):
        self._cluster_id = cluster_id
        self._common_mention = common_mention
        self._max_count = max_count
        self._total_count = total_count
        self._mention_list = mention_list
        self._postags_list = postags_list
    
class RawDocumentData(object):
    def __init__(self, document_text, gold_cluster_id_to_cluster_data, auto_cluster_id_to_cluster_data, gold_invalid_cluster_id_to_cluster_data, auto_invalid_cluster_id_to_cluster_data, raw_sequence_data_list):
        self._document_text = document_text
        self._gold_cluster_id_to_cluster_data = gold_cluster_id_to_cluster_data
        self._auto_cluster_id_to_cluster_data = auto_cluster_id_to_cluster_data
        self._gold_invalid_cluster_id_to_cluster_data = gold_invalid_cluster_id_to_cluster_data
        self._auto_invalid_cluster_id_to_cluster_data = auto_invalid_cluster_id_to_cluster_data
        self._raw_sequence_data_list = raw_sequence_data_list
        
# =============================================================================
# Model 1
#
#  This model used two LSTMs, one from the left to right, one from right to 
#  left. The output of the left and right LSTM will be concatenated and the 
#  output will be used as the input of a forward network
#
#  lstm_size: the hidden size of the LSTM
#  hidden_size: the size of the fully connected layers.
#  drop_rate: Dropout rate.
#  rep_l1: the matrix of word embeddings for the input sequence 1 of lstml.
#  len_l1: the true length of the input sequence 1 of lstml.
#  rep_l2: the matrix of word embeddings for the input sequence 2 of lstml.
#  len_l2: the true length of the input sequence 2 of lstml
#  rep_r: the matrix of word embeddings for the input sequence of lstmr.
#  len_r: the true length of the input sequence of lstmr
#

class Model_1(NN.Module):
    def __init__(self, use_gpu, lstm_size, hidden_size, drop_out_1, drop_out_2):
        super(Model_1, self).__init__()

        numpy.random.seed(2)
        torch.manual_seed(2)

        # Set tensor type when using GPU
        if use_gpu:
            self.float_type = torch.cuda.FloatTensor
            self.long_type = torch.cuda.LongTensor
            torch.cuda.manual_seed_all(2)
        # Set tensor type when using CPU
        else:
            self.float_type = torch.FloatTensor
            self.long_type = torch.LongTensor
        
        # Define parameters for model
        self.random_v = torch.nn.Parameter(torch.randn(1, 768))
        self.lstm_size = lstm_size
        self.hidden_size = hidden_size
        feature_size = 768
        self.drop_out_1 = drop_out_1
        self.drop_out_2 = drop_out_2

        # The LSTMs:
        # lstm1: from left to right; lstm2: from right to left
        self.lstm1 = NN.LSTMCell(feature_size, lstm_size)
        self.lstm2 = NN.LSTMCell(feature_size, lstm_size)
        
        # The fully connectedy layers
        self.linear1 = NN.Linear(lstm_size * 2 + 18, hidden_size)

        # The fully connectedy layer for scoring
        self.linear4 = NN.Linear(hidden_size, 1)

    # Initialize the hidden states and cell states of LSTM
    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.lstm_size).type(self.float_type),
                torch.zeros(1, batch_size, self.lstm_size).type(self.float_type))

    def get_updated_variable(self, data_list, max_len):
        result_list = []
        length_list = []
        for i in range(len(data_list)):
            data = data_list[i]
            list_1 = data[:50]
            v_1 = Variable(torch.from_numpy(numpy.asarray(list_1, dtype=numpy.float32)), requires_grad=False).type(self.float_type)
            list_3 = data[50:]
            v_3 = Variable(torch.from_numpy(numpy.asarray(list_3, dtype=numpy.float32)), requires_grad=False).type(self.float_type)
            
            length_list.append(len(data)+1)
            
            padding = False
            v_x = None
            if len(data) + 1 < max_len:
                padding = True
                list_x = []
                for j in range(max_len - len(data) - 1):
                    temp = [0] * 768
                    list_x.append(temp)
                v_x = Variable(torch.from_numpy(numpy.asarray(list_x, dtype=numpy.float32)), requires_grad=False).type(self.float_type)
            
            v_4 = None
            if padding:
                v_4 = torch.cat((v_1, self.random_v, v_3, v_x), 0)
            else:
                v_4 = torch.cat((v_1, self.random_v, v_3), 0)
            result_list.append(v_4)
        
        result_variable = torch.stack(result_list, 0)
        length_list = numpy.asarray(length_list, dtype=numpy.int32)
                                    
        return result_variable, length_list
    
    # Forward process
    def forward(self, seql_1_list, seql_2_list, max_seql_length, rep_r, len_r, first_mention_bit, second_mention_bit, focus_mention_length, other_mention_length, coreference_chain_length, used_bit_seql_1, same_bit_seql_1, used_bit_seql_2, same_bit_seql_2, is_subject_list, is_object_list):
        rep_l1, batch_len_l1 = self.get_updated_variable(seql_1_list, max_seql_length)
        rep_l2, batch_len_l2 = self.get_updated_variable(seql_2_list, max_seql_length)
        
        len_l1 = Variable(torch.from_numpy(batch_len_l1),
                          requires_grad=False).type(self.long_type)
        len_l2 = Variable(torch.from_numpy(batch_len_l2),
                          requires_grad=False).type(self.long_type)
                    
        # Set batch size
        batch_size_l = rep_l1.size()[0]
        batch_size_r = rep_r.size()[0]
        max_seq_len_l = rep_l1.size()[1]
        max_seq_len_r = rep_r.size()[1]
        
        # Representation of input sequences
        seq_l1 = rep_l1
        seq_l2 = rep_l2
        seq_r = rep_r
        
        # Transform sequence representations to:
        # (sequence length * batch size * feature size)
        seq_l1 = seq_l1.transpose(1, 0)
        seq_l2 = seq_l2.transpose(1, 0)
        seq_r = seq_r.transpose(1, 0)
        
        first_mention_bit = first_mention_bit.view(batch_size_l, -1)
        second_mention_bit = second_mention_bit.view(batch_size_l, -1)
        
        used_bit_seql_1 = used_bit_seql_1.view(batch_size_l, -1)
        same_bit_seql_1 = same_bit_seql_1.view(batch_size_l, -1)
        used_bit_seql_2 = used_bit_seql_2.view(batch_size_l, -1)
        same_bit_seql_2 = same_bit_seql_2.view(batch_size_l, -1)
        is_subject_list = is_subject_list.view(batch_size_l, -1)
        is_object_list = is_object_list.view(batch_size_l, -1)
        
        hidden_r = self.init_hidden(batch_size_r)
        states_r = (hidden_r[0].view(batch_size_r, -1), hidden_r[1].view(batch_size_r, -1))
        
        lstm_outs_r = []
        cell_states_r = []
        for i in range(max_seq_len_r):
            states_r = self.lstm2(seq_r[i], states_r)
            lstm_outs_r.append(states_r[0].view(1, batch_size_r, -1))
            cell_states_r.append(states_r[1].view(1, batch_size_r, -1))
            
        lstm_outs_r = torch.cat(lstm_outs_r, 0)
        
        lstm_outs_r = lstm_outs_r.transpose(1, 0)
        length_r = (len_r-1).view(-1, 1, 1).expand(lstm_outs_r.size(0), 1, lstm_outs_r.size(2))
        hidden_two_r = torch.gather(lstm_outs_r, 1, length_r)
        # Last hidden states of LSTM_r
        hidden_two_r = hidden_two_r.transpose(1, 0)
        hidden_two_r = hidden_two_r.view(hidden_two_r.size(1), -1)
        
        # LSTM_l: (sequence length * mini batch * lstm size)
        # Input sequence 1
        hidden_l1 = self.init_hidden(batch_size_l)
        states_l1 = (hidden_l1[0].view(batch_size_l, -1), hidden_l1[1].view(batch_size_l, -1))

        lstm_outs_l1 = []
        cell_states_l1 = []
        for i in range(max_seq_len_l):
            states_l1 = self.lstm1(seq_l1[i], states_l1)
            lstm_outs_l1.append(states_l1[0].view(1, batch_size_l, -1))
            cell_states_l1.append(states_l1[1].view(1, batch_size_l, -1))
    
        lstm_outs_l1 = torch.cat(lstm_outs_l1, 0)

        lstm_outs_l1 = lstm_outs_l1.transpose(1, 0)
        length_l1 = (len_l1-1).view(-1, 1, 1).expand(lstm_outs_l1.size(0), 1, lstm_outs_l1.size(2))
        hidden_one_l1 = torch.gather(lstm_outs_l1, 1, length_l1)
        # Last hidden states of LSTM_l
        hidden_one_l1 = hidden_one_l1.transpose(1, 0)
        hidden_one_l1 = hidden_one_l1.view(hidden_one_l1.size(1), -1)
        
        # Concatenate the final hidden state of LSTM_l and LSTM_r
        lstm_out_l1 = torch.cat((hidden_one_l1, hidden_two_r), 1)
        lstm_out_l1 = F.dropout(lstm_out_l1, p=self.drop_out_1)
        
        lstm_out_l1 = torch.cat((lstm_out_l1, first_mention_bit, second_mention_bit, focus_mention_length, coreference_chain_length, used_bit_seql_1, same_bit_seql_1, is_subject_list, is_object_list), 1)
        # Fully connected layers
        fc_out_l1 = F.dropout(torch.tanh(self.linear1(lstm_out_l1)), p=self.drop_out_2)

        # Fully connected layer for scoring
        fc_out_l1 = self.linear4(fc_out_l1)

        # Input sequence 2
        hidden_l2 = self.init_hidden(batch_size_l)
        states_l2 = (hidden_l2[0].view(batch_size_l, -1), hidden_l2[1].view(batch_size_l, -1))

        lstm_outs_l2 = []
        cell_states_l2 = []
        for i in range(max_seq_len_l):
            states_l2 = self.lstm1(seq_l2[i], states_l2)
            lstm_outs_l2.append(states_l2[0].view(1, batch_size_l, -1))
            cell_states_l2.append(states_l2[1].view(1, batch_size_l, -1))
    
        lstm_outs_l2 = torch.cat(lstm_outs_l2, 0)

        lstm_outs_l2 = lstm_outs_l2.transpose(1, 0)
        length_l2 = (len_l2-1).view(-1, 1, 1).expand(lstm_outs_l2.size(0), 1, lstm_outs_l2.size(2))
        hidden_one_l2 = torch.gather(lstm_outs_l2, 1, length_l2)
        # Last hidden states of LSTM_l
        hidden_one_l2 = hidden_one_l2.transpose(1, 0)
        hidden_one_l2 = hidden_one_l2.view(hidden_one_l2.size(1), -1)
        
        # Concatenate the final hidden state of LSTM_l and LSTM_r
        lstm_out_l2 = torch.cat((hidden_one_l2, hidden_two_r), 1)
        lstm_out_l2 = F.dropout(lstm_out_l2, p=self.drop_out_1)

        lstm_out_l2 = torch.cat((lstm_out_l2, first_mention_bit, second_mention_bit, other_mention_length, coreference_chain_length, used_bit_seql_2, same_bit_seql_2, is_subject_list, is_object_list), 1)
        
        # Fully connected layers
        fc_out_l2 = F.dropout(torch.tanh(self.linear1(lstm_out_l2)), p=self.drop_out_2)

        # Fully connected layer for scoring
        fc_out_l2 = self.linear4(fc_out_l2)
        
        return fc_out_l1, fc_out_l2
        
###############################################################
# Recurrent neural network class
#
class RNNNet1(object):
    def __init__(self, mode):
        self.mode = mode
        self.raw_document_data_list = []
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', max_len=512)
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.bert_model.eval()
        
        # Set tensor type when using GPU
        if torch.cuda.is_available():
            self.use_gpu = True
            self.float_type = torch.cuda.FloatTensor
            self.long_type = torch.cuda.LongTensor
        # Set tensor type when using CPU
        else:
            self.use_gpu = False
            self.float_type = torch.FloatTensor
            self.long_type = torch.LongTensor

    def get_bert_embedding(self, text):
        token_embeddings_for_sentence = []
        tokenized_text = self.tokenizer.tokenize(text)
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens_ids_tensor = torch.tensor([tokens_ids])
        segments_ids = [0] * len(tokenized_text)
        segments_tensors = torch.tensor([segments_ids])
        input_mask = [1]*len(tokenized_text)
        input_mask = torch.tensor([input_mask], dtype=torch.long)

        all_encoder_layers, _ = self.bert_model(tokens_ids_tensor, token_type_ids=segments_tensors, attention_mask=input_mask)
        
        required_encoder_layer = all_encoder_layers[-2]
        required_encoder_layer = required_encoder_layer.view(-1, 768)
        required_encoder_layer_num = required_encoder_layer.data.numpy()
        
        words = text.split(' ')
        index = 0
        for x, word in enumerate(words):
            word_tokens = self.tokenizer.tokenize(word)
            token_embeddings_for_sentence.append(required_encoder_layer_num[index])  
            index = index + len(word_tokens)
         
        return token_embeddings_for_sentence
    
    # Get a batch of data from given data set.
    def get_batch(self, training_data_path, index):
        training_data_path = training_data_path+"_"+str(index)+".pkl"
        
        data_set = pickle.load(open(training_data_path, 'rb'))
        
        seq_l1 = data_set[0]
        seq_l2 = data_set[1]
        seq_r = data_set[2]
        len_r = data_set[3]
        first_mention_bit = data_set[21]
        second_mention_bit = data_set[22]
        focus_mention_length = data_set[23]
        other_mention_length = data_set[24]
        coreference_chain_length = data_set[25]
        used_bit_seql_1 = data_set[26]
        same_bit_seql_1 = data_set[27]
        used_bit_seql_2 = data_set[28]
        same_bit_seql_2 = data_set[29]

        is_subject_list = data_set[30]
        is_object_list = data_set[31]
        
        label = data_set[32]
        max_seql_length = data_set[33]
        
        return seq_l1, seq_l2, seq_r, len_r, first_mention_bit, second_mention_bit, focus_mention_length, other_mention_length, coreference_chain_length, used_bit_seql_1, same_bit_seql_1, used_bit_seql_2, same_bit_seql_2, is_subject_list, is_object_list, label, max_seql_length

    def get_cluster_data_from_line(self, line):       
        cluster_id = -1
        common_mention = ""
        max_count = -1
        total_count = -1

        mention_list = []
        postags_list = []
        line = line.strip()
        line_parts = line.split("\t")
        cluster_part = line_parts[0].strip()
        common_mention_part = line_parts[1].strip()
        max_count_part = line_parts[2].strip()
        total_count_part = line_parts[3].strip()
        is_gold_cluster = True
        
        if cluster_part.startswith("<cluster_id>:") or cluster_part.startswith("<gold_cluster_id>:"):
            temp_parts = cluster_part.split()
            cluster_id = int(temp_parts[1])
        elif cluster_part.startswith("<auto_cluster_id>:"):
            temp_parts = cluster_part.split()
            cluster_id = int(temp_parts[1])
            is_gold_cluster = False
        else:
            print("There is an error in the dataset 1.")
            
        if common_mention_part.startswith("<common_mention>:"):
            temp_parts = common_mention_part.split()
            for i in range(1,len(temp_parts)):
                ttt = temp_parts[i].lower()
                # if ttt.endswith('\'s') and len(ttt) != 2:
                #     ttt = ttt.replace('\'s', ' \'s')
                # elif ttt.endswith('’s') and len(ttt) != 2:
                #     ttt = ttt.replace('’s', ' ’s')
                common_mention = common_mention + " " + ttt
            common_mention = common_mention.strip()
        else:
            print("There is an error in the dataset 2.")
        
        if max_count_part.startswith("<max_count>:"):
            temp_parts = max_count_part.split()
            max_count = int(temp_parts[1])
        else:
            print("There is an error in the dataset 3.")
         
        if total_count_part.startswith("<total_count>:"):
            temp_parts = total_count_part.split()
            total_count = int(temp_parts[1])
        else:
            print("There is an error in the dataset 4.")
         
        if is_gold_cluster:
            for i in range(int((len(line_parts)-4) / 2)):
                mention_part = line_parts[4+i*2].strip()
                postags_part = line_parts[5+i*2].strip()
                mention = ""
                postag = ""
                if mention_part.startswith("<mention>:"):
                    temp_parts_1 = mention_part.split()
                    for j in range(1, len(temp_parts_1)):
                        ttt = temp_parts_1[j].lower()
                        mention = mention + " " + ttt
                    mention = mention.strip()
                else:
                    print("There is an error in the dataset 5.")
                
                if postags_part.startswith("<postag>:"):
                    temp_parts_2 = postags_part.split()
                    for j in range(1, len(temp_parts_2)):
                        postag = postag + " " + temp_parts_2[j]
                    postag = postag.strip()
                else:
                    print("There is an error in the dataset 6.")
            
                mention_list.append(mention)
                postags_list.append(postag)
        else:
            for i in range(4, len(line_parts)):
                mention_part = line_parts[i].strip()
                mention = ""
                if mention_part.startswith("<mention>:"):
                    temp_parts = mention_part.split()
                    for j in range(1, len(temp_parts)):
                        ttt = temp_parts[j].lower()
                        mention = mention + " " + ttt
                    mention = mention.strip()
                else:
                    print("There is an error in the dataset 7.")
                mention_list.append(mention)
                
        raw_cluster_data = RawClusterData(cluster_id, common_mention, max_count, total_count, mention_list, postags_list)
        return raw_cluster_data
      
    def get_invalid_cluster_data_from_line(self, line):       
        cluster_id = -1
        common_mention = ""
        max_count = -1
        total_count = -1

        mention_list = []
        postags_list = []
        line = line.strip()
        line_parts = line.split("\t")
        cluster_part = line_parts[0].strip()
        common_mention_part = line_parts[1].strip()
        max_count_part = line_parts[2].strip()
        total_count_part = line_parts[3].strip()
        is_gold_cluster = True
        
        if cluster_part.startswith("<invalid_cluster_id>:") or cluster_part.startswith("<gold_invalid_cluster_id>:"):
            temp_parts = cluster_part.split()
            cluster_id = int(temp_parts[1])
        elif cluster_part.startswith("<auto_invalid_cluster_id>:"):
            temp_parts = cluster_part.split()
            cluster_id = int(temp_parts[1])
            is_gold_cluster = False
        else:
            print("There is an error in the dataset 8.")
            
        if common_mention_part.startswith("<common_mention>:"):
            temp_parts = common_mention_part.split()
            for i in range(1,len(temp_parts)):
                ttt = temp_parts[i].lower()
                common_mention = common_mention + " " + ttt
            common_mention = common_mention.strip()
        else:
            print("There is an error in the dataset 9.")
        
        if max_count_part.startswith("<max_count>:"):
            temp_parts = max_count_part.split()
            max_count = int(temp_parts[1])
        else:
            print("There is an error in the dataset 10.")
        
        if total_count_part.startswith("<total_count>:"):
            temp_parts = total_count_part.split()
            total_count = int(temp_parts[1])
        else:
            print("There is an error in the dataset 11.")
         
        if is_gold_cluster:
            for i in range(int((len(line_parts)-4) / 2)):
                mention_part = line_parts[4+i*2].strip()
                postags_part = line_parts[5+i*2].strip()
                mention = ""
                postag = ""
                if mention_part.startswith("<mention>:"):
                    temp_parts_1 = mention_part.split()
                    for j in range(1, len(temp_parts_1)):
                        ttt = temp_parts_1[j].lower()
                        mention = mention + " " + ttt
                    mention = mention.strip()
                else:
                    print("There is an error in the dataset 12.")
                
                if postags_part.startswith("<postag>:"):
                    temp_parts_2 = postags_part.split()
                    for j in range(1, len(temp_parts_2)):
                        postag = postag + " " + temp_parts_2[j]
                    postag = postag.strip()
                else:
                    print("There is an error in the dataset 13.")
            
                mention_list.append(mention)
                postags_list.append(postag)
        else:
            for i in range(4, len(line_parts)):
                mention_part = line_parts[i].strip()
                mention = ""
                if mention_part.startswith("<mention>:"):
                    temp_parts = mention_part.split()
                    for j in range(1, len(temp_parts)):
                        ttt = temp_parts[j].lower()
                        mention = mention + " " + ttt
                    mention = mention.strip()
                else:
                    print("There is an error in the dataset 14.")
                mention_list.append(mention)
                
        raw_cluster_data = RawClusterData(cluster_id, common_mention, max_count, total_count, mention_list, postags_list)
        return raw_cluster_data
    
    def get_focus_mention_from_line(self, line):
        is_identified_mention = -1
        is_annotated_mention = -1
        is_identified_verb = -1
        is_annotated_verb = -1
        start_pos = -1
        identified_original_focus_mention = ""
        annotated_original_focus_mention = ""
        identified_converted_focus_mention = ""
        annotated_converted_focus_mention = ""
        identified_original_verb = ""
        annotated_original_verb = ""
        identified_converted_verb = ""
        annotated_converted_verb = ""
        original_focus_mention = ""
        converted_focus_mention = ""
        cluster_id = -1
        sentence_num = -1
        index_in_sentence = -1
        is_subject = -1
        is_object = -1
        
        line = line.strip()
        line_parts = line.split("\t")
        is_identified_mention_part = ""
        is_annotated_mention_part = ""
        is_identified_verb_part = ""
        is_annotated_verb_part = ""
        start_pos_part = ""
        identified_original_focus_mention_part = ""
        annotated_original_focus_mention_part = ""
        identified_converted_focus_mention_part = ""
        annotated_converted_focus_mention_part = ""
        identified_original_verb_part = ""
        annotated_original_verb_part = ""
        identified_converted_verb_part = ""
        annotated_converted_verb_part = ""
        original_focus_mention_part = ''
        converted_focus_mention_part = ''
        cluster_part = ''
        sentence_num_part = ''
        index_in_sentence_part = ''
        is_subject_part = ''
        is_object_part = ''

        if line.startswith("<focus_mention>:"):
            original_focus_mention_part = line_parts[0]
            cluster_part = line_parts[len(line_parts)-5].strip()
            sentence_num_part = line_parts[len(line_parts)-4].strip()
            index_in_sentence_part=line_parts[len(line_parts)-3].strip()
            is_subject_part = line_parts[len(line_parts)-2].strip()
            is_object_part = line_parts[len(line_parts)-1].strip()
        elif line.startswith("<original_focus_mention>:"):
            original_focus_mention_part = line_parts[0]
            converted_focus_mention_part = line_parts[1]
            cluster_part = line_parts[len(line_parts)-5].strip()
            sentence_num_part = line_parts[len(line_parts)-4].strip()
            index_in_sentence_part=line_parts[len(line_parts)-3].strip()
            is_subject_part = line_parts[len(line_parts)-2].strip()
            is_object_part = line_parts[len(line_parts)-1].strip()
        elif line.startswith("<identified_mention>:") and len(line_parts) > 5:            
            is_identified_mention_part = line_parts[0]
            is_annotated_mention_part = line_parts[1]
            start_pos_part = line_parts[2]           
            identified_original_focus_mention_part = line_parts[3]
            annotated_original_focus_mention_part = line_parts[4]
            converted_focus_mention_part = line_parts[5]
            cluster_part = line_parts[len(line_parts)-5].strip()
            sentence_num_part = line_parts[len(line_parts)-4].strip()
            index_in_sentence_part=line_parts[len(line_parts)-3].strip()
            is_subject_part = line_parts[len(line_parts)-2].strip()
            is_object_part = line_parts[len(line_parts)-1].strip()
        elif line.startswith("<identified_mention>:") and len(line_parts) == 5:
            is_identified_mention_part = line_parts[0]
            is_annotated_mention_part = line_parts[1]
            start_pos_part = line_parts[2]
            annotated_original_focus_mention_part = line_parts[3]
            annotated_converted_focus_mention_part = line_parts[4]
        elif line.startswith("<identified_verb>:"):
            is_identified_verb_part = line_parts[0]
            is_annotated_verb_part = line_parts[1]
            start_pos_part = line_parts[2]
            verb_part_1 = line_parts[3]
            verb_part_2 = line_parts[4]
            if verb_part_1.find("<annotated_converted_verb>:") != -1:
                annotated_converted_verb_part = verb_part_1
            elif verb_part_1.find("<annotated_original_verb>:") != -1:
                annotated_original_verb_part = verb_part_1
            elif verb_part_1.find("<identified_original_verb>:") != -1:
                identified_original_verb_part = verb_part_1
            if verb_part_2.find("<identified_converted_verb>:") != -1:
                identified_converted_verb_part = verb_part_2
            elif verb_part_2.find("<annotated_converted_verb>:") != -1:
                annotated_converted_verb_part = verb_part_2
                
        if is_identified_mention_part == "" and is_annotated_mention_part == "" and is_identified_verb_part == "" and is_annotated_verb_part == "": 
            if original_focus_mention_part.startswith("<focus_mention>:"):
                temp_parts = original_focus_mention_part.split()
                for i in range(1,len(temp_parts)):
                    original_focus_mention = original_focus_mention + " " + temp_parts[i]
                original_focus_mention = original_focus_mention.strip()
                converted_focus_mention = original_focus_mention
            elif original_focus_mention_part.startswith("<original_focus_mention>:"):
                temp_parts = original_focus_mention_part.split()
                for i in range(1,len(temp_parts)):
                    original_focus_mention = original_focus_mention + " " + temp_parts[i]
                original_focus_mention = original_focus_mention.strip()
                temp_parts = converted_focus_mention_part.split()
                for i in range(1,len(temp_parts)):
                    converted_focus_mention = converted_focus_mention + " " + temp_parts[i]
                converted_focus_mention = converted_focus_mention.strip()
            else:
                print(original_focus_mention_part)
                print("There is an error in the dataset 9.")
             
            if sentence_num_part.startswith("<sentence_num>:"):
                temp_parts = sentence_num_part.split()
                sentence_num = int(temp_parts[1])
            else:
                print("There is an error in the dataset 15.")
            
            if cluster_part.startswith("<cluster_id>:"):
                temp_parts = cluster_part.split()
                cluster_id = int(temp_parts[1])
            else:
                print("There is an error in the dataset 16.")
         
            if is_subject_part.startswith("<is_subject>:"):
                temp_parts = is_subject_part.split()
                is_subject = int(temp_parts[1])
            else:
                print("There is an error in the dataset 17.")
            
            if is_object_part.startswith("<is_object>:"):
                temp_parts = is_object_part.split()
                is_object = int(temp_parts[1])
            else:
                print("There is an error in the dataset 18.")
        
            if index_in_sentence_part.startswith("<index_in_sentence>:"):
                temp_parts = index_in_sentence_part.split()
                index_in_sentence = int(temp_parts[1])
            else:
                print("There is an error in the dataset 19.")        
        elif (is_identified_mention_part != "" or is_annotated_mention_part != "") and len(line_parts) > 5:   
            if is_identified_mention_part.startswith("<identified_mention>:"):
                temp_parts = is_identified_mention_part.split()
                is_identified_mention = int(temp_parts[1])
            else:
                print(is_identified_mention_part)
                print("There is an error in the dataset 100.")
                
            if is_annotated_mention_part.startswith("<annotated_mention>:"):
                temp_parts = is_annotated_mention_part.split()
                is_annotated_mention = int(temp_parts[1])
            else:
                print(is_annotated_mention_part)
                print("There is an error in the dataset 101.")
             
            if start_pos_part.startswith("<start_pos>:"):
                temp_parts = start_pos_part.split()
                start_pos = int(temp_parts[1])
            else:
                print(start_pos_part)
                print("There is an error in the dataset 102.")
             
            if identified_original_focus_mention_part.startswith("<identified_original_focus_mention>:"):
                temp_parts = identified_original_focus_mention_part.split()
                for i in range(1,len(temp_parts)):
                    identified_original_focus_mention = identified_original_focus_mention + " " + temp_parts[i]
                identified_original_focus_mention = identified_original_focus_mention.strip()
            else:
                print(identified_original_focus_mention_part)
                print("There is an error in the dataset 103.")
            
            if annotated_original_focus_mention_part.startswith("<annotated_original_focus_mention>:"):
                temp_parts = annotated_original_focus_mention_part.split()
                for i in range(1,len(temp_parts)):
                    annotated_original_focus_mention = annotated_original_focus_mention + " " + temp_parts[i]
                annotated_original_focus_mention = annotated_original_focus_mention.strip()
            else:
                print(annotated_original_focus_mention_part)
                print("There is an error in the dataset 104.")
                
            if converted_focus_mention_part.startswith("<converted_focus_mention>:"):
                temp_parts = converted_focus_mention_part.split()
                for i in range(1,len(temp_parts)):
                    converted_focus_mention = converted_focus_mention + " " + temp_parts[i]
                converted_focus_mention = converted_focus_mention.strip()
            else:
                print(converted_focus_mention_part)
                print("There is an error in the dataset 9.")

            if sentence_num_part.startswith("<sentence_num>:"):
                temp_parts = sentence_num_part.split()
                sentence_num = int(temp_parts[1])
            else:
                print("There is an error in the dataset 15.")
            
            if cluster_part.startswith("<cluster_id>:"):
                temp_parts = cluster_part.split()
                cluster_id = int(temp_parts[1])
            else:
                print("There is an error in the dataset 16.")
         
            if is_subject_part.startswith("<is_subject>:"):
                temp_parts = is_subject_part.split()
                is_subject = int(temp_parts[1])
            else:
                print("There is an error in the dataset 17.")
            
            if is_object_part.startswith("<is_object>:"):
                temp_parts = is_object_part.split()
                is_object = int(temp_parts[1])
            else:
                print("There is an error in the dataset 18.")
        
            if index_in_sentence_part.startswith("<index_in_sentence>:"):
                temp_parts = index_in_sentence_part.split()
                index_in_sentence = int(temp_parts[1])
            else:
                print("There is an error in the dataset 19.") 
        elif (is_identified_mention_part != "" or is_annotated_mention_part != "") and len(line_parts) == 5:
            if is_identified_mention_part.startswith("<identified_mention>:"):
                temp_parts = is_identified_mention_part.split()
                is_identified_mention = int(temp_parts[1])
            else:
                print(is_identified_mention_part)
                print("There is an error in the dataset 100.")
                
            if is_annotated_mention_part.startswith("<annotated_mention>:"):
                temp_parts = is_annotated_mention_part.split()
                is_annotated_mention = int(temp_parts[1])
            else:
                print(is_annotated_mention_part)
                print("There is an error in the dataset 101.")
             
            if start_pos_part.startswith("<start_pos>:"):
                temp_parts = start_pos_part.split()
                start_pos = int(temp_parts[1])
            else:
                print(start_pos_part)
                print("There is an error in the dataset 102.")
             
            if annotated_original_focus_mention_part.startswith("<annotated_original_focus_mention>:"):
                temp_parts = annotated_original_focus_mention_part.split()
                for i in range(1,len(temp_parts)):
                    annotated_original_focus_mention = annotated_original_focus_mention + " " + temp_parts[i]
                annotated_original_focus_mention = annotated_original_focus_mention.strip()
            else:
                print(annotated_original_focus_mention_part)
                print("There is an error in the dataset 105.")
            
            if annotated_converted_focus_mention_part.startswith("<annotated_converted_focus_mention>:"):
                temp_parts = annotated_converted_focus_mention_part.split()
                for i in range(1,len(temp_parts)):
                    annotated_converted_focus_mention = annotated_converted_focus_mention + " " + temp_parts[i]
                annotated_converted_focus_mention = annotated_converted_focus_mention.strip()
            else:
                print(annotated_converted_focus_mention_part)
                print("There is an error in the dataset 106.")
        elif (is_identified_verb_part != "" or is_annotated_verb_part != ""):
            if is_identified_verb_part.startswith("<identified_verb>:"):
                temp_parts = is_identified_verb_part.split()
                is_identified_verb = int(temp_parts[1])
            else:
                print(is_identified_verb_part)
                print("There is an error in the dataset 107.")
                
            if is_annotated_verb_part.startswith("<annotated_verb>:"):
                temp_parts = is_annotated_verb_part.split()
                is_annotated_verb = int(temp_parts[1])
            else:
                print(is_annotated_verb_part)
                print("There is an error in the dataset 108.")
             
            if start_pos_part.startswith("<start_pos>:"):
                temp_parts = start_pos_part.split()
                start_pos = int(temp_parts[1])
            else:
                print(start_pos_part)
                print("There is an error in the dataset 102.")
             
            if identified_original_verb_part.startswith("<identified_original_verb>:"):
                temp_parts = identified_original_verb_part.split()
                for i in range(1,len(temp_parts)):
                    identified_original_verb = identified_original_verb + " " + temp_parts[i]
                identified_original_verb = identified_original_verb.strip()
            
            if annotated_original_verb_part.startswith("<annotated_original_verb>:"):
                temp_parts = annotated_original_verb_part.split()
                for i in range(1,len(temp_parts)):
                    annotated_original_verb = annotated_original_verb + " " + temp_parts[i]
                annotated_original_verb = annotated_original_verb.strip()
           
            if identified_converted_verb_part.startswith("<identified_converted_verb>:"):
                temp_parts = identified_converted_verb_part.split()
                for i in range(1,len(temp_parts)):
                    identified_converted_verb = identified_converted_verb + " " + temp_parts[i]
                identified_converted_verb = identified_converted_verb.strip()
             
            if annotated_converted_verb_part.startswith("<annotated_converted_verb>:"):
                temp_parts = annotated_converted_verb_part.split()
                for i in range(1,len(temp_parts)):
                    annotated_converted_verb = annotated_converted_verb + " " + temp_parts[i]
                annotated_converted_verb = annotated_converted_verb.strip()
        
        return is_identified_mention, is_annotated_mention, is_identified_verb, is_annotated_verb, start_pos, identified_original_focus_mention, annotated_original_focus_mention, identified_converted_focus_mention, annotated_converted_focus_mention, identified_original_verb, annotated_original_verb, identified_converted_verb, annotated_converted_verb, original_focus_mention, converted_focus_mention, sentence_num, cluster_id, index_in_sentence, is_subject, is_object
    
    def get_distance_info_from_line(self, line):
        mention_to_pre_mention_distance = {}
        mention_to_mentions_in_between = {}
        
        line = line.strip()
        line_parts = line.split("\t")
        
        for i in range(int(len(line_parts) / 3)):
            mention_part = line_parts[i*3].strip()
            pre_mention_distance_part = line_parts[1+i*3].strip()
            mentions_in_between_part = line_parts[2+i*3].strip()
            
            mention = ""
            pre_mention_distance = -1
            mentions_in_between = -1
            if mention_part.startswith("<mention>:"):
                mention = mention_part[11:]
            else:
                print("There is an error in the dataset 20.")
                
            if pre_mention_distance_part.startswith("<distance_to_pre_mention>:"):
                temp_parts = pre_mention_distance_part.split()
                pre_mention_distance = int(temp_parts[1])
            else:
                print("There is an error in the dataset 21.")
                
            if mentions_in_between_part.startswith("<mentions_in_between>:"):
                temp_parts = mentions_in_between_part.split()
                mentions_in_between = int(temp_parts[1])
            else:
                print("There is an error in the dataset 22.")
                
            mention_to_pre_mention_distance[mention] = pre_mention_distance
            mention_to_mentions_in_between[mention] = mentions_in_between
            
        return mention_to_pre_mention_distance, mention_to_mentions_in_between
    
    def get_related_mention_from_line(self, line):
        related_mention = []
        cluster_id = -1
    
        line = line.strip()
        line_parts = line.split("\t")
        related_mention_part = line_parts[0].strip()
        cluster_part = line_parts[1].strip()
        original_related_mention_text = ""
        converted_related_mention_text = ""
        if related_mention_part.startswith("<related_mention>:"):
            temp_parts = related_mention_part.split()
            start_pos = int(temp_parts[1])
            end_pos= int(temp_parts[2])
            related_mention.append(start_pos)
            related_mention.append(end_pos)
        else:
            print("There is an error in the dataset 23.")
            
        if cluster_part.startswith("<cluster_id>:"):
            temp_parts = cluster_part.split()
            cluster_id = int(temp_parts[1])
        else:
            print("There is an error in the dataset 24.")
         
        if len(line_parts) > 2:
            original_related_mention_text_part = line_parts[2].strip()
            converted_related_mention_text_part = line_parts[3].strip()
            if original_related_mention_text_part.startswith("<original_related_mention_text>:"):
                temp_parts = original_related_mention_text_part.split()
                for i in range(1, len(temp_parts)):
                    ttt = temp_parts[i].lower()
                    original_related_mention_text = original_related_mention_text + " " + ttt
                original_related_mention_text = original_related_mention_text.strip()
            else:
                print("There is an error in the dataset 25.")
            
            if converted_related_mention_text_part.startswith("<converted_related_mention_text>:"):
                temp_parts = converted_related_mention_text_part.split()
                for i in range(1, len(temp_parts)):
                    ttt = temp_parts[i].lower()
                    converted_related_mention_text = converted_related_mention_text + " " + ttt
                converted_related_mention_text = converted_related_mention_text.strip()
            else:
                print("There is an error in the dataset 26.")
                
        return [related_mention, cluster_id, original_related_mention_text, converted_related_mention_text]
    
    def get_raw_sequence_from_line(self, line):
        raw_sequence = []
        line = line.strip()
        line_parts = line.split()
        if line.startswith("<raw_sequence>:"):
            for i in range(1, len(line_parts)):
                raw_sequence.append(line_parts[i])
        else:
            print("There is an error in the dataset 27.")
            
        return raw_sequence
    
    def get_postag_sequence_from_line(self, line):
        postag_sequence = []
        line = line.strip()
        line_parts = line.split()
        if line.startswith("<postags>:"):
            for i in range(1, len(line_parts)):
                postag_sequence.append(line_parts[i])
        else:
            print("There is an error in the dataset 28.")
            
        return postag_sequence
    
    def get_related_sentence_from_line(self, line):
        sentence_num = -1
        sentence_text = ""
        
        line = line.strip()
        line_parts = line.split("\t")
        sentence_num_part = line_parts[0].strip()
        sentence_text_part = line_parts[1].strip()
        
        if sentence_num_part.startswith("<sentence_num>:"):
            temp_parts = sentence_num_part.split()
            sentence_num = int(temp_parts[1])
        else:
            print("There is an error in the dataset 29.")
            
        if sentence_text_part.startswith("<sentence_text>:"):
            temp_parts = sentence_text_part.split()
            for i in range(1, len(temp_parts)):
                sentence_text = sentence_text + " " + temp_parts[i]
            sentence_text = sentence_text.strip()
        else:
            print("There is an error in the dataset 30.")
            
        return [sentence_num, sentence_text]
    
    def get_token_index_info_from_line(self, line):
        start_token_index_in_sentence= -1
        end_token_index_in_sentence = -1
        
        line = line.strip()
        line_parts = line.split("\t")
        start_token_index_part = line_parts[0].strip()
        end_token_index_part = line_parts[1].strip()
        
        if start_token_index_part.startswith("<start_token_index_in_sentence>:"):
            temp_parts = start_token_index_part.split()
            start_token_index_in_sentence = int(temp_parts[1])
        else:
            print("There is an error in the dataset 31.")
            
        if end_token_index_part.startswith("<end_token_index_in_sentence>:"):
            temp_parts = end_token_index_part.split()
            end_token_index_in_sentence = int(temp_parts[1])
        else:
            print("There is an error in the dataset 32.")
            
        return [start_token_index_in_sentence, end_token_index_in_sentence]
    
    def get_original_pre_mention_sequence_from_line(self, line):
        mention_sequence = []
        line = line.strip()
        line_parts = line.split("\t")
        if line.startswith("<pre_mention_sequence>:"):
            for i in range(len(line_parts)):
                line_part = line_parts[i]
                if i == 0:
                    line_part = line_part[24:]
                mention_sequence.append(line_part)
        elif line.startswith("<original_pre_mention_sequence>:"):
            for i in range(len(line_parts)):
                line_part = line_parts[i]
                if i == 0:
                    line_part = line_part[33:]
                mention_sequence.append(line_part)
        else:
            print("There is an error in the dataset 33.")
            
        return mention_sequence
    
    def get_converted_pre_mention_sequence_from_line(self, line):
        mention_sequence = []
        line = line.strip()
        line_parts = line.split("\t")
        if line.startswith("<converted_pre_mention_sequence>:"):
            for i in range(len(line_parts)):
                line_part = line_parts[i]
                if i == 0:
                    line_part = line_part[34:]
                mention_sequence.append(line_part)
        else:
            print("There is an error in the dataset 34.")
            
        return mention_sequence
    
    def get_pre_mention_info_from_line(self, line):
        mention_info = None
        line = line.strip()
        line_parts = line.split("\t")
        pre_mention_text_part = line_parts[0].strip()
        pre_mention_index_in_sentence_part = line_parts[1].strip()
        pre_mention_in_sentence_part = line_parts[2].strip()
        pre_mention_sentence_num_part = line_parts[3].strip()
        
        pre_mention_text = ""
        pre_mention_index_in_sentence = -1
        pre_mention_in_sentence = ""
        pre_mention_sentence_num = -1
        
        if pre_mention_text_part.startswith("<pre_mention_text>:"):
            temp_parts = pre_mention_text_part.split()
            for i in range(1, len(temp_parts)):
                pre_mention_text = pre_mention_text + " " + temp_parts[i]
            pre_mention_text = pre_mention_text.strip()
        else:
            print("There is an error in the dataset 35.")
        
        if pre_mention_index_in_sentence_part.startswith("<pre_mention_index_in_sentence>:"):
            temp_parts = pre_mention_index_in_sentence_part.split()
            pre_mention_index_in_sentence = int(temp_parts[1])
        else:
            print("There is an error in the dataset 36.")
            
        if pre_mention_in_sentence_part.startswith("<pre_mention_in_sentence>:"):
            temp_parts = pre_mention_in_sentence_part.split()
            for i in range(1, len(temp_parts)):
                pre_mention_in_sentence = pre_mention_in_sentence + " " + temp_parts[i]
            pre_mention_in_sentence = pre_mention_in_sentence.strip()
        else:
            print("There is an error in the dataset 37.")
          
        if pre_mention_sentence_num_part.startswith("<pre_mention_sentence_num>:"):
            temp_parts = pre_mention_sentence_num_part.split()
            pre_mention_sentence_num = int(temp_parts[1])
        else:
            print("There is an error in the dataset 38.")
            
        mention_info = MentionInfo(pre_mention_text, pre_mention_index_in_sentence, pre_mention_in_sentence, pre_mention_sentence_num)
        
        return mention_info
    
    def get_pre_mention_cluster_id_sequence_from_line(self, line):
        mention_cluster_id_sequence = []
        line = line.strip()
        line_parts = line.split("\t")
        if line.startswith("<pre_mention_cluster_id_sequence>:"):
            for i in range(len(line_parts)):
                line_part = line_parts[i]
                if i == 0:
                    line_part = line_part[35:]
                mention_cluster_id_sequence.append(int(line_part))
        else:
            print("There is an error in the dataset 39.")
        return mention_cluster_id_sequence
    
    def get_pre_mention_distance_sequence_from_line(self, line):
        mention_distance_sequence = []
        line = line.strip()
        line_parts = line.split("\t")
        if line.startswith("<pre_mention_distance_sequence>:"):
            for i in range(len(line_parts)):
                line_part = line_parts[i]
                if i == 0:
                    line_part = line_part[33:]
                mention_distance_sequence.append(int(line_part))
        else:
            print("There is an error in the dataset 40.")
        return mention_distance_sequence
    
    def get_original_post_mention_sequence_from_line(self, line):
        mention_sequence = []
        line = line.strip()
        line_parts = line.split("\t")
        if line.startswith("<post_mention_sequence>:"):
            for i in range(len(line_parts)):
                line_part = line_parts[i]
                if i == 0:
                    line_part = line_part[25:]
                mention_sequence.append(line_part)
        elif line.startswith("<original_post_mention_sequence>:"):
            for i in range(len(line_parts)):
                line_part = line_parts[i]
                if i == 0:
                    line_part = line_part[34:]
                mention_sequence.append(line_part)
        else:
            print("There is an error in the dataset 41.")
            
        return mention_sequence
    
    def get_converted_post_mention_sequence_from_line(self, line):
        mention_sequence = []
        line = line.strip()
        line_parts = line.split("\t")
        if line.startswith("<converted_post_mention_sequence>:"):
            for i in range(len(line_parts)):
                line_part = line_parts[i]
                if i == 0:
                    line_part = line_part[35:]
                mention_sequence.append(line_part)
        else:
            print("There is an error in the dataset 42.")
            
        return mention_sequence
    
    def get_post_mention_info_from_line(self, line):
        mention_info = None
        line = line.strip()
        line_parts = line.split("\t")
        post_mention_text_part = line_parts[0].strip()
        post_mention_index_in_sentence_part = line_parts[1].strip()
        post_mention_in_sentence_part = line_parts[2].strip()
        post_mention_sentence_num_part = line_parts[3].strip()
        
        post_mention_text = ""
        post_mention_index_in_sentence = -1
        post_mention_in_sentence = ""
        post_mention_sentence_num = -1
        
        if post_mention_text_part.startswith("<post_mention_text>:"):
            temp_parts = post_mention_text_part.split()
            for i in range(1, len(temp_parts)):
                post_mention_text = post_mention_text + " " + temp_parts[i]
            post_mention_text = post_mention_text.strip()
        else:
            print("There is an error in the dataset 43.")
        
        if post_mention_index_in_sentence_part.startswith("<post_mention_index_in_sentence>:"):
            temp_parts = post_mention_index_in_sentence_part.split()
            post_mention_index_in_sentence = int(temp_parts[1])
        else:
            print("There is an error in the dataset 44.")
            
        if post_mention_in_sentence_part.startswith("<post_mention_in_sentence>:"):
            temp_parts = post_mention_in_sentence_part.split()
            for i in range(1, len(temp_parts)):
                post_mention_in_sentence = post_mention_in_sentence + " " + temp_parts[i]
            post_mention_in_sentence = post_mention_in_sentence.strip()
        else:
            print("There is an error in the dataset 45.")
        
        if post_mention_sentence_num_part.startswith("<post_mention_sentence_num>:"):
            temp_parts = post_mention_sentence_num_part.split()
            post_mention_sentence_num = int(temp_parts[1])
        else:
            print("There is an error in the dataset 46.")
            
        mention_info = MentionInfo(post_mention_text, post_mention_index_in_sentence, post_mention_in_sentence, post_mention_sentence_num)
            
        return mention_info
    
    def get_post_mention_cluster_id_sequence_from_line(self, line):
        mention_cluster_id_sequence = []
        line = line.strip()
        line_parts = line.split("\t")
        if line.startswith("<post_mention_cluster_id_sequence>:"):
            for i in range(len(line_parts)):
                line_part = line_parts[i]
                if i == 0:
                    line_part = line_part[36:]
                mention_cluster_id_sequence.append(int(line_part))
        else:
            print("There is an error in the dataset 47.")
        return mention_cluster_id_sequence
    
    def get_post_mention_distance_sequence_from_line(self, line):
        mention_distance_sequence = []
        line = line.strip()
        line_parts = line.split("\t")
        if line.startswith("<post_mention_distance_sequence>:"):
            for i in range(len(line_parts)):
                line_part = line_parts[i]
                if i == 0:
                    line_part = line_part[34:]
                mention_distance_sequence.append(int(line_part))
        else:
            print("There is an error in the dataset 48.")
        return mention_distance_sequence
    
    def read_file(self, file_path):
        self.raw_document_data_list = []
        
        f = open(file_path, 'r', encoding='utf-8')
        lines = f.readlines()

        start_of_document_text = False
        end_of_document_text = True
        first_line = True
        end_of_cluster = False
        encounter_valid_invalid_cluster = False
        
        document_text = ""
        is_identified_mention = -1
        is_annotated_mention = -1
        is_identified_verb = -1
        is_annotated_verb = -1
        start_pos = -1
        identified_original_focus_mention = ""
        annotated_original_focus_mention = ""
        identified_converted_focus_mention = ""
        annotated_converted_focus_mention = ""
        identified_original_verb = ""
        annotated_original_verb = ""
        identified_converted_verb = ""
        annotated_converted_verb = ""
        original_focus_mention = ""
        converted_focus_mention = ""
        focus_mention_cluster_id = -1
        focus_mention_sentence_num = -1
        focus_mention_index_in_sentence = -1
        is_subject = -1
        is_object = -1
        related_mentions = []
        related_original_mention_text_list = []
        related_converted_mention_text_list = []
        related_mention_cluster_ids = []
        raw_sequence = []
        postag_sequence = []
        related_sentence_num_to_sentence_text = {}
        start_token_index_in_sentence = -1
        end_token_index_in_sentence = -1
        original_pre_mention_sequence = []
        converted_pre_mention_sequence = []
        pre_mention_info_list = []
        pre_mention_cluster_id_sequence = []
        pre_mention_distance_sequence = []
        original_post_mention_sequence = []
        converted_post_mention_sequence = []
        post_mention_info_list = []
        post_mention_cluster_id_sequence = []
        post_mention_distance_sequence = []
        
        gold_cluster_id_to_cluster_data = {}
        auto_cluster_id_to_cluster_data = {}
        gold_invalid_cluster_id_to_cluster_data = {}
        auto_invalid_cluster_id_to_cluster_data = {}
        raw_sequence_data_list = []
        
        previous_line = ""
        for line in lines:
            if first_line and line.startswith("<document_text>"):
                first_line = False
                start_of_document_text = True
                end_of_document_text = False
            elif line.startswith("<document_text>"):
                start_of_document_text = False
                end_of_document_text = True
            elif line.startswith("<cluster_id>:") or line.startswith("<gold_cluster_id>:") or line.startswith("<auto_cluster_id>:"):
                # Start of another document
                if end_of_cluster:
                    raw_document_data = RawDocumentData(document_text, gold_cluster_id_to_cluster_data, auto_cluster_id_to_cluster_data, gold_invalid_cluster_id_to_cluster_data, auto_invalid_cluster_id_to_cluster_data, raw_sequence_data_list)
                    self.raw_document_data_list.append(raw_document_data)
                    document_text = ""
                    gold_cluster_id_to_cluster_data = {}
                    auto_cluster_id_to_cluster_data = {}
                    gold_invalid_cluster_id_to_cluster_data = {}
                    auto_invalid_cluster_id_to_cluster_data = {}
                    raw_sequence_data_list = []
                raw_cluster_data = self.get_cluster_data_from_line(line)
                if line.startswith("<cluster_id>") or line.startswith("<gold_cluster_id>:"):                    
                    gold_cluster_id_to_cluster_data[raw_cluster_data._cluster_id] = raw_cluster_data
                else:
                    auto_cluster_id_to_cluster_data[raw_cluster_data._cluster_id]=raw_cluster_data
                end_of_cluster = False
                encounter_valid_invalid_cluster = False
            elif line.startswith("<invalid_cluster_id>:") or line.startswith("<gold_invalid_cluster_id>:") or line.startswith("<auto_invalid_cluster_id>:"):
                if previous_line.startswith("<cluster_id>:") or previous_line.startswith("<gold_cluster_id>:") or previous_line.startswith("<auto_cluster_id>:"):
                    encounter_valid_invalid_cluster = True
                if encounter_valid_invalid_cluster:
                    raw_cluster_data = self.get_invalid_cluster_data_from_line(line)
                    if line.startswith("<invalid_cluster_id>:") or line.startswith("<gold_invalid_cluster_id>:"):
                        gold_invalid_cluster_id_to_cluster_data[raw_cluster_data._cluster_id] = raw_cluster_data
                    else:
                        auto_invalid_cluster_id_to_cluster_data[raw_cluster_data._cluster_id]=raw_cluster_data
            elif line.startswith("<focus_mention>:") or line.startswith("<original_focus_mention>:") or line.startswith("<identified_mention>:") or line.startswith("<identified_verb>:"):
                is_identified_mention, is_annotated_mention, is_identified_verb, is_annotated_verb, start_pos, identified_original_focus_mention, annotated_original_focus_mention, identified_converted_focus_mention, annotated_converted_focus_mention, identified_original_verb, annotated_original_verb, identified_converted_verb, annotated_converted_verb, original_focus_mention, converted_focus_mention, focus_mention_sentence_num, focus_mention_cluster_id, focus_mention_index_in_sentence, is_subject, is_object = self.get_focus_mention_from_line(line)
                if not end_of_cluster:
                    end_of_cluster = True
                encounter_valid_invalid_cluster = False
                if is_identified_mention == 0 or is_identified_verb == 1 or is_annotated_verb == 1:
                    raw_sequence_data = RawSequenceData(is_identified_mention, is_annotated_mention, is_identified_verb, is_annotated_verb, start_pos, identified_original_focus_mention, annotated_original_focus_mention, identified_converted_focus_mention, annotated_converted_focus_mention, identified_original_verb, annotated_original_verb, identified_converted_verb, annotated_converted_verb, original_focus_mention, converted_focus_mention, focus_mention_cluster_id, focus_mention_sentence_num, focus_mention_index_in_sentence, is_subject, is_object, related_mentions, related_mention_cluster_ids, related_original_mention_text_list, related_converted_mention_text_list, raw_sequence, postag_sequence, related_sentence_num_to_sentence_text, start_token_index_in_sentence, end_token_index_in_sentence, original_pre_mention_sequence, converted_pre_mention_sequence, pre_mention_cluster_id_sequence, pre_mention_distance_sequence, pre_mention_info_list, original_post_mention_sequence, converted_post_mention_sequence, post_mention_cluster_id_sequence, post_mention_distance_sequence, post_mention_info_list)
                    raw_sequence_data_list.append(raw_sequence_data)
                    is_identified_mention = -1
                    is_annotated_mention = -1
                    is_identified_verb = -1
                    is_annotated_verb = -1
                    start_pos = -1
                    identified_original_focus_mention = ""
                    annotated_original_focus_mention = ""
                    identified_converted_focus_mention = ""
                    annotated_converted_focus_mention = ""
                    identified_original_verb = ""
                    annotated_original_verb = ""
                    identified_converted_verb = ""
                    annotated_converted_verb = ""
                    original_focus_mention = ""
                    converted_focus_mention = ""
                    focus_mention_cluster_id = -1
                    focus_mention_sentence_num = -1
                    focus_mention_index_in_sentence = -1
                    is_subject = -1
                    is_object = -1
                    related_mentions = []
                    related_mention_cluster_ids = []
                    related_original_mention_text_list = []
                    related_converted_mention_text_list = []
                    raw_sequence = []
                    postag_sequence = []
                    related_sentence_num_to_sentence_text = {}
                    start_token_index_in_sentence = -1
                    end_token_index_in_sentence = -1
                    original_pre_mention_sequence= []
                    converted_pre_mention_sequence = []
                    pre_mention_cluster_id_sequence = []
                    pre_mention_distance_sequence = []
                    pre_mention_info_list = []
                    original_post_mention_sequence = []
                    converted_post_mention_sequence = []
                    post_mention_cluster_id_sequence= []
                    post_mention_distance_sequence = []
                    post_mention_info_list = []
            elif line.startswith("<no related mentions>"):
                continue
            elif line.startswith("<related_mention>:"):
                result = self.get_related_mention_from_line(line)
                related_mention = result[0]
                related_mention_cluster_id = result[1]
                original_related_mention_text = result[2]
                converted_related_mention_text = result[3]
                related_mentions.append(related_mention)
                related_mention_cluster_ids.append(related_mention_cluster_id)
                related_original_mention_text_list.append(original_related_mention_text)
                related_converted_mention_text_list.append(converted_related_mention_text)
            elif line.startswith("<raw_sequence>:"):
                raw_sequence = self.get_raw_sequence_from_line(line)
            elif line.startswith("<postags>:"):
                postag_sequence = self.get_postag_sequence_from_line(line)       
            elif line.startswith("<sentence_num>:"):
                sentence_result = self.get_related_sentence_from_line(line)
                related_sentence_num_to_sentence_text[sentence_result[0]] = sentence_result[1]
            elif line.startswith("<start_token_index_in_sentence>:"):
                token_index_result = self.get_token_index_info_from_line(line)
                start_token_index_in_sentence = token_index_result[0]
                end_token_index_in_sentence = token_index_result[1]
            elif line.startswith("<pre_mention_sequence>:") or line.startswith("<original_pre_mention_sequence>:"):
                original_pre_mention_sequence = self.get_original_pre_mention_sequence_from_line(line)
            elif line.startswith("<converted_pre_mention_sequence>:"):
                converted_pre_mention_sequence = self.get_converted_pre_mention_sequence_from_line(line)
            elif line.startswith("<pre_mention_text>:"):
                pre_mention_info = self.get_pre_mention_info_from_line(line)
                pre_mention_info_list.append(pre_mention_info)
            elif line.startswith("<pre_mention_cluster_id_sequence>:"):
                pre_mention_cluster_id_sequence = self.get_pre_mention_cluster_id_sequence_from_line(line)
            elif line.startswith("<pre_mention_distance_sequence>:"):
                pre_mention_distance_sequence = self.get_pre_mention_distance_sequence_from_line(line)
            elif line.startswith("<post_mention_sequence>:") or line.startswith("<original_post_mention_sequence>:"):
                original_post_mention_sequence = self.get_original_post_mention_sequence_from_line(line)
            elif line.startswith("<converted_post_mention_sequence>:"):
                converted_post_mention_sequence = self.get_converted_post_mention_sequence_from_line(line)
            elif line.startswith("<post_mention_text>:"):
                post_mention_info = self.get_post_mention_info_from_line(line)
                post_mention_info_list.append(post_mention_info)
            elif line.startswith("<post_mention_cluster_id_sequence>:"):
                post_mention_cluster_id_sequence = self.get_post_mention_cluster_id_sequence_from_line(line)
            elif line.startswith("<post_mention_distance_sequence>:"):
                post_mention_distance_sequence = self.get_post_mention_distance_sequence_from_line(line)
                raw_sequence_data = RawSequenceData(is_identified_mention, is_annotated_mention, is_identified_verb, is_annotated_verb, start_pos, identified_original_focus_mention, annotated_original_focus_mention, identified_converted_focus_mention, annotated_converted_focus_mention, identified_original_verb, annotated_original_verb, identified_converted_verb, annotated_converted_verb, original_focus_mention, converted_focus_mention, focus_mention_cluster_id, focus_mention_sentence_num, focus_mention_index_in_sentence, is_subject, is_object, related_mentions, related_mention_cluster_ids, related_original_mention_text_list, related_converted_mention_text_list, raw_sequence, postag_sequence, related_sentence_num_to_sentence_text, start_token_index_in_sentence, end_token_index_in_sentence, original_pre_mention_sequence, converted_pre_mention_sequence, pre_mention_cluster_id_sequence, pre_mention_distance_sequence, pre_mention_info_list, original_post_mention_sequence, converted_post_mention_sequence, post_mention_cluster_id_sequence, post_mention_distance_sequence, post_mention_info_list)
                raw_sequence_data_list.append(raw_sequence_data)
                is_identified_mention = -1
                is_annotated_mention = -1
                is_identified_verb = -1
                is_annotated_verb = -1
                start_pos = -1
                identified_original_focus_mention = ""
                annotated_original_focus_mention = ""
                identified_converted_focus_mention = ""
                annotated_converted_focus_mention = ""
                identified_original_verb = ""
                annotated_original_verb = ""
                identified_converted_verb = ""
                annotated_converted_verb = ""
                original_focus_mention = ""
                converted_focus_mention = ""
                focus_mention_cluster_id = -1
                focus_mention_sentence_num = -1
                focus_mention_index_in_sentence = -1
                is_subject = -1
                is_object = -1
                related_mentions = []
                related_mention_cluster_ids = []
                related_original_mention_text_list = []
                related_converted_mention_text_list = []
                raw_sequence = []
                postag_sequence = []
                related_sentence_num_to_sentence_text = {}
                start_token_index_in_sentence = -1
                end_token_index_in_sentence = -1
                original_pre_mention_sequence= []
                converted_pre_mention_sequence = []
                pre_mention_cluster_id_sequence = []
                pre_mention_distance_sequence = []
                pre_mention_info_list = []
                original_post_mention_sequence = []
                converted_post_mention_sequence = []
                post_mention_cluster_id_sequence= []
                post_mention_distance_sequence = []
                post_mention_info_list = []
            elif start_of_document_text and (not end_of_document_text):
                document_text = document_text + line
            else:
                print("there is an error.")
                print(line)
            previous_line = line
            
        if len(gold_cluster_id_to_cluster_data) > 0 and len(raw_sequence_data_list) > 0:
            raw_document_data = RawDocumentData(document_text, gold_cluster_id_to_cluster_data, auto_cluster_id_to_cluster_data, gold_invalid_cluster_id_to_cluster_data, auto_invalid_cluster_id_to_cluster_data, raw_sequence_data_list)
            self.raw_document_data_list.append(raw_document_data)
                
        f.close()
    
    def list_to_array(self, seq_list, max_len, dim):
        selist = []
        length = []
        for one in seq_list:
            length.append(len(one))
            if len(one) < max_len:
                if dim == 2:
                    for i in range(max_len - len(one)):
                        temp = [0] * 768
                        one.append(temp)
                else:
                    one.extend(list(numpy.zeros(max_len - len(one), 
                                            dtype=numpy.float32)))
            selist.append(one)

        selist = numpy.asarray(selist, dtype=numpy.float32)
        length = numpy.asarray(length, dtype=numpy.int32)

        return selist, length
    
    def create_padding(self, data_set, max_seqr_length):
        seqr_v, seqr_l = self.list_to_array(data_set[0], max_seqr_length, 2)
        return seqr_v, seqr_l, numpy.asarray(data_set[1], dtype=numpy.float32), numpy.asarray(data_set[2], dtype=numpy.float32), numpy.asarray(data_set[3], dtype=numpy.float32), numpy.asarray(data_set[4], dtype=numpy.float32), numpy.asarray(data_set[5], dtype=numpy.float32), numpy.asarray(data_set[6], dtype=numpy.float32), numpy.asarray(data_set[7], dtype=numpy.float32), numpy.asarray(data_set[8], dtype=numpy.float32), numpy.asarray(data_set[9], dtype=numpy.float32)

    def get_similar_clusters(self, cluster_id_to_cluster_data):
        cluster_id_to_he_clusters = {}
        cluster_id_to_she_clusters = {}
        cluster_id_to_it_clusters = {}
        cluster_id_to_they_clusters = {}
        
        for cluster_id, cluster in cluster_id_to_cluster_data.items():
            found_cluster = False
            for mention in cluster._mention_list:
                updated_mention = mention.lower()
                if updated_mention == "he" or updated_mention == "him" or updated_mention == "his" or updated_mention == "himself":
                    cluster_id_to_he_clusters[cluster_id] = cluster
                    found_cluster = True
                    break
                elif updated_mention == "she" or updated_mention == "her" or updated_mention == "herself":
                    cluster_id_to_she_clusters[cluster_id] = cluster
                    found_cluster = True
                    break
                elif updated_mention == "it" or updated_mention == "its" or updated_mention == "itself":
                    cluster_id_to_it_clusters[cluster_id] = cluster
                    found_cluster = True
                    break
                elif updated_mention == "they" or updated_mention == "them" or updated_mention == "their" or updated_mention == "themselves":
                    cluster_id_to_they_clusters[cluster_id] = cluster
                    found_cluster = True
                    break
            if not found_cluster:
                cluster_id_to_he_clusters[cluster_id] = cluster
                
        return[cluster_id_to_he_clusters, cluster_id_to_she_clusters, cluster_id_to_it_clusters, cluster_id_to_they_clusters]
    
    def get_mention_distance_embedding(self, mention_length):
        rep_distance_list = []
        for x in mention_length:
            temp = []
            if x == 1:
                temp = [1, 0, 0, 0, 0, 0]
            elif x == 2:
                temp = [0, 1, 0, 0, 0, 0]
            elif x == 3:
                temp = [0, 0, 1, 0, 0, 0]
            elif x == 4:
                temp = [0, 0, 0, 1, 0, 0]
            elif x == 5:
                temp = [0, 0, 0, 0, 1, 0]
            else:
                temp = [0, 0, 0, 0, 0, 1]
            rep_distance_list.append(temp)
        
        rep_distance_array = numpy.asarray(rep_distance_list, dtype=numpy.float32)
        return rep_distance_array
    
    def get_coreference_chain_distance_embedding(self, coreference_chain_length):
        rep_distance_list = []
        for x in coreference_chain_length:
            temp = []
            if x <= 5:
                temp = [1, 0, 0, 0, 0, 0]
            elif 5< x and x <= 10:
                temp = [0, 1, 0, 0, 0, 0]
            elif 10 < x and x <= 15:
                temp = [0, 0, 1, 0, 0, 0]
            elif 15 < x and x <= 20:
                temp = [0, 0, 0, 1, 0, 0]
            elif 20 < x and x <= 25:
                temp = [0, 0, 0, 0, 1, 0]
            else:
                temp = [0, 0, 0, 0, 0, 1]
            rep_distance_list.append(temp)
        
        rep_distance_array = numpy.asarray(rep_distance_list, dtype=numpy.float32)
        return rep_distance_array
    
    def get_updated_sentence(self, sentence_text, index_in_sentence, mention, original_mention_len):
        updated_sentence_text = ""
        
        sentence_text_parts = sentence_text.split()
        mention_parts = mention.split()
        
        for i in range(index_in_sentence):
            updated_sentence_text = updated_sentence_text + " " + sentence_text_parts[i]
            
        for i in range(len(mention_parts)):
            updated_sentence_text = updated_sentence_text + " " + mention_parts[i]
            
        for i in range(index_in_sentence + original_mention_len, len(sentence_text_parts)):
            updated_sentence_text = updated_sentence_text + " " + sentence_text_parts[i]
            
        updated_sentence_text = updated_sentence_text.strip()
        
        return updated_sentence_text
    
    def get_focus_mention_offset(self, index_in_sentence, mention, original_mention, focus_mention_index_in_sentence):
        mention_parts = mention.split()
        original_mention_parts = original_mention.split()
        offset = 0
        
        if focus_mention_index_in_sentence > index_in_sentence:
            offset = len(mention_parts) - len(original_mention_parts)
            
        return offset
    
    def get_end_token_offset(self, token_index_to_sentence_num, sentence_num, related_mentions):
        offset = 0
        
        for related_mention in related_mentions:
            start_pos = related_mention[0]
            end_pos = related_mention[1] 
            mention_sentence_num = token_index_to_sentence_num[start_pos]
            
            if mention_sentence_num != sentence_num:
                continue
            
            offset = offset + start_pos - end_pos
            
        return offset
    
    def get_embedding_sequence_from_left(self, sorted_sentence_num_list, sentence_num_to_sentence_embedding, start_token_index_in_sentence, focus_mention_sentence_num, focus_mention_index_in_sentence, focus_mention_parts, from_sentence=-1):
        result_sequence = []
        if from_sentence == -1:
            from_sentence = sorted_sentence_num_list[0]
        for sentence_num in sorted_sentence_num_list:
            sentence_embedding = sentence_num_to_sentence_embedding[sentence_num]
            if sentence_num < from_sentence:
                continue
            if sentence_num == from_sentence and sentence_num != focus_mention_sentence_num:
                for i in range(start_token_index_in_sentence, len(sentence_embedding)):
                    result_sequence.append(sentence_embedding[i])
            elif sentence_num == focus_mention_sentence_num and sentence_num != from_sentence:
                for i in range(focus_mention_index_in_sentence + len(focus_mention_parts)):
                    result_sequence.append(sentence_embedding[i])
                break
            elif from_sentence == sentence_num and sentence_num == focus_mention_sentence_num:
                for i in range(start_token_index_in_sentence, focus_mention_index_in_sentence + len(focus_mention_parts)):
                    result_sequence.append(sentence_embedding[i])
                break
            else:
                for i in range(len(sentence_embedding)):
                    result_sequence.append(sentence_embedding[i])
            
        return result_sequence
     
    def get_embedding_sequence_from_right(self, sorted_sentence_num_list, sentence_num_to_sentence_embedding, end_token_index_in_sentence, focus_mention_sentence_num, focus_mention_index_in_sentence, focus_mention_parts, to_sentence = -1):
        result_sequence = []
        
        if to_sentence == -1:
            to_sentence = sorted_sentence_num_list[len(sorted_sentence_num_list)-1]
            
        for i in range(len(sorted_sentence_num_list)-1, -1, -1):             
            sentence_num = sorted_sentence_num_list[i]
            sentence_embedding = sentence_num_to_sentence_embedding[sentence_num]
            if sentence_num > to_sentence:
                continue
            if sentence_num == to_sentence and sentence_num != focus_mention_sentence_num:
                for j in range(end_token_index_in_sentence, -1, -1):
                    result_sequence.append(sentence_embedding[j])
            elif sentence_num == focus_mention_sentence_num and sentence_num != to_sentence:
                for j in range(len(sentence_embedding)-1, focus_mention_index_in_sentence+len(focus_mention_parts)-1, -1):
                    result_sequence.append(sentence_embedding[j])
                break           
            elif sentence_num == to_sentence and sentence_num == focus_mention_sentence_num:
                for j in range(end_token_index_in_sentence, focus_mention_index_in_sentence+len(focus_mention_parts)-1, -1):
                    result_sequence.append(sentence_embedding[j])
                break
            else:
                for j in range(len(sentence_embedding)-1, -1, -1):
                    result_sequence.append(sentence_embedding[j])
                    
        return result_sequence
    
#    def check_double_spaces(self, sentence_text):
#        double_space = 0
#        i = 0
#        while i < (len(sentence_text) - 1):
#            c = sentence_text[i]
#            next_c = sentence_text[i+1]
#            if c == ' ' and next_c == ' ':
#                double_space = double_space + 1
#                i = i + 2
#            else:
#                i = i + 1
#        return double_space
#    
    def map_raw_sequence_to_sentence(self, pre_padding_count, sorted_sentence_num_list, sentence_num_to_sentence_text, start_token_index_in_sentence, end_token_index_in_sentence):
        token_index = pre_padding_count
        token_index_to_sentence_num = {}
        token_index_to_index_in_sentence = {}
        for i in range(len(sorted_sentence_num_list)):
            sentence_num = sorted_sentence_num_list[i]
            sentence_text = sentence_num_to_sentence_text[sentence_num]
#            double_space = self.check_double_spaces(sentence_text)
            sentence_text_parts = sentence_text.split()
            if i == 0 and len(sorted_sentence_num_list) > 1:
                for j in range(start_token_index_in_sentence, len(sentence_text_parts)):
                    token_index_to_sentence_num[token_index] = sentence_num
                    token_index_to_index_in_sentence[token_index] = j
                    token_index = token_index + 1
            elif i == len(sorted_sentence_num_list) - 1 and len(sorted_sentence_num_list) > 1:
                for j in range(end_token_index_in_sentence+1):
                    token_index_to_sentence_num[token_index] = sentence_num
                    token_index_to_index_in_sentence[token_index] = j
                    token_index = token_index + 1
            elif len(sorted_sentence_num_list) == 1:
                for j in range(start_token_index_in_sentence, end_token_index_in_sentence+1):
                    token_index_to_sentence_num[token_index] = sentence_num
                    token_index_to_index_in_sentence[token_index] = j
                    token_index = token_index + 1
            else:
                for j in range(len(sentence_text_parts)):
                    token_index_to_sentence_num[token_index] = sentence_num
                    token_index_to_index_in_sentence[token_index] = j
                    token_index = token_index + 1
                    
        return token_index_to_sentence_num, token_index_to_index_in_sentence
    
    def get_updated_sentence_num_to_sentence_text(self, sentence_num_to_sentence_text, token_index_to_sentence_num, token_index_to_index_in_sentence, related_mentions):
        updated_sentence_num_to_sentence_text = {}
        sentence_num_to_mention_list = {}

        for related_mention in related_mentions:           
            start_pos = related_mention[0]
            end_pos = related_mention[1]
            
            if start_pos <= 50:
                continue
            
            if start_pos not in token_index_to_sentence_num:
                print('xxxxx')
            sentence_num = token_index_to_sentence_num[start_pos]
            start_index_in_sentence = token_index_to_index_in_sentence[start_pos]
            end_index_in_sentence = token_index_to_index_in_sentence[end_pos]
            if sentence_num not in sentence_num_to_mention_list:
                temp = []
                temp.append([start_index_in_sentence, end_index_in_sentence])
                sentence_num_to_mention_list[sentence_num] = temp
            else:
                sentence_num_to_mention_list[sentence_num].append([start_index_in_sentence, end_index_in_sentence])
            
        for sentence_num, mention_list in sentence_num_to_mention_list.items():
            sorted_mention_list = sorted(mention_list, key=lambda x: x[0], reverse=True)
            sentence_text = sentence_num_to_sentence_text[sentence_num]
            updated_sentence_text = sentence_text
            for sorted_mention in sorted_mention_list:
                start_pos = sorted_mention[0]
                end_pos = sorted_mention[1]
                updated_sentence_text = self.get_updated_sentence(updated_sentence_text, start_pos, '[MASK]', end_pos - start_pos + 1)
            
            updated_sentence_num_to_sentence_text[sentence_num] = updated_sentence_text
                
        return updated_sentence_num_to_sentence_text
      
    def get_from_info(self, sentence_num_to_sentence_text, focus_mention_sentence_num, focus_mention_index_in_sentence):
        total_token_size = 50
        pre_padding = 0
        from_sentence = -1
        from_token = -1
        sorted_sentence_num_list = sorted(sentence_num_to_sentence_text.keys(), reverse=True)
        for sentence_num in sorted_sentence_num_list:
            sentence_text = sentence_num_to_sentence_text[sentence_num]
            if sentence_num > focus_mention_sentence_num:
                continue
            elif sentence_num == focus_mention_sentence_num:
                if focus_mention_index_in_sentence >= 50:
                    from_token = focus_mention_index_in_sentence - 50
                    from_sentence = sentence_num
                    total_token_size = 0
                    break
                else:
                    total_token_size = 50 - focus_mention_index_in_sentence
            else:
                sentence_text_parts = sentence_text.split()
                if len(sentence_text_parts) >= total_token_size:
                    from_sentence = sentence_num
                    from_token = len(sentence_text_parts) - total_token_size
                    total_token_size = 0
                    break
                else:
                    total_token_size = total_token_size - len(sentence_text_parts)
                    
        if total_token_size != 0:
            from_sentence = sorted_sentence_num_list[len(sorted_sentence_num_list)-1]
            from_token = 0
            pre_padding = total_token_size
            
        return from_sentence, from_token, pre_padding
    
    def get_to_info(self, sentence_num_to_sentence_text, focus_mention_sentence_num, focus_mention_index_in_sentence, focus_mention):
        total_token_size = 50
        post_padding = 0
        to_sentence = -1
        to_token = -1
        focus_mention_parts = focus_mention.split()
        sorted_sentence_num_list = sorted(sentence_num_to_sentence_text.keys())
        pre_sentence_len = 0
        for sentence_num in sorted_sentence_num_list:
            sentence_text = sentence_num_to_sentence_text[sentence_num]
            sentence_text_parts = sentence_text.split()
            pre_sentence_len = len(sentence_text_parts)
            if sentence_num < focus_mention_sentence_num:
                continue
            elif sentence_num == focus_mention_sentence_num:
                if focus_mention_index_in_sentence+len(focus_mention_parts) + 50 <= len(sentence_text_parts):
                    to_token = focus_mention_index_in_sentence+len(focus_mention_parts) + 49
                    to_sentence = sentence_num
                    total_token_size = 0
                    break
                else:
                    total_token_size = 50 - (len(sentence_text_parts) - focus_mention_index_in_sentence - len(focus_mention_parts))
            else:
                if len(sentence_text_parts) >= total_token_size:
                    to_sentence = sentence_num
                    to_token = total_token_size - 1
                    total_token_size = 0
                    break
                else:
                    total_token_size = total_token_size - len(sentence_text_parts)
                    
        if total_token_size != 0:
            to_sentence = sorted_sentence_num_list[len(sorted_sentence_num_list)-1]
            to_token = pre_sentence_len - 1
            post_padding = total_token_size
            
        return to_sentence, to_token, post_padding
    
    def get_used_bit_and_same_bit(self, mention, cluster_id, cluster_id_to_used_mentions, cluster_id_to_last_mention):
        used_bit = 0
        same_bit = 0
        if cluster_id in cluster_id_to_used_mentions:
            temp = cluster_id_to_used_mentions[cluster_id]
            if mention.lower() in temp:
                used_bit = 1
            
        if cluster_id in cluster_id_to_last_mention:             
            temp = cluster_id_to_last_mention[cluster_id]
            if temp == mention.lower():
                same_bit = 1

        return used_bit, same_bit                                         
    
    def prune_mention_set_dev(self, mention_set, focus_mention):
        updated_mention_set=[]
        list1 = ['his','her','our','their']
        list2 = ['himself','herself','ourselves','themselves']
        list3 = ['you','he','she','we','they']
        list4 = ['you','him','her','us','them']
        if (focus_mention.lower() in list1) or (focus_mention.lower().endswith('\'s')) or (focus_mention.lower().endswith('’s')):
            for mention in mention_set:
                if (mention.lower() in list1) or (mention.lower().endswith('\'s')) or (mention.lower().endswith('’s')):
                    updated_mention_set.append(mention.lower())
        elif focus_mention.lower() in list2:
            for mention in mention_set:
                if mention.lower() in list2:
                    updated_mention_set.append(mention.lower())        
        elif focus_mention.lower() in list3:
            for mention in mention_set:
                if (mention.lower() in list1) or (mention.lower().endswith('\'s')) or (mention.lower().endswith('’s')) or (mention.lower() in list2) or (mention.lower() in list4):
                    continue
                else:
                    updated_mention_set.append(mention.lower())
        elif focus_mention.lower() in list4:
            for mention in mention_set:
                if (mention.lower() in list1) or (mention.lower().endswith('\'s')) or (mention.lower().endswith('’s')) or (mention.lower() in list2) or (mention.lower() in list3):
                    continue
                else:
                    updated_mention_set.append(mention.lower())
        elif (focus_mention.lower() not in list1) and (not focus_mention.lower().endswith('\'s')) and (not focus_mention.lower().endswith('’s')):
            for mention in mention_set:
                if (mention.lower() in list1) or (mention.lower().endswith('\'s')) or (mention.lower().endswith('’s')):
                    continue
                else:
                    updated_mention_set.append(mention.lower())
        elif focus_mention.lower() not in list2:
            for mention in mention_set:
                if mention.lower() in list2:
                    continue
                else:
                    updated_mention_set.append(mention.lower())
        else:
            updated_mention_set = mention_set
            
        return updated_mention_set
    
    def prune_mention_set_test(self, mention_set, focus_mention):
        updated_mention_set=[]
        list1 = ['my','your','his', 'her','our','their']
        list2 = ['myself','yourself','himself','herself','ourselves','themselves']
        list3 = ['i','you','he','she','we','they']
        list4 = ['me','you','him','her','us','them']
        if (focus_mention.lower() in list1) or (focus_mention.lower().endswith('\'s')) or (focus_mention.lower().endswith('’s')):
            for mention in mention_set:
                if (mention.lower() in list1) or (mention.lower().endswith('\'s')) or (mention.lower().endswith('’s')):
                    updated_mention_set.append(mention.lower())
        elif focus_mention.lower() in list2:
            for mention in mention_set:
                if mention.lower() in list2:
                    updated_mention_set.append(mention.lower())        
        elif focus_mention.lower() in list3:
            for mention in mention_set:
                if (mention.lower() in list1) or (mention.lower().endswith('\'s')) or (mention.lower().endswith('’s')) or (mention.lower() in list2) or (mention.lower() in list4):
                    continue
                else:
                    updated_mention_set.append(mention.lower())
        elif focus_mention.lower() in list4:
            for mention in mention_set:
                if (mention.lower() in list1) or (mention.lower().endswith('\'s')) or (mention.lower().endswith('’s')) or (mention.lower() in list2) or (mention.lower() in list3):
                    continue
                else:
                    updated_mention_set.append(mention.lower())
        elif (focus_mention.lower() not in list1) and (not focus_mention.lower().endswith('\'s')) and (not focus_mention.lower().endswith('’s')):
            for mention in mention_set:
                if (mention.lower() in list1) or (mention.lower().endswith('\'s')) or (mention.lower().endswith('’s')):
                    continue
                else:
                    updated_mention_set.append(mention.lower())
        elif focus_mention.lower() not in list2:
            for mention in mention_set:
                if mention.lower() in list2:
                    continue
                else:
                    updated_mention_set.append(mention.lower())
        else:
            updated_mention_set = mention_set
            
        return updated_mention_set 
        
    def get_model_development_data_scenario2(self, data_dir, model, margin):
        # Put model to evaluation mode
        model.eval()

        with torch.no_grad():
            # Evaluate the trained model on validate set
            valid_losses = []
            total_correct = 0
            total_size = 0
            total_entity = 0
            
            files = [join(data_dir,ff) for ff in listdir(data_dir) if isfile(join(data_dir,ff))]
            
            for data_path in files:
                self.read_file(data_path)

                previous_mentions = []
                previous_mention_cluster_ids = []
                correct_previous_mentions = []
                previous_mention_index_in_sentence_list = []
                previous_mention_sentence_num_list = []

                for raw_document_data in self.raw_document_data_list:
                    cluster_id_to_cluster_data = raw_document_data._gold_cluster_id_to_cluster_data
                    raw_sequence_data_list = raw_document_data._raw_sequence_data_list
                    results = self.get_similar_clusters(cluster_id_to_cluster_data)
                    cluster_id_to_he_clusters = results[0]
                    cluster_id_to_she_clusters = results[1]
                    cluster_id_to_it_clusters = results[2]
                    cluster_id_to_they_clusters = results[3]
                
                    appeared_focus_mention_cluster_id_list = []
                    used_cluster_id_to_count = {}
                    cluster_id_to_used_mentions = {}
                    cluster_id_to_last_mention = {}
                    original_sentence_num_to_embedding = {}
                    for raw_sequence_data in raw_sequence_data_list:
                        is_subject = raw_sequence_data._is_subject
                        is_object = raw_sequence_data._is_object
                        focus_mention_cluster_id = raw_sequence_data._focus_mention_cluster_id
                        focus_mention = raw_sequence_data._original_focus_mention
                        focus_mention_parts = focus_mention.split()
                        if focus_mention_cluster_id not in cluster_id_to_cluster_data:
                            print('eee')
                        focus_mention_cluster_data = cluster_id_to_cluster_data[focus_mention_cluster_id]
                        focus_mention_cluster_total_mention = focus_mention_cluster_data._total_count
                        focus_mention_sentence_num = raw_sequence_data._focus_mention_sentence_num
                        focus_mention_index_in_sentence = raw_sequence_data._focus_mention_index_in_sentence
                        temp_mention_set = focus_mention_cluster_data._mention_list
                        mention_set = self.prune_mention_set_dev(temp_mention_set, focus_mention)
                        
                        related_mentions = raw_sequence_data._related_mentions
                        related_mention_cluster_ids = raw_sequence_data._related_mention_cluster_ids
                    
                        updated_related_mentions = []
                        clean_related_mentions = []
                        clean_related_mention_cluster_ids = []
                        for i in range(len(related_mention_cluster_ids)):
                            related_mention_cluster_id = related_mention_cluster_ids[i]
                            if related_mention_cluster_id == focus_mention_cluster_id:
                                updated_related_mentions.append(related_mentions[i])
                                clean_related_mentions.append(related_mentions[i])
                                clean_related_mention_cluster_ids.append(related_mention_cluster_id)
                            elif (related_mention_cluster_id in cluster_id_to_he_clusters and focus_mention_cluster_id in cluster_id_to_he_clusters) or (related_mention_cluster_id in cluster_id_to_she_clusters and focus_mention_cluster_id in cluster_id_to_she_clusters) or (related_mention_cluster_id in cluster_id_to_it_clusters and focus_mention_cluster_id in cluster_id_to_it_clusters) or (related_mention_cluster_id in cluster_id_to_they_clusters):
                                clean_related_mentions.append(related_mentions[i])
                                clean_related_mention_cluster_ids.append(related_mention_cluster_id)
                
                        raw_sequence = raw_sequence_data._raw_sequence
                        related_sentence_num_to_sentence_text = raw_sequence_data._related_sentence_num_to_sentence_text
                        start_token_index_in_sentence = raw_sequence_data._start_token_index_in_sentence
                        end_token_index_in_sentence = raw_sequence_data._end_token_index_in_sentence
                    
                        mentions_to_replace = []
                        temp_dict = {}
                        for related_mention in related_mentions:                  
                            if related_mention[0] < 50:
                                mentions_to_replace.append(related_mention)
                                temp_dict[related_mention[0]] = related_mention
                            
                        if len(mention_set) < 2:
                            if len(mentions_to_replace) == 0:
                                previous_mentions = []
                                previous_mention_cluster_ids = []
                                correct_previous_mentions = []
                                previous_mention_index_in_sentence_list = []
                                previous_mention_sentence_num_list = []
                            
                            previous_mentions.append(focus_mention)
                            previous_mention_cluster_ids.append(focus_mention_cluster_id)
                            correct_previous_mentions.append(focus_mention)
                            previous_mention_index_in_sentence_list.append(focus_mention_index_in_sentence)
                            previous_mention_sentence_num_list.append(focus_mention_sentence_num)
                            
                            if focus_mention_cluster_id not in appeared_focus_mention_cluster_id_list:
                                appeared_focus_mention_cluster_id_list.append(focus_mention_cluster_id)
                                
                            total_size =  total_size + 1
                            total_correct = total_correct + 1
                            continue
                        
                        max_seql_length = 0
                        max_seqr_length = 0
                        
                        seql_1_list = []
                        seql_2_list = []
                        seqr_list = []
                        first_mention_bit_seql_list = []
                        second_mention_bit_seql_list = []
                        focus_mention_length_list = []
                        other_mention_length_list = []
                        coreference_chain_length_list = []
                    
                        used_bit_seql_1_list = []
                        used_bit_seql_2_list = []
                        same_bit_seql_1_list = []
                        same_bit_seql_2_list = []
                    
                        is_subject_list = []
                        is_object_list = []
                        label_list = []
                    
                        first_mention_bit = 0
                        second_mention_bit = 0
                        if focus_mention_cluster_id not in used_cluster_id_to_count:
                            used_cluster_id_to_count[focus_mention_cluster_id] = 1
                            first_mention_bit = 1
                        elif used_cluster_id_to_count[focus_mention_cluster_id] == 1:
                            used_cluster_id_to_count[focus_mention_cluster_id] = 2
                            second_mention_bit = 1
                        else:
                            focus_mention_cluster_id_count = used_cluster_id_to_count[focus_mention_cluster_id]
                            used_cluster_id_to_count[focus_mention_cluster_id] = focus_mention_cluster_id_count + 1
                        
                        pre_padding_count = 0
                        for i in range(len(raw_sequence)):
                            word = raw_sequence[i]
                            if word == "<pad>":
                                pre_padding_count = pre_padding_count + 1
                            else:
                                break
                            
                        sorted_sentence_num_list = sorted(related_sentence_num_to_sentence_text.keys())
                        token_index_to_sentence_num, token_index_to_index_in_sentence = self.map_raw_sequence_to_sentence(pre_padding_count, sorted_sentence_num_list, related_sentence_num_to_sentence_text, start_token_index_in_sentence, end_token_index_in_sentence)
                        updated_sentence_num_to_sentence_text_2 = self.get_updated_sentence_num_to_sentence_text(related_sentence_num_to_sentence_text, token_index_to_sentence_num, token_index_to_index_in_sentence, clean_related_mentions)                   
                        last_sentence_num = sorted_sentence_num_list[len(sorted_sentence_num_list)-1]
                        end_token_index_offset_2 = self.get_end_token_offset(token_index_to_sentence_num, last_sentence_num, clean_related_mentions)
                
                        sentence_num_to_sentence_embedding_2 = {}  
                    
                        used_bit_focus, same_bit_focus = self.get_used_bit_and_same_bit(focus_mention, focus_mention_cluster_id, cluster_id_to_used_mentions, cluster_id_to_last_mention)
                    
                        if len(mentions_to_replace) == 0:
                            for sentence_num in related_sentence_num_to_sentence_text:
                                if sentence_num in updated_sentence_num_to_sentence_text_2:
                                    sentence_text = updated_sentence_num_to_sentence_text_2[sentence_num]
                                    token_embeddings_for_sentence = self.get_bert_embedding(sentence_text)
                                    sentence_num_to_sentence_embedding_2[sentence_num] = token_embeddings_for_sentence
                                else:
                                    if sentence_num in original_sentence_num_to_embedding:
                                        token_embeddings_for_sentence = original_sentence_num_to_embedding[sentence_num]
                                        sentence_num_to_sentence_embedding_2[sentence_num] = token_embeddings_for_sentence
                                    else:
                                        sentence_text = related_sentence_num_to_sentence_text[sentence_num]
                                        token_embeddings_for_sentence = self.get_bert_embedding(sentence_text)
                                        sentence_num_to_sentence_embedding_2[sentence_num] = token_embeddings_for_sentence
                                        original_sentence_num_to_embedding[sentence_num] = token_embeddings_for_sentence
                            
                            previous_mention_cluster_ids = []
                            correct_previous_mentions = []
                            previous_mentions = []
                            previous_mention_index_in_sentence_list = []
                            previous_mention_sentence_num_list = []
                        
                            seql_focus = []
                            focus_seq_length = 51 + len(focus_mention_parts)
            
                            for i in range(len(raw_sequence)):
                                word = raw_sequence[i]
                                if word == "<pad>":
                                    padding_embedding = [0] * 768
                                    padding_embedding= numpy.asarray(padding_embedding, dtype=numpy.float32)
                                    seql_focus.append(padding_embedding)
                                else:
                                    break
                            
                            temp_seql_focus = self.get_embedding_sequence_from_left(sorted_sentence_num_list, sentence_num_to_sentence_embedding_2, start_token_index_in_sentence, focus_mention_sentence_num, focus_mention_index_in_sentence, focus_mention_parts)
                
                            seql_focus.extend(temp_seql_focus)
            
                            if max_seql_length < focus_seq_length:
                                max_seql_length = focus_seq_length
                                
                            seqr = []
                            for i in range(len(raw_sequence)-1, len(raw_sequence)-51, -1):
                                word = raw_sequence[i]
                                if word == "<pad>":
                                    padding_embedding = [0] * 768
                                    padding_embedding= numpy.asarray(padding_embedding, dtype=numpy.float32)
                                    seqr.append(padding_embedding)
                                else:
                                    break

                            temp_seqr = self.get_embedding_sequence_from_right(sorted_sentence_num_list, sentence_num_to_sentence_embedding_2, end_token_index_in_sentence+end_token_index_offset_2, focus_mention_sentence_num, focus_mention_index_in_sentence, focus_mention_parts)                                
                            seqr.extend(temp_seqr)
                        
                            seqr_length = len(seqr)
                            if max_seqr_length < seqr_length:
                                max_seqr_length = seqr_length
                        
                            for i in range(len(mention_set)):
                                mention = mention_set[i]

                                if mention.lower() != focus_mention.lower():                                
                                    seql_other = []
                                    mention_parts = mention.split()
                                    other_seq_length = 51 + len(mention_parts)
                    
                                    focus_mention_sentence = ""
                                    if focus_mention_sentence_num not in related_sentence_num_to_sentence_text:
                                        print('rrr')
                                    if focus_mention_sentence_num in updated_sentence_num_to_sentence_text_2:
                                        focus_mention_sentence = updated_sentence_num_to_sentence_text_2[focus_mention_sentence_num]
                                    else:
                                        focus_mention_sentence = related_sentence_num_to_sentence_text[focus_mention_sentence_num]
                                    updated_sentence_text = self.get_updated_sentence(focus_mention_sentence, focus_mention_index_in_sentence, mention, len(focus_mention_parts))
                                    token_embeddings_for_sentence = self.get_bert_embedding(updated_sentence_text)
                                    sentence_num_to_sentence_embedding_2[focus_mention_sentence_num] = token_embeddings_for_sentence
                            
                                    for j in range(len(raw_sequence)):
                                        word = raw_sequence[j]
                                        if word == "<pad>":
                                            padding_embedding = [0] * 768
                                            padding_embedding= numpy.asarray(padding_embedding, dtype=numpy.float32)
                                            seql_other.append(padding_embedding)
                                        else:
                                            break
                    
                                    temp_seql_other = self.get_embedding_sequence_from_left(sorted_sentence_num_list, sentence_num_to_sentence_embedding_2, start_token_index_in_sentence, focus_mention_sentence_num, focus_mention_index_in_sentence, mention_parts)
                        
                                    seql_other.extend(temp_seql_other)
                                
                                    if max_seql_length < other_seq_length:
                                        max_seql_length = other_seq_length
                                
                                    seql_focus_copy = seql_focus.copy()
                                    seqr_copy = seqr.copy()
                                    seql_1_list.append(seql_focus_copy)
                                    seql_2_list.append(seql_other)
                                    seqr_list.append(seqr_copy)
                                
                                    first_mention_bit_seql_list.append(first_mention_bit)
                                    second_mention_bit_seql_list.append(second_mention_bit)
                                
                                    focus_mention_length_list.append(len(focus_mention_parts))
                                    other_mention_length_list.append(len(mention_parts))
                                    coreference_chain_length_list.append(focus_mention_cluster_total_mention)
                                
                                    used_bit_other, same_bit_other = self.get_used_bit_and_same_bit(mention, focus_mention_cluster_id, cluster_id_to_used_mentions, cluster_id_to_last_mention)
                        
                                    used_bit_seql_1_list.append(used_bit_focus)
                                    same_bit_seql_1_list.append(same_bit_focus)
                                    used_bit_seql_2_list.append(used_bit_other)
                                    same_bit_seql_2_list.append(same_bit_other)
                                    
                                    is_subject_list.append(is_subject)
                                    is_object_list.append(is_object)
                                    label_list.append(1)
                        else:
                            temp_dict_sorted_keys = sorted(temp_dict.keys(), reverse=True)
                            x = 0
                            sorted_key_to_replaced_mention = {}
                            sorted_key_to_replaced_mention_cluster_id = {}
                            sorted_key_to_previous_mention_index = {}
                            for sorted_key in temp_dict_sorted_keys:
                                replaced_mention = previous_mentions[len(previous_mentions) - 1 - x]
                                replaced_mention_cluster_id = previous_mention_cluster_ids[len(previous_mention_cluster_ids) - 1 - x]
                                sorted_key_to_replaced_mention[sorted_key] = replaced_mention
                                sorted_key_to_replaced_mention_cluster_id[sorted_key] = replaced_mention_cluster_id
                                sorted_key_to_previous_mention_index[sorted_key] = len(previous_mentions) - 1 - x
                                x = x + 1

                            updated_focus_mention_index_in_sentence = focus_mention_index_in_sentence
                            for sorted_key in temp_dict_sorted_keys:
                                previous_mention_index = sorted_key_to_previous_mention_index[sorted_key]
                                previous_mention = previous_mentions[previous_mention_index]
                                correct_previous_mention = correct_previous_mentions[previous_mention_index]
                                correct_previous_mention_parts = correct_previous_mention.split()
                                previous_mention_index_in_sentence = previous_mention_index_in_sentence_list[previous_mention_index]
                                previous_mention_sentence_num = previous_mention_sentence_num_list[previous_mention_index]
                                previous_mention_sentence_text = ""
                                if previous_mention_sentence_num in updated_sentence_num_to_sentence_text_2:
                                    previous_mention_sentence_text = updated_sentence_num_to_sentence_text_2[previous_mention_sentence_num]
                                else:
                                    previous_mention_sentence_text = related_sentence_num_to_sentence_text[previous_mention_sentence_num]
                                
                                previous_mention_cluster_id = previous_mention_cluster_ids[previous_mention_index]
                                if (previous_mention_cluster_id == focus_mention_cluster_id) or (previous_mention_cluster_id in cluster_id_to_he_clusters and focus_mention_cluster_id in cluster_id_to_he_clusters) or (previous_mention_cluster_id in cluster_id_to_she_clusters and focus_mention_cluster_id in cluster_id_to_she_clusters) or (previous_mention_cluster_id in cluster_id_to_it_clusters and focus_mention_cluster_id in cluster_id_to_it_clusters) or (previous_mention_cluster_id in cluster_id_to_they_clusters):
                                    updated_sentence_text = self.get_updated_sentence(previous_mention_sentence_text, previous_mention_index_in_sentence, previous_mention, len(correct_previous_mention_parts))
                                    updated_sentence_num_to_sentence_text_2[previous_mention_sentence_num] = updated_sentence_text
                            
                                    if previous_mention_sentence_num == focus_mention_sentence_num:
                                        focus_mention_offset = self.get_focus_mention_offset(previous_mention_index_in_sentence, previous_mention, correct_previous_mention, focus_mention_index_in_sentence)
                                        updated_focus_mention_index_in_sentence = updated_focus_mention_index_in_sentence + focus_mention_offset
                                        
                            all_updated_sentence_num_to_sentence_text_2 = {}
                            for sentence_num in related_sentence_num_to_sentence_text:
                                if sentence_num in updated_sentence_num_to_sentence_text_2:
                                    sentence_text = updated_sentence_num_to_sentence_text_2[sentence_num]
                                    token_embeddings_for_sentence = self.get_bert_embedding(sentence_text)
                                    sentence_num_to_sentence_embedding_2[sentence_num] = token_embeddings_for_sentence
                                    all_updated_sentence_num_to_sentence_text_2[sentence_num] = sentence_text
                                else:
                                    all_updated_sentence_num_to_sentence_text_2[sentence_num] = related_sentence_num_to_sentence_text[sentence_num]
                                    if sentence_num in original_sentence_num_to_embedding:
                                        token_embeddings_for_sentence = original_sentence_num_to_embedding[sentence_num]
                                        sentence_num_to_sentence_embedding_2[sentence_num] = token_embeddings_for_sentence
                                    else:
                                        sentence_text = related_sentence_num_to_sentence_text[sentence_num]
                                        token_embeddings_for_sentence = self.get_bert_embedding(sentence_text)
                                        sentence_num_to_sentence_embedding_2[sentence_num] = token_embeddings_for_sentence
                                        original_sentence_num_to_embedding[sentence_num] = token_embeddings_for_sentence
                            
                            seql_focus = []
                            focus_seq_length = 51 + len(focus_mention_parts)
            
                            updated_from_sentence, updated_from_token, updated_pre_padding = self.get_from_info(all_updated_sentence_num_to_sentence_text_2, focus_mention_sentence_num, updated_focus_mention_index_in_sentence)
                        
                            for i in range(updated_pre_padding):
                                padding_embedding = [0] * 768
                                padding_embedding= numpy.asarray(padding_embedding, dtype=numpy.float32)
                                seql_focus.append(padding_embedding)
            
                            temp_seql_focus = self.get_embedding_sequence_from_left(sorted_sentence_num_list, sentence_num_to_sentence_embedding_2, updated_from_token, focus_mention_sentence_num, updated_focus_mention_index_in_sentence, focus_mention_parts, updated_from_sentence)
                
                            seql_focus.extend(temp_seql_focus)
                            
                            if max_seql_length < focus_seq_length:
                                max_seql_length = focus_seq_length
                                
                            seqr = []
                            updated_to_sentence, updated_to_token, updated_post_padding = self.get_to_info(all_updated_sentence_num_to_sentence_text_2, focus_mention_sentence_num, updated_focus_mention_index_in_sentence, focus_mention)
                        
                            for i in range(updated_post_padding):
                                padding_embedding = [0] * 768
                                padding_embedding= numpy.asarray(padding_embedding, dtype=numpy.float32)
                                seqr.append(padding_embedding)
                         
                            temp_seqr = self.get_embedding_sequence_from_right(sorted_sentence_num_list, sentence_num_to_sentence_embedding_2, updated_to_token, focus_mention_sentence_num, updated_focus_mention_index_in_sentence, focus_mention_parts, updated_to_sentence)                                
                            seqr.extend(temp_seqr)
                        
                            seqr_length = len(seqr)
                            if max_seqr_length < seqr_length:
                                max_seqr_length = seqr_length

                            for i in range(len(mention_set)):
                                mention = mention_set[i]

                                if mention.lower() != focus_mention.lower():                                
                                    seql_other = []
                                    mention_parts = mention.split()
                                    other_seq_length = 51 + len(mention_parts)
                    
                                    focus_mention_sentence = ""
                                    if focus_mention_sentence_num in updated_sentence_num_to_sentence_text_2:
                                        focus_mention_sentence = updated_sentence_num_to_sentence_text_2[focus_mention_sentence_num]
                                    else:
                                        focus_mention_sentence = related_sentence_num_to_sentence_text[focus_mention_sentence_num]
                        
                                    updated_sentence_text = self.get_updated_sentence(focus_mention_sentence, updated_focus_mention_index_in_sentence, mention, len(focus_mention_parts))
                                    token_embeddings_for_sentence = self.get_bert_embedding(updated_sentence_text)
                                    sentence_num_to_sentence_embedding_2[focus_mention_sentence_num] = token_embeddings_for_sentence
                                
                                    for j in range(updated_pre_padding):
                                        padding_embedding = [0] * 768
                                        padding_embedding= numpy.asarray(padding_embedding, dtype=numpy.float32)
                                        seql_other.append(padding_embedding)
                            
                                    temp_seql_other = self.get_embedding_sequence_from_left(sorted_sentence_num_list, sentence_num_to_sentence_embedding_2, updated_from_token, focus_mention_sentence_num, updated_focus_mention_index_in_sentence, mention_parts, updated_from_sentence)                                
                        
                                    seql_other.extend(temp_seql_other)
                                    
                                    if max_seql_length < other_seq_length:
                                        max_seql_length = other_seq_length
                                        
                                    seql_focus_copy = seql_focus.copy()
                                    seqr_copy = seqr.copy()
                                    seql_1_list.append(seql_focus_copy)
                                    seql_2_list.append(seql_other)
                                    seqr_list.append(seqr_copy)
                                
                                    first_mention_bit_seql_list.append(first_mention_bit)
                                    second_mention_bit_seql_list.append(second_mention_bit)
                                
                                    focus_mention_length_list.append(len(focus_mention_parts))
                                    other_mention_length_list.append(len(mention_parts))
                                    coreference_chain_length_list.append(focus_mention_cluster_total_mention)
                                
                                    used_bit_other, same_bit_other = self.get_used_bit_and_same_bit(mention, focus_mention_cluster_id, cluster_id_to_used_mentions, cluster_id_to_last_mention)
                        
                                    used_bit_seql_1_list.append(used_bit_focus)
                                    same_bit_seql_1_list.append(same_bit_focus)
                                    used_bit_seql_2_list.append(used_bit_other)
                                    same_bit_seql_2_list.append(same_bit_other)
                                    
                                    is_subject_list.append(is_subject)
                                    is_object_list.append(is_object)
                                    label_list.append(1)

                        total_size = total_size + 1
                        if focus_mention_cluster_id not in appeared_focus_mention_cluster_id_list:
                            appeared_focus_mention_cluster_id_list.append(focus_mention_cluster_id)
                        batch_r, batch_len_r, batch_first_mention_bit, batch_second_mention_bit, batch_used_bit_seql_1, batch_same_bit_seql_1, batch_used_bit_seql_2, batch_same_bit_seql_2, batch_is_subject_list, batch_is_object_list, batch_label = \
                        self.create_padding([seqr_list, first_mention_bit_seql_list, second_mention_bit_seql_list, used_bit_seql_1_list, same_bit_seql_1_list, used_bit_seql_2_list, same_bit_seql_2_list, is_subject_list, is_object_list, label_list], max_seqr_length)

                        rep_focus_mention_length_array = self.get_mention_distance_embedding(focus_mention_length_list)
                        rep_other_mention_length_array = self.get_mention_distance_embedding(other_mention_length_list)
                        rep_coreference_chain_length_array = self.get_coreference_chain_distance_embedding(coreference_chain_length_list)

                        rep_r = Variable(torch.from_numpy(batch_r),
                                     requires_grad=False).type(self.float_type)
                        len_r = Variable(torch.from_numpy(batch_len_r),
                                     requires_grad=False).type(self.long_type)
                        rep_first_mention_bit = Variable(torch.from_numpy(batch_first_mention_bit),
                                     requires_grad=False).type(self.float_type)
                        rep_second_mention_bit = Variable(torch.from_numpy(batch_second_mention_bit),
                                     requires_grad=False).type(self.float_type)
                        rep_focus_mention_length = Variable(torch.from_numpy(rep_focus_mention_length_array),
                                     requires_grad=False).type(self.float_type)
                        rep_other_mention_length = Variable(torch.from_numpy(rep_other_mention_length_array),
                                     requires_grad=False).type(self.float_type)
                        rep_coreference_chain_length = Variable(torch.from_numpy(rep_coreference_chain_length_array),
                                     requires_grad=False).type(self.float_type)
                        rep_used_bit_seql_1 = Variable(torch.from_numpy(batch_used_bit_seql_1),
                                     requires_grad=False).type(self.float_type)
                        rep_same_bit_seql_1 = Variable(torch.from_numpy(batch_same_bit_seql_1),
                                     requires_grad=False).type(self.float_type)
                        rep_used_bit_seql_2 = Variable(torch.from_numpy(batch_used_bit_seql_2),
                                     requires_grad=False).type(self.float_type)
                        rep_same_bit_seql_2 = Variable(torch.from_numpy(batch_same_bit_seql_2),
                                     requires_grad=False).type(self.float_type)

                        rep_is_subject_list = Variable(torch.from_numpy(batch_is_subject_list),
                                     requires_grad=False).type(self.float_type)
                        rep_is_object_list = Variable(torch.from_numpy(batch_is_object_list),
                                     requires_grad=False).type(self.float_type)
                        label = Variable(torch.from_numpy(batch_label),
                                     requires_grad=False).type(self.float_type)
                            
                        # Forward pass: predict scores
                        score_l1, score_l2 = model(seql_1_list, seql_2_list, max_seql_length, rep_r, len_r, rep_first_mention_bit, rep_second_mention_bit, rep_focus_mention_length, rep_other_mention_length, rep_coreference_chain_length, rep_used_bit_seql_1, rep_same_bit_seql_1, rep_used_bit_seql_2, rep_same_bit_seql_2, rep_is_subject_list, rep_is_object_list)
                        loss = NN.MarginRankingLoss(margin=margin)(score_l1, score_l2, label) 
                        valid_losses.append(loss.item())
                      
                        larger_score = 0
                        if self.use_gpu:
                            s2 = score_l2.cpu()
                            s1 = score_l1.cpu()                            
                        s2 = s2.data.numpy()
                        s1 = s1.data.numpy()
                        max_index = -1
                        max_value = -10000000
                        for ii in range(len(s1)):
                            score_1 = s1[ii]
                            score_2 = s2[ii]
                            if score_2 > score_1 and score_2 > max_value:
                                max_value = score_2
                                max_index = ii
                                larger_score = 2
                            elif score_1 >= score_2 and score_1 > max_value:
                                max_value = score_1
                                max_index = ii
                                larger_score = 1
                
                        chosen_mention = ""
                        # Evaluation 2
                        if larger_score == 1:
                            total_correct = total_correct + 1
                            previous_mentions.append(focus_mention)
                            previous_mention_cluster_ids.append(focus_mention_cluster_id)
                            correct_previous_mentions.append(focus_mention)
                            previous_mention_index_in_sentence_list.append(focus_mention_index_in_sentence)
                            previous_mention_sentence_num_list.append(focus_mention_sentence_num)
                            chosen_mention = focus_mention 
                        else:                  
                            INDEX = 0
                            for mention in mention_set:
                                if mention.lower() != focus_mention.lower():
                                    if max_index == INDEX:
                                        previous_mentions.append(mention)
                                        previous_mention_cluster_ids.append(focus_mention_cluster_id)
                                        correct_previous_mentions.append(focus_mention)
                                        previous_mention_index_in_sentence_list.append(focus_mention_index_in_sentence)
                                        previous_mention_sentence_num_list.append(focus_mention_sentence_num)
                                        chosen_mention = mention
                                        break
                                    INDEX = INDEX + 1
                        if focus_mention_cluster_id not in cluster_id_to_used_mentions:
                            temp = []
                            temp.append(chosen_mention.lower())
                            cluster_id_to_used_mentions[focus_mention_cluster_id] = temp
                        else:
                            temp = cluster_id_to_used_mentions[focus_mention_cluster_id]
                            if chosen_mention.lower() not in temp:
                                temp.append(chosen_mention.lower())
                                cluster_id_to_used_mentions[focus_mention_cluster_id] = temp
                        if focus_mention_cluster_id not in cluster_id_to_last_mention:
                            cluster_id_to_last_mention[focus_mention_cluster_id] = chosen_mention.lower()
                        else:
                            temp = cluster_id_to_last_mention[focus_mention_cluster_id]
                            if temp != chosen_mention.lower():
                                cluster_id_to_last_mention[focus_mention_cluster_id] = chosen_mention.lower()
                    
                    total_entity = total_entity + len(appeared_focus_mention_cluster_id_list)
        
        return valid_losses, total_correct, total_size
    
    def get_model_testing_data_scenario2(self, f, f_temp, data_dir, model, margin, gold_setting=False, write_result=False):
        # Put model to evaluation mode
        model.eval()

        pov_test_auto_file_list = ['dispatches','faraway','mind','mother','narcissist','night','nobody','recipe','selkie','understand','walk','water']

        with torch.no_grad():
            # Evaluate the trained model on validate set
            valid_losses = []
            total_correct = 0
            total_size = 0
            total_entity = 0
            
            total_annotated_mention = 0      
            total_annotated_mention_correctly_identified = 0
            total_annotated_mention_correctly_identified_correctly_converted = 0
            total_annotated_mention_incorrectly_identified = 0
            total_annotated_mention_incorrectly_identified_incorrectly_converted = 0
            total_annotated_mention_not_identified = 0
            total_annotated_mention_not_identified_incorrectly_converted = 0
            total_identified_mention_not_annotated = 0
            total_identified_mention_not_annotated_incorrectly_converted = 0
            total_annotated_verb = 0
            total_annotated_verb_correctly_identified = 0
            total_annotated_verb_correctly_identified_correctly_converted = 0
            total_annotated_verb_not_identified = 0
            total_annotated_verb_not_identified_incorrectly_converted = 0
            total_identified_verb_not_annotated = 0
            total_identified_verb_not_annotated_incorrectly_converted = 0
            total_se_error = 0
            
            files = [join(data_dir,ff) for ff in listdir(data_dir) if isfile(join(data_dir,ff))]
            
            for data_path in files:
                self.read_file(data_path)
                    
                is_test = False
                for pov_test_auto_file in pov_test_auto_file_list:
                    if data_path.find(pov_test_auto_file) != -1:
                        is_test = True
                        break
                    
                previous_mentions = []
                previous_mention_cluster_ids = []
                correct_previous_mentions = []
                previous_mention_index_in_sentence_list = []
                previous_mention_sentence_num_list = []

                for raw_document_data in self.raw_document_data_list:
                    current_total_correct = 0
                    current_total_size = 0
                    current_total_entity = 0
                    current_total_confounding_mentions = 0
                    current_total_se_count = 0
                    
                    current_total_annotated_mention = 0      
                    current_total_annotated_mention_correctly_identified = 0
                    current_total_annotated_mention_correctly_identified_correctly_converted = 0
                    current_total_annotated_mention_incorrectly_identified = 0
                    current_total_annotated_mention_incorrectly_identified_incorrectly_converted = 0
                    current_total_annotated_mention_not_identified = 0
                    current_total_annotated_mention_not_identified_incorrectly_converted = 0
                    current_total_identified_mention_not_annotated = 0
                    current_total_identified_mention_not_annotated_incorrectly_converted = 0
                    current_total_annotated_verb = 0
                    current_total_annotated_verb_correctly_identified = 0
                    current_total_annotated_verb_correctly_identified_correctly_converted = 0
                    current_total_annotated_verb_not_identified = 0
                    current_total_annotated_verb_not_identified_incorrectly_converted = 0
                    current_total_identified_verb_not_annotated = 0
                    current_total_identified_verb_not_annotated_incorrectly_converted = 0
                    current_total_se_error = 0
                    
                    if write_result:
                        f.write("<document_text>")
                        f.write("\n")
                        f.write(raw_document_data._document_text)
                        f.write("\n")
                        f.write("<document_text>")
                        f.write('\n')
                        
                    gold_cluster_id_to_cluster_data = raw_document_data._gold_cluster_id_to_cluster_data
                    current_total_entity = len(gold_cluster_id_to_cluster_data)
                    for gold_cluster_id in gold_cluster_id_to_cluster_data:
                        cluster_data = gold_cluster_id_to_cluster_data[gold_cluster_id]
                        mention_list = cluster_data._mention_list
                        current_total_se_count = current_total_se_count + len(mention_list)
                        
                    auto_cluster_id_to_cluster_data = raw_document_data._auto_cluster_id_to_cluster_data
                    raw_sequence_data_list = raw_document_data._raw_sequence_data_list
                    results = None
                    if gold_setting:
                        results = self.get_similar_clusters(gold_cluster_id_to_cluster_data)
                    else:
                        results = self.get_similar_clusters(auto_cluster_id_to_cluster_data)
                    cluster_id_to_he_clusters = results[0]
                    cluster_id_to_she_clusters = results[1]
                    cluster_id_to_it_clusters = results[2]
                    cluster_id_to_they_clusters = results[3]
                
                    used_cluster_id_to_count = {}
                    cluster_id_to_used_mentions = {}
                    cluster_id_to_last_mention = {}
                    original_sentence_num_to_embedding = {}
                    appeared_focus_mention_cluster_id_list = []
                    for raw_sequence_data in raw_sequence_data_list:  
                        is_identified_mention = raw_sequence_data._is_identified_mention
                        is_annotated_mention = raw_sequence_data._is_annotated_mention
                        is_identified_verb = raw_sequence_data._is_identified_verb
                        is_annotated_verb = raw_sequence_data._is_annotated_verb
                    
                        is_subject = raw_sequence_data._is_subject
                        is_object = raw_sequence_data._is_object
                        focus_mention_cluster_id = raw_sequence_data._focus_mention_cluster_id
                        
                        if not gold_setting:
                            if is_identified_verb == 1 and is_annotated_verb == 1:
                                total_annotated_verb = total_annotated_verb + 1
                                current_total_annotated_verb = current_total_annotated_verb + 1
                                total_annotated_verb_correctly_identified = total_annotated_verb_correctly_identified + 1
                                current_total_annotated_verb_correctly_identified = current_total_annotated_verb_correctly_identified + 1
                                identified_converted_verb = raw_sequence_data._identified_converted_verb
                                annotated_converted_verb = raw_sequence_data._annotated_converted_verb
                                if identified_converted_verb.lower() == annotated_converted_verb.lower():
                                    total_annotated_verb_correctly_identified_correctly_converted = total_annotated_verb_correctly_identified_correctly_converted + 1
                                    current_total_annotated_verb_correctly_identified_correctly_converted = current_total_annotated_verb_correctly_identified_correctly_converted + 1
                                if write_result:
                                    f.write("<identified_verb>: 1")
                                    f.write('\t')
                                    f.write("<annotated_verb>: 1")
                                    f.write('\t')
                                    f.write("<identified_converted_verb>: ")
                                    f.write(identified_converted_verb)
                                    f.write('\t')
                                    f.write("<annotated_converted_verb>: ")
                                    f.write(annotated_converted_verb)
                                    f.write('\t')
                                    f.write("<start_pos>: ")
                                    f.write(str(raw_sequence_data._start_pos))
                                    f.write("\n")                                   
                                continue
                            elif is_identified_verb == 0 and is_annotated_verb == 1:
                                total_annotated_verb = total_annotated_verb + 1
                                current_total_annotated_verb = current_total_annotated_verb + 1
                                total_annotated_verb_not_identified = total_annotated_verb_not_identified + 1
                                current_total_annotated_verb_not_identified = current_total_annotated_verb_not_identified + 1
                                annotated_original_verb = raw_sequence_data._annotated_original_verb
                                annotated_converted_verb = raw_sequence_data._annotated_converted_verb
                                if annotated_original_verb.lower() != annotated_converted_verb.lower():
                                    total_annotated_verb_not_identified_incorrectly_converted = total_annotated_verb_not_identified_incorrectly_converted + 1
                                    current_total_annotated_verb_not_identified_incorrectly_converted = current_total_annotated_verb_not_identified_incorrectly_converted + 1
                                if write_result:
                                    f.write("<identified_verb>: 0")
                                    f.write('\t')
                                    f.write("<annotated_verb>: 1")
                                    f.write('\t')
                                    f.write("<annotated_original_verb>: ")
                                    f.write(annotated_original_verb)
                                    f.write('\t')
                                    f.write("<annotated_converted_verb>: ")
                                    f.write(annotated_converted_verb)
                                    f.write('\t')
                                    f.write("<start_pos>: ")
                                    f.write(str(raw_sequence_data._start_pos))
                                    f.write("\n")  
                                continue
                            elif is_identified_verb == 1 and is_annotated_verb == 0:
                                total_identified_verb_not_annotated = total_identified_verb_not_annotated + 1
                                current_total_identified_verb_not_annotated = current_total_identified_verb_not_annotated + 1
                                identified_original_verb = raw_sequence_data._identified_original_verb
                                identified_converted_verb = raw_sequence_data._identified_converted_verb
                                if identified_original_verb.lower() != identified_converted_verb.lower():
                                    total_identified_verb_not_annotated_incorrectly_converted = total_identified_verb_not_annotated_incorrectly_converted + 1
                                    current_total_identified_verb_not_annotated_incorrectly_converted = current_total_identified_verb_not_annotated_incorrectly_converted + 1
                                if write_result:
                                    f.write("<identified_verb>: 1")
                                    f.write('\t')
                                    f.write("<annotated_verb>: 0")
                                    f.write('\t')
                                    f.write("<identified_original_verb>: ")
                                    f.write(identified_original_verb)
                                    f.write('\t')
                                    f.write("<identified_converted_verb>: ")
                                    f.write(identified_converted_verb)
                                    f.write('\t')
                                    f.write("<start_pos>: ")
                                    f.write(str(raw_sequence_data._start_pos))
                                    f.write("\n")
                                continue
                            
                            if is_identified_mention == 0 and is_annotated_mention == 1:
                                total_annotated_mention = total_annotated_mention + 1
                                current_total_annotated_mention = current_total_annotated_mention + 1
                                total_annotated_mention_not_identified = total_annotated_mention_not_identified + 1
                                current_total_annotated_mention_not_identified = current_total_annotated_mention_not_identified + 1
                                annotated_original_mention = raw_sequence_data._annotated_original_focus_mention
                                annotated_converted_mention = raw_sequence_data._annotated_converted_focus_mention
                                if annotated_original_mention.lower() != annotated_converted_mention.lower():
                                    total_annotated_mention_not_identified_incorrectly_converted = total_annotated_mention_not_identified_incorrectly_converted + 1
                                    current_total_annotated_mention_not_identified_incorrectly_converted = current_total_annotated_mention_not_identified_incorrectly_converted + 1
                                if write_result:
                                    f.write("<identified_mention>: 0")
                                    f.write('\t')
                                    f.write("<annotated_mention>: 1")
                                    f.write('\t')
                                    f.write("<annotated_original_mention>: ")
                                    f.write(annotated_original_mention)
                                    f.write('\t')
                                    f.write("<annotated_converted_mention>: ")
                                    f.write(annotated_converted_mention)
                                    f.write('\t')
                                    f.write("<start_pos>: ")
                                    f.write(str(raw_sequence_data._start_pos))
                                    f.write("\n")
                                continue
                            
                        original_focus_mention = raw_sequence_data._identified_original_focus_mention
                        converted_focus_mention = raw_sequence_data._converted_focus_mention
                        original_focus_mention_parts = original_focus_mention.split()
                        converted_focus_mention_parts = converted_focus_mention.split()
                        gold_focus_mention_cluster_data = gold_cluster_id_to_cluster_data[focus_mention_cluster_id]
                        auto_focus_mention_cluster_data = auto_cluster_id_to_cluster_data[focus_mention_cluster_id]
                        focus_mention_cluster_total_mention = gold_focus_mention_cluster_data._total_count
                        focus_mention_sentence_num = raw_sequence_data._focus_mention_sentence_num
                        focus_mention_index_in_sentence = raw_sequence_data._focus_mention_index_in_sentence
                        temp_mention_set = []
                        if gold_setting:
                            temp_mention_set = gold_focus_mention_cluster_data._mention_list
                        else:
                            temp_mention_set = auto_focus_mention_cluster_data._mention_list
                        mention_set = self.prune_mention_set_test(temp_mention_set, original_focus_mention)
                        
                        related_mentions = raw_sequence_data._related_mentions
                        related_mention_cluster_ids = raw_sequence_data._related_mention_cluster_ids
                    
                        updated_related_mentions = []
                        clean_related_mentions = []
                        clean_related_mention_cluster_ids = []
                        for i in range(len(related_mention_cluster_ids)):
                            related_mention_cluster_id = related_mention_cluster_ids[i]
                            if related_mention_cluster_id == focus_mention_cluster_id:
                                updated_related_mentions.append(related_mentions[i])
                                clean_related_mentions.append(related_mentions[i])
                                clean_related_mention_cluster_ids.append(related_mention_cluster_id)
                            elif (related_mention_cluster_id in cluster_id_to_he_clusters and focus_mention_cluster_id in cluster_id_to_he_clusters) or (related_mention_cluster_id in cluster_id_to_she_clusters and focus_mention_cluster_id in cluster_id_to_she_clusters) or (related_mention_cluster_id in cluster_id_to_it_clusters and focus_mention_cluster_id in cluster_id_to_it_clusters) or (related_mention_cluster_id in cluster_id_to_they_clusters):
                                clean_related_mentions.append(related_mentions[i])
                                clean_related_mention_cluster_ids.append(related_mention_cluster_id)
                
                        raw_sequence = raw_sequence_data._raw_sequence
                        related_sentence_num_to_sentence_text = raw_sequence_data._related_sentence_num_to_sentence_text
                        start_token_index_in_sentence = raw_sequence_data._start_token_index_in_sentence
                        end_token_index_in_sentence = raw_sequence_data._end_token_index_in_sentence
                    
                        mentions_to_replace = []
                        temp_dict = {}
                        for related_mention in related_mentions:                  
                            if related_mention[0] < 50:
                                mentions_to_replace.append(related_mention)
                                temp_dict[related_mention[0]] = related_mention
                            
                        if len(mention_set) < 2:
                            if len(mentions_to_replace) == 0:
                                previous_mentions = []
                                previous_mention_cluster_ids = []
                                correct_previous_mentions = []
                                previous_mention_index_in_sentence_list = []
                                previous_mention_sentence_num_list = []
                            
                            previous_mentions.append(converted_focus_mention)
                            previous_mention_cluster_ids.append(focus_mention_cluster_id)
                            correct_previous_mentions.append(original_focus_mention)
                            previous_mention_index_in_sentence_list.append(focus_mention_index_in_sentence)
                            previous_mention_sentence_num_list.append(focus_mention_sentence_num)
                            if gold_setting:
                                total_size = total_size + 1
                                current_total_size = current_total_size + 1
                                total_correct = total_correct + 1
                                current_total_correct = current_total_correct + 1
                                
                                if focus_mention_cluster_id != -1:
                                    current_total_confounding_mentions = current_total_confounding_mentions + 1
                                    
                                if focus_mention_cluster_id not in appeared_focus_mention_cluster_id_list:
                                    appeared_focus_mention_cluster_id_list.append(focus_mention_cluster_id)
                                if write_result:
                                    f.write("<original_focus_mention>: ")
                                    f.write(original_focus_mention)
                                    f.write('\t')
                                    f.write("<converted_focus_mention>: ")
                                    f.write(converted_focus_mention)
                                    f.write('\t')
                                    f.write("<start_pos>: ")
                                    f.write(str(raw_sequence_data._start_pos))
                                    f.write("\n")
                            else:
                                if is_identified_mention == 1 and is_annotated_mention == 1:
                                    total_annotated_mention = total_annotated_mention + 1
                                    current_total_annotated_mention = current_total_annotated_mention + 1
                                    annotated_original_mention = raw_sequence_data._annotated_original_focus_mention
                                    identified_original_mention = raw_sequence_data._identified_original_focus_mention
                                    annotated_converted_mention = raw_sequence_data._annotated_converted_focus_mention
                                    identified_converted_mention = raw_sequence_data._identified_converted_focus_mention
                                    if annotated_original_mention.lower() == identified_original_mention.lower():
                                        total_annotated_mention_correctly_identified = total_annotated_mention_correctly_identified + 1
                                        current_total_annotated_mention_correctly_identified = current_total_annotated_mention_correctly_identified + 1
                                        if len(mention_set) == 0:
                                            print('yyyy')
                                        if mention_set[0].lower() == converted_focus_mention.lower():
                                            total_annotated_mention_correctly_identified_correctly_converted = total_annotated_mention_correctly_identified_correctly_converted + 1
                                            current_total_annotated_mention_correctly_identified_correctly_converted = current_total_annotated_mention_correctly_identified_correctly_converted + 1
                                        else:
                                            total_se_error = total_se_error + 1
                                            current_total_se_error = current_total_se_error + 1
                                    else:
                                        total_annotated_mention_incorrectly_identified = total_annotated_mention_incorrectly_identified + 1
                                        current_total_annotated_mention_incorrectly_identified = current_total_annotated_mention_incorrectly_identified + 1
                                        if (annotated_original_mention.lower() != converted_focus_mention.lower()) or (identified_original_mention.lower() != mention_set[0].lower()):
                                            total_annotated_mention_incorrectly_identified_incorrectly_converted = total_annotated_mention_incorrectly_identified_incorrectly_converted + 1
                                            current_total_annotated_mention_incorrectly_identified_incorrectly_converted = current_total_annotated_mention_incorrectly_identified_incorrectly_converted + 1
                                    if write_result:
                                        f.write("<identified_mention>: 1")
                                        f.write('\t')
                                        f.write("<annotated_mention>: 1")
                                        f.write('\t')
                                        f.write("<identified_original_mention>: ")
                                        f.write(identified_original_mention)
                                        f.write('\t')
                                        f.write("<annotated_original_mention>: ")
                                        f.write(annotated_original_mention)
                                        f.write('\t')
                                        f.write("<identified_converted_mention>: ")
                                        f.write(mention_set[0])
                                        f.write('\t')
                                        f.write("<annotated_converted_mention>: ")
                                        f.write(converted_focus_mention)
                                        f.write('\t')
                                        f.write("<start_pos>: ")
                                        f.write(str(raw_sequence_data._start_pos))
                                        f.write("\n")
                                elif is_identified_mention == 1 and is_annotated_mention == 0:
                                    total_identified_mention_not_annotated = total_identified_mention_not_annotated + 1
                                    current_total_identified_mention_not_annotated = current_total_identified_mention_not_annotated + 1
                                    identified_original_mention = raw_sequence_data._identified_original_focus_mention
                                    if mention_set[0].lower() != identified_original_mention.lower():
                                        total_identified_mention_not_annotated_incorrectly_converted = total_identified_mention_not_annotated_incorrectly_converted + 1
                                        current_total_identified_mention_not_annotated_incorrectly_converted = current_total_identified_mention_not_annotated_incorrectly_converted + 1
                                    if write_result:
                                        f.write("<identified_mention>: 1")
                                        f.write('\t')
                                        f.write("<annotated_mention>: 0")
                                        f.write('\t')
                                        f.write("<identified_original_mention>: ")
                                        f.write(identified_original_mention)
                                        f.write('\t')
                                        f.write("<identified_converted_mention>: ")
                                        f.write(mention_set[0])
                                        f.write('\t')
                                        f.write("<start_pos>: ")
                                        f.write(str(raw_sequence_data._start_pos))
                                        f.write("\n")
                            continue
                     
                        if write_result:
                            if gold_setting:
                                f.write("<original_focus_mention>: ")
                                f.write(original_focus_mention)
                                f.write('\t')
                                f.write("<converted_focus_mention>: ")
                                f.write(converted_focus_mention)
                                f.write('\t')
                                f.write("<start_pos>: ")
                                f.write(str(raw_sequence_data._start_pos))
                                f.write("\n")
                            else:
                                f.write("<identified_mention>: ")
                                f.write(str(is_identified_mention))
                                f.write('\t')
                                f.write("<annotated_mention>: ")
                                f.write(str(is_annotated_mention))
                                f.write('\t')
                                f.write("<start_pos>: ")
                                f.write(str(raw_sequence_data._start_pos))
                                f.write('\n')
                            f.write("<mention_set>: ")
                            for mention in mention_set:
                                f.write("%s " %mention)
                            f.write("\n")
                            f.write("<raw_sequence>: ")
                            for rs in raw_sequence:
                                f.write("%s " %rs)
                            f.write("\n")
                        
                        no_correct_string_in_se = False
                        max_seql_length = 0
                        max_seqr_length = 0
                        
                        seql_1_list = []
                        seql_2_list = []
                        seqr_list = []
                        first_mention_bit_seql_list = []
                        second_mention_bit_seql_list = []
                        focus_mention_length_list = []
                        other_mention_length_list = []
                        coreference_chain_length_list = []
                    
                        used_bit_seql_1_list = []
                        used_bit_seql_2_list = []
                        same_bit_seql_1_list = []
                        same_bit_seql_2_list = []
                    
                        is_subject_list = []
                        is_object_list = []
                        label_list = []
                    
                        first_mention_bit = 0
                        second_mention_bit = 0
                        if focus_mention_cluster_id not in used_cluster_id_to_count:
                            used_cluster_id_to_count[focus_mention_cluster_id] = 1
                            first_mention_bit = 1
                        elif used_cluster_id_to_count[focus_mention_cluster_id] == 1:
                            used_cluster_id_to_count[focus_mention_cluster_id] = 2
                            second_mention_bit = 1
                        else:
                            focus_mention_cluster_id_count = used_cluster_id_to_count[focus_mention_cluster_id]
                            used_cluster_id_to_count[focus_mention_cluster_id] = focus_mention_cluster_id_count + 1
                        
                        pre_padding_count = 0
                        for i in range(len(raw_sequence)):
                            word = raw_sequence[i]
                            if word == "<pad>":
                                pre_padding_count = pre_padding_count + 1
                            else:
                                break
                            
                        sorted_sentence_num_list = sorted(related_sentence_num_to_sentence_text.keys())
                        token_index_to_sentence_num, token_index_to_index_in_sentence = self.map_raw_sequence_to_sentence(pre_padding_count, sorted_sentence_num_list, related_sentence_num_to_sentence_text, start_token_index_in_sentence, end_token_index_in_sentence)
                        updated_sentence_num_to_sentence_text_2 = self.get_updated_sentence_num_to_sentence_text(related_sentence_num_to_sentence_text, token_index_to_sentence_num, token_index_to_index_in_sentence, clean_related_mentions)                   
                        last_sentence_num = sorted_sentence_num_list[len(sorted_sentence_num_list)-1]
                        end_token_index_offset_2 = self.get_end_token_offset(token_index_to_sentence_num, last_sentence_num, clean_related_mentions)
                
                        sentence_num_to_sentence_embedding_2 = {}  
                    
                        used_bit_focus, same_bit_focus = self.get_used_bit_and_same_bit(converted_focus_mention, focus_mention_cluster_id, cluster_id_to_used_mentions, cluster_id_to_last_mention)
                    
                        if len(mentions_to_replace) == 0:
                            for sentence_num in related_sentence_num_to_sentence_text:
                                if sentence_num in updated_sentence_num_to_sentence_text_2:
                                    sentence_text = updated_sentence_num_to_sentence_text_2[sentence_num]
                                    token_embeddings_for_sentence = self.get_bert_embedding(sentence_text)
                                    sentence_num_to_sentence_embedding_2[sentence_num] = token_embeddings_for_sentence
                                else:
                                    if sentence_num in original_sentence_num_to_embedding:
                                        token_embeddings_for_sentence = original_sentence_num_to_embedding[sentence_num]
                                        sentence_num_to_sentence_embedding_2[sentence_num] = token_embeddings_for_sentence
                                    else:
                                        sentence_text = related_sentence_num_to_sentence_text[sentence_num]
                                        token_embeddings_for_sentence = self.get_bert_embedding(sentence_text)
                                        sentence_num_to_sentence_embedding_2[sentence_num] = token_embeddings_for_sentence
                                        original_sentence_num_to_embedding[sentence_num] = token_embeddings_for_sentence
                            
                            previous_mention_cluster_ids = []
                            correct_previous_mentions = []
                            previous_mentions = []
                            previous_mention_index_in_sentence_list = []
                            previous_mention_sentence_num_list = []
                        
                            seql_focus = []
                            focus_seq_length = 51 + len(converted_focus_mention_parts)
            
                            for i in range(len(raw_sequence)):
                                word = raw_sequence[i]
                                if word == "<pad>":
                                    padding_embedding = [0] * 768
                                    padding_embedding= numpy.asarray(padding_embedding, dtype=numpy.float32)
                                    seql_focus.append(padding_embedding)
                                else:
                                    break
                            
                            ffocus_mention_sentence = ""
                            if focus_mention_sentence_num in updated_sentence_num_to_sentence_text_2:
                                ffocus_mention_sentence = updated_sentence_num_to_sentence_text_2[focus_mention_sentence_num]
                            else:
                                ffocus_mention_sentence = related_sentence_num_to_sentence_text[focus_mention_sentence_num]
                            uupdated_sentence_text = self.get_updated_sentence(ffocus_mention_sentence, focus_mention_index_in_sentence, converted_focus_mention, len(original_focus_mention_parts))
                            ttoken_embeddings_for_sentence = self.get_bert_embedding(uupdated_sentence_text)
                            sentence_num_to_sentence_embedding_2[focus_mention_sentence_num] = ttoken_embeddings_for_sentence
                            if focus_mention_sentence_num == last_sentence_num:
                                end_token_index_offset_2 = end_token_index_offset_2 + len(converted_focus_mention_parts) - len(original_focus_mention_parts)
                            temp_seql_focus = self.get_embedding_sequence_from_left(sorted_sentence_num_list, sentence_num_to_sentence_embedding_2, start_token_index_in_sentence, focus_mention_sentence_num, focus_mention_index_in_sentence, converted_focus_mention_parts)
                
                            seql_focus.extend(temp_seql_focus)
            
                            if max_seql_length < focus_seq_length:
                                max_seql_length = focus_seq_length
                                
                            seqr = []
                            for i in range(len(raw_sequence)-1, len(raw_sequence)-51, -1):
                                word = raw_sequence[i]
                                if word == "<pad>":
                                    padding_embedding = [0] * 768
                                    padding_embedding= numpy.asarray(padding_embedding, dtype=numpy.float32)
                                    seqr.append(padding_embedding)
                                else:
                                    break

                            temp_seqr = self.get_embedding_sequence_from_right(sorted_sentence_num_list, sentence_num_to_sentence_embedding_2, end_token_index_in_sentence+end_token_index_offset_2, focus_mention_sentence_num, focus_mention_index_in_sentence, converted_focus_mention_parts)                                
                            seqr.extend(temp_seqr)
                        
                            seqr_length = len(seqr)
                            if max_seqr_length < seqr_length:
                                max_seqr_length = seqr_length
                        
                            if converted_focus_mention.lower() not in mention_set:
                                no_correct_string_in_se = True
                            
                            for i in range(len(mention_set)):
                                mention = mention_set[i]

                                if mention.lower() != converted_focus_mention.lower():                                
                                    seql_other = []
                                    mention_parts = mention.split()
                                    other_seq_length = 51 + len(mention_parts)
                    
                                    focus_mention_sentence = ""
                                    if focus_mention_sentence_num in updated_sentence_num_to_sentence_text_2:
                                        focus_mention_sentence = updated_sentence_num_to_sentence_text_2[focus_mention_sentence_num]
                                    else:
                                        focus_mention_sentence = related_sentence_num_to_sentence_text[focus_mention_sentence_num]
                                    updated_sentence_text = self.get_updated_sentence(focus_mention_sentence, focus_mention_index_in_sentence, mention, len(original_focus_mention_parts))
                                    token_embeddings_for_sentence = self.get_bert_embedding(updated_sentence_text)
                                    sentence_num_to_sentence_embedding_2[focus_mention_sentence_num] = token_embeddings_for_sentence
                            
                                    for j in range(len(raw_sequence)):
                                        word = raw_sequence[j]
                                        if word == "<pad>":
                                            padding_embedding = [0] * 768
                                            padding_embedding= numpy.asarray(padding_embedding, dtype=numpy.float32)
                                            seql_other.append(padding_embedding)
                                        else:
                                            break
                    
                                    temp_seql_other = self.get_embedding_sequence_from_left(sorted_sentence_num_list, sentence_num_to_sentence_embedding_2, start_token_index_in_sentence, focus_mention_sentence_num, focus_mention_index_in_sentence, mention_parts)
                        
                                    seql_other.extend(temp_seql_other)
                                
                                    if max_seql_length < other_seq_length:
                                        max_seql_length = other_seq_length
                                
                                    seql_focus_copy = seql_focus.copy()
                                    seqr_copy = seqr.copy()
                                    seql_1_list.append(seql_focus_copy)
                                    seql_2_list.append(seql_other)
                                    seqr_list.append(seqr_copy)
                                
                                    first_mention_bit_seql_list.append(first_mention_bit)
                                    second_mention_bit_seql_list.append(second_mention_bit)
                                
                                    focus_mention_length_list.append(len(converted_focus_mention_parts))
                                    other_mention_length_list.append(len(mention_parts))
                                    coreference_chain_length_list.append(focus_mention_cluster_total_mention)
                                
                                    used_bit_other, same_bit_other = self.get_used_bit_and_same_bit(mention, focus_mention_cluster_id, cluster_id_to_used_mentions, cluster_id_to_last_mention)
                        
                                    used_bit_seql_1_list.append(used_bit_focus)
                                    same_bit_seql_1_list.append(same_bit_focus)
                                    used_bit_seql_2_list.append(used_bit_other)
                                    same_bit_seql_2_list.append(same_bit_other)
                                    
                                    is_subject_list.append(is_subject)
                                    is_object_list.append(is_object)
                                    label_list.append(1)
                        else:
                            temp_dict_sorted_keys = sorted(temp_dict.keys(), reverse=True)
                            x = 0
                            sorted_key_to_replaced_mention = {}
                            sorted_key_to_replaced_mention_cluster_id = {}
                            sorted_key_to_previous_mention_index = {}
                            for sorted_key in temp_dict_sorted_keys:
                                replaced_mention = previous_mentions[len(previous_mentions) - 1 - x]
                                replaced_mention_cluster_id = previous_mention_cluster_ids[len(previous_mention_cluster_ids) - 1 - x]
                                sorted_key_to_replaced_mention[sorted_key] = replaced_mention
                                sorted_key_to_replaced_mention_cluster_id[sorted_key] = replaced_mention_cluster_id
                                sorted_key_to_previous_mention_index[sorted_key] = len(previous_mentions) - 1 - x
                                x = x + 1

                            updated_focus_mention_index_in_sentence = focus_mention_index_in_sentence
                            for sorted_key in temp_dict_sorted_keys:
                                previous_mention_index = sorted_key_to_previous_mention_index[sorted_key]
                                previous_mention = previous_mentions[previous_mention_index]
                                correct_previous_mention = correct_previous_mentions[previous_mention_index]
                                correct_previous_mention_parts = correct_previous_mention.split()
                                previous_mention_index_in_sentence = previous_mention_index_in_sentence_list[previous_mention_index]
                                previous_mention_sentence_num = previous_mention_sentence_num_list[previous_mention_index]
                                previous_mention_sentence_text = ""
                                if previous_mention_sentence_num in updated_sentence_num_to_sentence_text_2:
                                    previous_mention_sentence_text = updated_sentence_num_to_sentence_text_2[previous_mention_sentence_num]
                                else:
                                    previous_mention_sentence_text = related_sentence_num_to_sentence_text[previous_mention_sentence_num]
                                
                                previous_mention_cluster_id = previous_mention_cluster_ids[previous_mention_index]
                                if (previous_mention_cluster_id == focus_mention_cluster_id) or (previous_mention_cluster_id in cluster_id_to_he_clusters and focus_mention_cluster_id in cluster_id_to_he_clusters) or (previous_mention_cluster_id in cluster_id_to_she_clusters and focus_mention_cluster_id in cluster_id_to_she_clusters) or (previous_mention_cluster_id in cluster_id_to_it_clusters and focus_mention_cluster_id in cluster_id_to_it_clusters) or (previous_mention_cluster_id in cluster_id_to_they_clusters):
                                    updated_sentence_text = self.get_updated_sentence(previous_mention_sentence_text, previous_mention_index_in_sentence, previous_mention, len(correct_previous_mention_parts))
                                    updated_sentence_num_to_sentence_text_2[previous_mention_sentence_num] = updated_sentence_text
                            
                                    if previous_mention_sentence_num == focus_mention_sentence_num:
                                        focus_mention_offset = self.get_focus_mention_offset(previous_mention_index_in_sentence, previous_mention, correct_previous_mention, focus_mention_index_in_sentence)
                                        updated_focus_mention_index_in_sentence = updated_focus_mention_index_in_sentence + focus_mention_offset
                                        
                            all_updated_sentence_num_to_sentence_text_2 = {}
                            for sentence_num in related_sentence_num_to_sentence_text:
                                if sentence_num in updated_sentence_num_to_sentence_text_2:
                                    sentence_text = updated_sentence_num_to_sentence_text_2[sentence_num]
                                    token_embeddings_for_sentence = self.get_bert_embedding(sentence_text)
                                    sentence_num_to_sentence_embedding_2[sentence_num] = token_embeddings_for_sentence
                                    all_updated_sentence_num_to_sentence_text_2[sentence_num] = sentence_text
                                else:
                                    all_updated_sentence_num_to_sentence_text_2[sentence_num] = related_sentence_num_to_sentence_text[sentence_num]
                                    if sentence_num in original_sentence_num_to_embedding:
                                        token_embeddings_for_sentence = original_sentence_num_to_embedding[sentence_num]
                                        sentence_num_to_sentence_embedding_2[sentence_num] = token_embeddings_for_sentence
                                    else:
                                        sentence_text = related_sentence_num_to_sentence_text[sentence_num]
                                        token_embeddings_for_sentence = self.get_bert_embedding(sentence_text)
                                        sentence_num_to_sentence_embedding_2[sentence_num] = token_embeddings_for_sentence
                                        original_sentence_num_to_embedding[sentence_num] = token_embeddings_for_sentence
                            
                            seql_focus = []
                            focus_seq_length = 51 + len(converted_focus_mention_parts)
            
                            updated_from_sentence, updated_from_token, updated_pre_padding = self.get_from_info(all_updated_sentence_num_to_sentence_text_2, focus_mention_sentence_num, updated_focus_mention_index_in_sentence)
                        
                            for i in range(updated_pre_padding):
                                padding_embedding = [0] * 768
                                padding_embedding= numpy.asarray(padding_embedding, dtype=numpy.float32)
                                seql_focus.append(padding_embedding)
                    
                            ffocus_mention_sentence = all_updated_sentence_num_to_sentence_text_2[focus_mention_sentence_num]
                            uupdated_sentence_text = self.get_updated_sentence(ffocus_mention_sentence, updated_focus_mention_index_in_sentence, converted_focus_mention, len(original_focus_mention_parts))
                            all_updated_sentence_num_to_sentence_text_2[focus_mention_sentence_num] = uupdated_sentence_text
                            ttoken_embeddings_for_sentence = self.get_bert_embedding(uupdated_sentence_text)
                            sentence_num_to_sentence_embedding_2[focus_mention_sentence_num] = ttoken_embeddings_for_sentence                            
                            temp_seql_focus = self.get_embedding_sequence_from_left(sorted_sentence_num_list, sentence_num_to_sentence_embedding_2, updated_from_token, focus_mention_sentence_num, updated_focus_mention_index_in_sentence, converted_focus_mention_parts, updated_from_sentence)
                
                            seql_focus.extend(temp_seql_focus)
                            
                            if max_seql_length < focus_seq_length:
                                max_seql_length = focus_seq_length
                                
                            seqr = []
                            updated_to_sentence, updated_to_token, updated_post_padding = self.get_to_info(all_updated_sentence_num_to_sentence_text_2, focus_mention_sentence_num, updated_focus_mention_index_in_sentence, converted_focus_mention)
                        
                            for i in range(updated_post_padding):
                                padding_embedding = [0] * 768
                                padding_embedding= numpy.asarray(padding_embedding, dtype=numpy.float32)
                                seqr.append(padding_embedding)
                         
                            temp_seqr = self.get_embedding_sequence_from_right(sorted_sentence_num_list, sentence_num_to_sentence_embedding_2, updated_to_token, focus_mention_sentence_num, updated_focus_mention_index_in_sentence, converted_focus_mention_parts, updated_to_sentence)                                
                            seqr.extend(temp_seqr)
                        
                            seqr_length = len(seqr)
                            if max_seqr_length < seqr_length:
                                max_seqr_length = seqr_length

                            if converted_focus_mention.lower() not in mention_set:
                                no_correct_string_in_se = True
                            
                            for i in range(len(mention_set)):
                                mention = mention_set[i]

                                if mention.lower() != converted_focus_mention.lower():                                
                                    seql_other = []
                                    mention_parts = mention.split()
                                    other_seq_length = 51 + len(mention_parts)
                    
                                    focus_mention_sentence = all_updated_sentence_num_to_sentence_text_2[focus_mention_sentence_num]                                           
                                    updated_sentence_text = self.get_updated_sentence(focus_mention_sentence, updated_focus_mention_index_in_sentence, mention, len(converted_focus_mention_parts))
                                    token_embeddings_for_sentence = self.get_bert_embedding(updated_sentence_text)
                                    sentence_num_to_sentence_embedding_2[focus_mention_sentence_num] = token_embeddings_for_sentence
                                
                                    for j in range(updated_pre_padding):
                                        padding_embedding = [0] * 768
                                        padding_embedding= numpy.asarray(padding_embedding, dtype=numpy.float32)
                                        seql_other.append(padding_embedding)
                            
                                    temp_seql_other = self.get_embedding_sequence_from_left(sorted_sentence_num_list, sentence_num_to_sentence_embedding_2, updated_from_token, focus_mention_sentence_num, updated_focus_mention_index_in_sentence, mention_parts, updated_from_sentence)                                
                        
                                    seql_other.extend(temp_seql_other)
                                    
                                    if max_seql_length < other_seq_length:
                                        max_seql_length = other_seq_length
                                        
                                    seql_focus_copy = seql_focus.copy()
                                    seqr_copy = seqr.copy()
                                    seql_1_list.append(seql_focus_copy)
                                    seql_2_list.append(seql_other)
                                    seqr_list.append(seqr_copy)
                                
                                    first_mention_bit_seql_list.append(first_mention_bit)
                                    second_mention_bit_seql_list.append(second_mention_bit)
                                
                                    focus_mention_length_list.append(len(converted_focus_mention_parts))
                                    other_mention_length_list.append(len(mention_parts))
                                    coreference_chain_length_list.append(focus_mention_cluster_total_mention)
                                
                                    used_bit_other, same_bit_other = self.get_used_bit_and_same_bit(mention, focus_mention_cluster_id, cluster_id_to_used_mentions, cluster_id_to_last_mention)
                        
                                    used_bit_seql_1_list.append(used_bit_focus)
                                    same_bit_seql_1_list.append(same_bit_focus)
                                    used_bit_seql_2_list.append(used_bit_other)
                                    same_bit_seql_2_list.append(same_bit_other)
                                    
                                    is_subject_list.append(is_subject)
                                    is_object_list.append(is_object)
                                    label_list.append(1)                        
                            
                        batch_r, batch_len_r, batch_first_mention_bit, batch_second_mention_bit, batch_used_bit_seql_1, batch_same_bit_seql_1, batch_used_bit_seql_2, batch_same_bit_seql_2, batch_is_subject_list, batch_is_object_list, batch_label = \
                        self.create_padding([seqr_list, first_mention_bit_seql_list, second_mention_bit_seql_list, used_bit_seql_1_list, same_bit_seql_1_list, used_bit_seql_2_list, same_bit_seql_2_list, is_subject_list, is_object_list, label_list], max_seqr_length)

                        rep_focus_mention_length_array = self.get_mention_distance_embedding(focus_mention_length_list)
                        rep_other_mention_length_array = self.get_mention_distance_embedding(other_mention_length_list)
                        rep_coreference_chain_length_array = self.get_coreference_chain_distance_embedding(coreference_chain_length_list)

                        rep_r = Variable(torch.from_numpy(batch_r),
                                     requires_grad=False).type(self.float_type)
                        len_r = Variable(torch.from_numpy(batch_len_r),
                                     requires_grad=False).type(self.long_type)
                        rep_first_mention_bit = Variable(torch.from_numpy(batch_first_mention_bit),
                                     requires_grad=False).type(self.float_type)
                        rep_second_mention_bit = Variable(torch.from_numpy(batch_second_mention_bit),
                                     requires_grad=False).type(self.float_type)
                        rep_focus_mention_length = Variable(torch.from_numpy(rep_focus_mention_length_array),
                                     requires_grad=False).type(self.float_type)
                        rep_other_mention_length = Variable(torch.from_numpy(rep_other_mention_length_array),
                                     requires_grad=False).type(self.float_type)
                        rep_coreference_chain_length = Variable(torch.from_numpy(rep_coreference_chain_length_array),
                                     requires_grad=False).type(self.float_type)
                        rep_used_bit_seql_1 = Variable(torch.from_numpy(batch_used_bit_seql_1),
                                     requires_grad=False).type(self.float_type)
                        rep_same_bit_seql_1 = Variable(torch.from_numpy(batch_same_bit_seql_1),
                                     requires_grad=False).type(self.float_type)
                        rep_used_bit_seql_2 = Variable(torch.from_numpy(batch_used_bit_seql_2),
                                     requires_grad=False).type(self.float_type)
                        rep_same_bit_seql_2 = Variable(torch.from_numpy(batch_same_bit_seql_2),
                                     requires_grad=False).type(self.float_type)

                        rep_is_subject_list = Variable(torch.from_numpy(batch_is_subject_list),
                                     requires_grad=False).type(self.float_type)
                        rep_is_object_list = Variable(torch.from_numpy(batch_is_object_list),
                                     requires_grad=False).type(self.float_type)
                        label = Variable(torch.from_numpy(batch_label),
                                     requires_grad=False).type(self.float_type)
                            
                        # Forward pass: predict scores
                        score_l1, score_l2 = model(seql_1_list, seql_2_list, max_seql_length, rep_r, len_r, rep_first_mention_bit, rep_second_mention_bit, rep_focus_mention_length, rep_other_mention_length, rep_coreference_chain_length, rep_used_bit_seql_1, rep_same_bit_seql_1, rep_used_bit_seql_2, rep_same_bit_seql_2, rep_is_subject_list, rep_is_object_list)
                        loss = NN.MarginRankingLoss(margin=margin)(score_l1, score_l2, label) 
                        valid_losses.append(loss.item())
                      
                        larger_score = 0
                        if self.use_gpu:
                            s2 = score_l2.cpu()
                            s1 = score_l1.cpu()                            
                        s2 = s2.data.numpy()
                        s1 = s1.data.numpy()
                        max_index = -1
                        max_2_index = -1
                        max_value = -10000000
                        max_2_value = -10000000
                        for ii in range(len(s1)):
                            score_1 = s1[ii]
                            score_2 = s2[ii]
                            if score_2 > max_2_value:
                                max_2_value = score_2
                                max_2_index = ii
                            if score_2 > score_1 and score_2 > max_value:
                                max_value = score_2
                                max_index = ii
                                larger_score = 2
                            elif score_1 >= score_2 and score_1 > max_value:
                                max_value = score_1
                                max_index = ii
                                larger_score = 1
                
                        chosen_mention = ""
                        if larger_score == 1:
                            previous_mentions.append(converted_focus_mention)
                            previous_mention_cluster_ids.append(focus_mention_cluster_id)
                            correct_previous_mentions.append(original_focus_mention)
                            previous_mention_index_in_sentence_list.append(focus_mention_index_in_sentence)
                            previous_mention_sentence_num_list.append(focus_mention_sentence_num)
                            chosen_mention = converted_focus_mention 
                        else:                  
                            INDEX = 0
                            if no_correct_string_in_se:
                                chosen_mention = mention_set[max_2_index]
                                previous_mentions.append(chosen_mention)
                                previous_mention_cluster_ids.append(focus_mention_cluster_id)
                                correct_previous_mentions.append(original_focus_mention)
                                previous_mention_index_in_sentence_list.append(focus_mention_index_in_sentence)
                                previous_mention_sentence_num_list.append(focus_mention_sentence_num)
                            else:
                                for mention in mention_set:
                                    if mention.lower() != converted_focus_mention.lower():
                                        if max_index == INDEX:
                                            previous_mentions.append(mention)
                                            previous_mention_cluster_ids.append(focus_mention_cluster_id)
                                            correct_previous_mentions.append(original_focus_mention)
                                            previous_mention_index_in_sentence_list.append(focus_mention_index_in_sentence)
                                            previous_mention_sentence_num_list.append(focus_mention_sentence_num)
                                            chosen_mention = mention
                                            break
                                        INDEX = INDEX + 1
                            
                        if gold_setting:
                            total_size = total_size + 1
                            current_total_size = current_total_size + 1
                            
                            if focus_mention_cluster_id != -1:
                                current_total_confounding_mentions = current_total_confounding_mentions + 1
                                    
                            if focus_mention_cluster_id not in appeared_focus_mention_cluster_id_list:
                                appeared_focus_mention_cluster_id_list.append(focus_mention_cluster_id)
                            if chosen_mention.lower() == converted_focus_mention.lower():
                                total_correct = total_correct + 1
                                current_total_correct = current_total_correct + 1
                            elif no_correct_string_in_se:
                                total_se_error = total_se_error + 1
                                current_total_se_error = current_total_se_error + 1
                        else:
                            annotated_original_mention = raw_sequence_data._annotated_original_focus_mention
                            identified_original_mention = raw_sequence_data._identified_original_focus_mention
                            if is_identified_mention == 1 and is_annotated_mention == 1:
                                total_annotated_mention = total_annotated_mention + 1
                                current_total_annotated_mention = current_total_annotated_mention + 1
                            
                                if annotated_original_mention.lower() == identified_original_mention.lower():
                                    total_annotated_mention_correctly_identified = total_annotated_mention_correctly_identified + 1
                                    current_total_annotated_mention_correctly_identified = current_total_annotated_mention_correctly_identified + 1
                                    if chosen_mention.lower() == converted_focus_mention.lower():
                                        total_annotated_mention_correctly_identified_correctly_converted = total_annotated_mention_correctly_identified_correctly_converted + 1
                                        current_total_annotated_mention_correctly_identified_correctly_converted = current_total_annotated_mention_correctly_identified_correctly_converted + 1
                                    else:
                                        if no_correct_string_in_se:
                                            total_se_error = total_se_error + 1
                                            current_total_se_error = current_total_se_error + 1
                                else:
                                    total_annotated_mention_incorrectly_identified = total_annotated_mention_incorrectly_identified + 1
                                    current_total_annotated_mention_incorrectly_identified = current_total_annotated_mention_incorrectly_identified + 1
                                    if (annotated_original_mention.lower() != converted_focus_mention.lower()) or (identified_original_mention.lower() != chosen_mention.lower()):
                                        total_annotated_mention_incorrectly_identified_incorrectly_converted = total_annotated_mention_incorrectly_identified_incorrectly_converted + 1
                                        current_total_annotated_mention_incorrectly_identified_incorrectly_converted = current_total_annotated_mention_incorrectly_identified_incorrectly_converted + 1
                            elif is_identified_mention == 1 and is_annotated_mention == 0:
                                total_identified_mention_not_annotated = total_identified_mention_not_annotated + 1
                                current_total_identified_mention_not_annotated = current_total_identified_mention_not_annotated + 1
                                if chosen_mention.lower() != identified_original_mention.lower():
                                    total_identified_mention_not_annotated_incorrectly_converted = total_identified_mention_not_annotated_incorrectly_converted + 1
                                    current_total_identified_mention_not_annotated_incorrectly_converted = current_total_identified_mention_not_annotated_incorrectly_converted + 1
                        if write_result:
                            if chosen_mention.lower() == converted_focus_mention.lower():
                                f.write('correct choice: ')
                            else:
                                f.write('wrong choice: ')
                            f.write('\n')
                            for x in range(50):
                                f.write("%s " %raw_sequence[x])
                            f.write(chosen_mention)
                            for x in range(50+len(original_focus_mention_parts), len(raw_sequence)):
                                f.write(" %s" %raw_sequence[x])
                            f.write("\n")
                                
                        if focus_mention_cluster_id not in cluster_id_to_used_mentions:
                            temp = []
                            temp.append(chosen_mention.lower())
                            cluster_id_to_used_mentions[focus_mention_cluster_id] = temp
                        else:
                            temp = cluster_id_to_used_mentions[focus_mention_cluster_id]
                            if chosen_mention.lower() not in temp:
                                temp.append(chosen_mention.lower())
                                cluster_id_to_used_mentions[focus_mention_cluster_id] = temp
                        if focus_mention_cluster_id not in cluster_id_to_last_mention:
                            cluster_id_to_last_mention[focus_mention_cluster_id] = chosen_mention.lower()
                        else:
                            temp = cluster_id_to_last_mention[focus_mention_cluster_id]
                            if temp != chosen_mention.lower():
                                cluster_id_to_last_mention[focus_mention_cluster_id] = chosen_mention.lower()
                    
                    total_entity = total_entity + len(appeared_focus_mention_cluster_id_list)
                    
                    f_temp.write(data_path)
                    f_temp.write("\n")
                    
                    if gold_setting:
                        f_temp.write("current total correct: ")
                        f_temp.write(str(current_total_correct))
                        f_temp.write('\t')
                        f_temp.write("current total size: ")
                        f_temp.write(str(current_total_size))
                        f_temp.write('\t')
                    f_temp.write("current total annotated mention correctly identified: ")
                    f_temp.write(str(current_total_annotated_mention_correctly_identified))
                    f_temp.write('\t')
                    f_temp.write("current total annotated mention correctly identified correctly converted: ")
                    f_temp.write(str(current_total_annotated_mention_correctly_identified_correctly_converted))
                    f_temp.write('\t')
                    f_temp.write("current total annotated mention incorrectly identified: ")
                    f_temp.write(str(current_total_annotated_mention_incorrectly_identified))
                    f_temp.write('\t')
                    f_temp.write("current total annotated mention not identified: ")
                    f_temp.write(str(current_total_annotated_mention_not_identified_incorrectly_converted))
                    f_temp.write('\t')
                    f_temp.write("current total identified mention not annotated: ")
                    f_temp.write(str(current_total_identified_mention_not_annotated))
                    f_temp.write('\t')
                    f_temp.write("current total annotated verb: ")
                    f_temp.write(str(current_total_annotated_verb))
                    f_temp.write('\t')
                    f_temp.write("current total annotated verb correctly identified: ")
                    f_temp.write(str(current_total_annotated_verb_correctly_identified))
                    f_temp.write('\t')
                    f_temp.write("current total annotated verb correctly identified correctly converted: ")
                    f_temp.write(str(current_total_annotated_verb_correctly_identified_correctly_converted))
                    f_temp.write('\t')
                    f_temp.write("current total annotated verb not identified: ")
                    f_temp.write(str(current_total_annotated_verb_not_identified))
                    f_temp.write('\t')
                    f_temp.write("current total identified verb not annotated: ")
                    f_temp.write(str(current_total_identified_verb_not_annotated))
                    f_temp.write('\n')
        
        return valid_losses, total_size, total_correct, total_annotated_mention, total_annotated_mention_correctly_identified, total_annotated_mention_correctly_identified_correctly_converted, total_annotated_mention_incorrectly_identified, total_annotated_mention_incorrectly_identified_incorrectly_converted, total_annotated_mention_not_identified, total_annotated_mention_not_identified_incorrectly_converted, total_identified_mention_not_annotated, total_identified_mention_not_annotated_incorrectly_converted, total_annotated_verb, total_annotated_verb_correctly_identified, total_annotated_verb_correctly_identified_correctly_converted, total_annotated_verb_not_identified, total_annotated_verb_not_identified_incorrectly_converted, total_identified_verb_not_annotated, total_identified_verb_not_annotated_incorrectly_converted, total_se_error
     

    def evaluate(self, model, FLAGS):
        start_time = time.time()
        
        margin = FLAGS.margin

        development_data_dir = FLAGS.development_data_dir
        pov_gold_testing_data_dir = FLAGS.pov_gold_testing_data_dir
        pov_auto_dev_data_dir = FLAGS.pov_auto_dev_data_dir
        pov_auto_testing_data_dir = FLAGS.pov_auto_testing_data_dir

        output_file = 'result_f_m1_dev_conll_gold_test_three' + ".txt"
        f_output = open(output_file, 'w+', encoding='utf-8')
        f_temp_gold = open('temp_gold.txt', 'w+', encoding='utf-8')
        f_temp_test = open('temp_test.txt', 'w+', encoding='utf-8')
        f_temp_dev = open('temp_dev.txt', 'w+', encoding='utf-8')
        file_name_gold = 'output_f_m1_dev_conll_gold_test_pov_gold.txt'
        f_gold = open(file_name_gold, 'w+', encoding='utf-8')
        file_name_dev = 'output_f_m1_dev_conll_gold_test_pov_dev.txt'
        f_dev = open(file_name_dev, 'w+', encoding='utf-8')
        file_name_test = 'output_f_m1_dev_conll_gold_test_pov_test.txt'
        f_test = open(file_name_test, 'w+', encoding='utf-8')
            
#        valid_losses, total_correct_dev, total_size_dev = self.get_model_development_data_scenario2(development_data_dir, model, margin)
#        valid_loss = numpy.average(valid_losses)       
#        print_msg = (f'valid_loss: {valid_loss:.5f}')
#        print(print_msg)                     
#        valid_losses = []
#                
#        print ('Accuracy of the trained model on validation set %f' % (total_correct_dev / total_size_dev))
#        print('dev size: ')
#        print(total_size_dev)
#        print('correct size: ')
#        print(total_correct_dev)
        
        valid_losses, total_size, total_correct, total_annotated_mention, total_annotated_mention_correctly_identified, total_annotated_mention_correctly_identified_correctly_converted, total_annotated_mention_incorrectly_identified, total_annotated_mention_incorrectly_identified_incorrectly_converted, total_annotated_mention_not_identified, total_annotated_mention_not_identified_incorrectly_converted, total_identified_mention_not_annotated, total_identified_mention_not_annotated_incorrectly_converted, total_annotated_verb, total_annotated_verb_correctly_identified, total_annotated_verb_correctly_identified_correctly_converted, total_annotated_verb_not_identified, total_annotated_verb_not_identified_incorrectly_converted, total_identified_verb_not_annotated, total_identified_verb_not_annotated_incorrectly_converted, total_se_error =self.get_model_testing_data_scenario2(f_gold, f_temp_gold, pov_gold_testing_data_dir, model, margin, True, True)
        valid_losses = []
        print('Results on pov data - gold setting: ')                
        print("total correct: ")
        print(total_correct)
        print("total size: ")
        print(total_size)
        print("total annotated mention correctly identified: ")
        print(total_annotated_mention_correctly_identified)
        print("total annotated mention correctly identified correctly converted: ")
        print(total_annotated_mention_correctly_identified_correctly_converted)
        print("total annotated mention incorrectly identified: ")
        print(total_annotated_mention_incorrectly_identified)
        print("total annotated mention not identified: ")
        print(total_annotated_mention_not_identified_incorrectly_converted)
        print("total identified mention not annotated: ")
        print(total_identified_mention_not_annotated)
        print("total annoated verb: ")
        print(total_annotated_verb)
        print("total annotated verb correctly identified: ")
        print(total_annotated_verb_correctly_identified)
        print("total annotated verb correctly identified correctly converted: ")
        print(total_annotated_verb_correctly_identified_correctly_converted)
        print("total annotated verb not identified: ")
        print(total_annotated_verb_not_identified)
        print("total identified verb not annotated: ")
        print(total_identified_verb_not_annotated)


        f_output.write('Results on pov data - gold setting: ')
        f_output.write("\n")
        f_output.write("total correct: ")
        f_output.write(str(total_correct))
        f_output.write('\t')
        f_output.write("total size: ")
        f_output.write(str(total_size))
        f_output.write('\t')
        f_output.write("total annotated mention correctly identified: ")
        f_output.write(str(total_annotated_mention_correctly_identified))
        f_output.write('\t')
        f_output.write("total annotated mention correctly identified correctly converted: ")
        f_output.write(str(total_annotated_mention_correctly_identified_correctly_converted))
        f_output.write('\t')
        f_output.write("total annotated mention incorrectly identified: ")
        f_output.write(str(total_annotated_mention_incorrectly_identified))
        f_output.write('\t')
        f_output.write("total annotated mention not identified: ")
        f_output.write(str(total_annotated_mention_not_identified_incorrectly_converted))
        f_output.write('\t')
        f_output.write("total identified mention not annotated: ")
        f_output.write(str(total_identified_mention_not_annotated))
        f_output.write('\t')
        f_output.write("total annotated verb: ")
        f_output.write(str(total_annotated_verb))
        f_output.write('\t')
        f_output.write("total annotated verb correctly identified: ")
        f_output.write(str(total_annotated_verb_correctly_identified))
        f_output.write('\t')
        f_output.write("total annotated verb correctly identified correctly converted: ")
        f_output.write(str(total_annotated_verb_correctly_identified_correctly_converted))
        f_output.write('\t')
        f_output.write("total annotated verb not identified: ")
        f_output.write(str(total_annotated_verb_not_identified))
        f_output.write('\t')
        f_output.write("total identified verb not annotated: ")
        f_output.write(str(total_identified_verb_not_annotated))
        f_output.write('\n')
#            
        valid_losses, total_size, total_correct, total_annotated_mention, total_annotated_mention_correctly_identified, total_annotated_mention_correctly_identified_correctly_converted, total_annotated_mention_incorrectly_identified, total_annotated_mention_incorrectly_identified_incorrectly_converted, total_annotated_mention_not_identified, total_annotated_mention_not_identified_incorrectly_converted, total_identified_mention_not_annotated, total_identified_mention_not_annotated_incorrectly_converted, total_annotated_verb, total_annotated_verb_correctly_identified, total_annotated_verb_correctly_identified_correctly_converted, total_annotated_verb_not_identified, total_annotated_verb_not_identified_incorrectly_converted, total_identified_verb_not_annotated, total_identified_verb_not_annotated_incorrectly_converted, total_se_error=self.get_model_testing_data_scenario2(f_dev, f_temp_dev, pov_auto_dev_data_dir, model, margin, False, True)
        valid_losses = []
            
        correct_mention = total_annotated_mention_correctly_identified_correctly_converted
        correct_verb = total_annotated_verb_correctly_identified_correctly_converted
        recall = 1.0 * (correct_mention + correct_verb) / (total_annotated_mention_correctly_identified + total_annotated_mention_incorrectly_identified + total_annotated_mention_not_identified_incorrectly_converted + total_annotated_verb)
        precision = 1.0 * (correct_mention + correct_verb) / (total_annotated_mention_correctly_identified + total_annotated_mention_incorrectly_identified + total_identified_mention_not_annotated + total_annotated_verb_correctly_identified + total_identified_verb_not_annotated)
        F1 = 2*recall*precision/(recall+precision)
        
        print('Results on pov data - dev: ')   
        print('Recall: ')
        print(recall)
        print('Precision: ')
        print(precision)
        print('F1 score: ')
        print(F1)             
        
        print("total annotated mention correctly identified: ")
        print(total_annotated_mention_correctly_identified)
        print("total annotated mention correctly identified correctly converted: ")
        print(total_annotated_mention_correctly_identified_correctly_converted)
        print("total annotated mention incorrectly identified: ")
        print(total_annotated_mention_incorrectly_identified)
        print("total annotated mention not identified: ")
        print(total_annotated_mention_not_identified_incorrectly_converted)
        print("total identified mention not annotated: ")
        print(total_identified_mention_not_annotated)
        print("total annoated verb: ")
        print(total_annotated_verb)
        print("total annotated verb correctly identified: ")
        print(total_annotated_verb_correctly_identified)
        print("total annotated verb correctly identified correctly converted: ")
        print(total_annotated_verb_correctly_identified_correctly_converted)
        print("total annotated verb not identified: ")
        print(total_annotated_verb_not_identified)
        print("total identified verb not annotated: ")
        print(total_identified_verb_not_annotated)
            
        f_output.write('Results on pov data - dev: ')
        f_output.write("\n")
        f_output.write("total annotated mention correctly identified: ")
        f_output.write(str(total_annotated_mention_correctly_identified))
        f_output.write('\t')
        f_output.write("total annotated mention correctly identified correctly converted: ")
        f_output.write(str(total_annotated_mention_correctly_identified_correctly_converted))
        f_output.write('\t')
        f_output.write("total annotated mention incorrectly identified: ")
        f_output.write(str(total_annotated_mention_incorrectly_identified))
        f_output.write('\t')
        f_output.write("total annotated mention not identified: ")
        f_output.write(str(total_annotated_mention_not_identified_incorrectly_converted))
        f_output.write('\t')
        f_output.write("total identified mention not annotated: ")
        f_output.write(str(total_identified_mention_not_annotated))
        f_output.write('\t')
        f_output.write("total annoated verb: ")
        f_output.write(str(total_annotated_verb))
        f_output.write('\t')
        f_output.write("total annotated verb correctly identified: ")
        f_output.write(str(total_annotated_verb_correctly_identified))
        f_output.write('\t')
        f_output.write("total annotated verb correctly identified correctly converted: ")
        f_output.write(str(total_annotated_verb_correctly_identified_correctly_converted))
        f_output.write('\t')
        f_output.write("total annotated verb not identified: ")
        f_output.write(str(total_annotated_verb_not_identified))
        f_output.write('\t')
        f_output.write("total identified verb not annotated: ")
        f_output.write(str(total_identified_verb_not_annotated))
        f_output.write('\n')
#            
#            
        valid_losses, total_size, total_correct, total_annotated_mention, total_annotated_mention_correctly_identified, total_annotated_mention_correctly_identified_correctly_converted, total_annotated_mention_incorrectly_identified, total_annotated_mention_incorrectly_identified_incorrectly_converted, total_annotated_mention_not_identified, total_annotated_mention_not_identified_incorrectly_converted, total_identified_mention_not_annotated, total_identified_mention_not_annotated_incorrectly_converted, total_annotated_verb, total_annotated_verb_correctly_identified, total_annotated_verb_correctly_identified_correctly_converted, total_annotated_verb_not_identified, total_annotated_verb_not_identified_incorrectly_converted, total_identified_verb_not_annotated, total_identified_verb_not_annotated_incorrectly_converted, total_se_error=self.get_model_testing_data_scenario2(f_test, f_temp_test, pov_auto_testing_data_dir, model, margin, False, True)
        valid_losses = []
          
        correct_mention = total_annotated_mention_correctly_identified_correctly_converted
        correct_verb = total_annotated_verb_correctly_identified_correctly_converted
        recall = 1.0 * (correct_mention + correct_verb) / (total_annotated_mention_correctly_identified + total_annotated_mention_incorrectly_identified + total_annotated_mention_not_identified_incorrectly_converted + total_annotated_verb)
        precision = 1.0 * (correct_mention + correct_verb) / (total_annotated_mention_correctly_identified + total_annotated_mention_incorrectly_identified + total_identified_mention_not_annotated + total_annotated_verb_correctly_identified + total_identified_verb_not_annotated)
        F1 = 2*recall*precision/(recall+precision)

        print('Results on pov data - test: ')   
        print('Recall: ')
        print(recall)
        print('Precision: ')
        print(precision)
        print('F1 score: ')
        print(F1)  
        
        print("total annotated mention correctly identified: ")
        print(total_annotated_mention_correctly_identified)
        print("total annotated mention correctly identified correctly converted: ")
        print(total_annotated_mention_correctly_identified_correctly_converted)
        print("total annotated mention incorrectly identified: ")
        print(total_annotated_mention_incorrectly_identified)
        print("total annotated mention not identified: ")
        print(total_annotated_mention_not_identified_incorrectly_converted)
        print("total identified mention not annotated: ")
        print(total_identified_mention_not_annotated)
        print("total annoated verb: ")
        print(total_annotated_verb)
        print("total annotated verb correctly identified: ")
        print(total_annotated_verb_correctly_identified)
        print("total annotated verb correctly identified correctly converted: ")
        print(total_annotated_verb_correctly_identified_correctly_converted)
        print("total annotated verb not identified: ")
        print(total_annotated_verb_not_identified)
        print("total identified verb not annotated: ")
        print(total_identified_verb_not_annotated)
            
        f_output.write('Results on pov data - dev: ')
        f_output.write("\n")
        f_output.write("total annotated mention correctly identified: ")
        f_output.write(str(total_annotated_mention_correctly_identified))
        f_output.write('\t')
        f_output.write("total annotated mention correctly identified correctly converted: ")
        f_output.write(str(total_annotated_mention_correctly_identified_correctly_converted))
        f_output.write('\t')
        f_output.write("total annotated mention incorrectly identified: ")
        f_output.write(str(total_annotated_mention_incorrectly_identified))
        f_output.write('\t')
        f_output.write("total annotated mention not identified: ")
        f_output.write(str(total_annotated_mention_not_identified_incorrectly_converted))
        f_output.write('\t')
        f_output.write("total identified mention not annotated: ")
        f_output.write(str(total_identified_mention_not_annotated))
        f_output.write('\t')
        f_output.write("total annoated verb: ")
        f_output.write(str(total_annotated_verb))
        f_output.write('\t')
        f_output.write("total annotated verb correctly identified: ")
        f_output.write(str(total_annotated_verb_correctly_identified))
        f_output.write('\t')
        f_output.write("total annotated verb correctly identified correctly converted: ")
        f_output.write(str(total_annotated_verb_correctly_identified_correctly_converted))
        f_output.write('\t')
        f_output.write("total annotated verb not identified: ")
        f_output.write(str(total_annotated_verb_not_identified))
        f_output.write('\t')
        f_output.write("total identified verb not annotated: ")
        f_output.write(str(total_identified_verb_not_annotated))
        f_output.write('\n')
                 
        f_gold.close()
        f_test.close()
        f_dev.close()
        f_temp_gold.close()
        f_temp_test.close()
        f_temp_dev.close()
        f_output.close()
#        
        end_time = time.time()
        print ('the testing took: %d(s)' % (end_time - start_time))
            
    # Train and evaluate models
    def train_and_evaluate_scenario2(self, FLAGS):       
        num_epochs    = FLAGS.num_epochs
        learning_rate = FLAGS.learning_rate

        drop_rate_1   = FLAGS.dropout_rate_1
        drop_rate_2 = FLAGS.dropout_rate_2
        lstm_size   = FLAGS.lstm_size
        hidden_size = FLAGS.hidden_size
        margin = FLAGS.margin
        patience = FLAGS.patience
        
        training_data_path = FLAGS.conll_data_dir
        development_data_dir = FLAGS.development_data_dir
        pov_gold_testing_data_dir = FLAGS.pov_gold_testing_data_dir
        pov_auto_dev_data_dir = FLAGS.pov_auto_dev_data_dir
        pov_auto_testing_data_dir = FLAGS.pov_auto_testing_data_dir

        loaded_model_path = FLAGS.loaded_model_path
        is_training = FLAGS.is_training
        
        # Define models
        model = eval("Model_" + str(self.mode))(
                    self.use_gpu, lstm_size, hidden_size, drop_rate_1, drop_rate_2)
    
        if loaded_model_path is not None:
            model.load_state_dict(torch.load(loaded_model_path))
        # If GPU is availabel, then run experiments on GPU
        # For testing purpose, temprally comment out
        if self.use_gpu:
            model.cuda()

        if is_training == 0:
            model.eval()
            self.evaluate(model,FLAGS)
            return
            
        # ======================================================================
        # define training operation
        #
        optimizer = OPT.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=learning_rate)

        # ======================================================================
        # define accuracy operation
        # ----------------- YOUR CODE HERE ----------------------
        #

        output_file = 'result_f_m1_dev_conll_gold_test_three' + ".txt"
        f_output = open(output_file, 'w+', encoding='utf-8')
        f_temp_gold = open('temp_gold.txt', 'w+', encoding='utf-8')
        f_temp_test = open('temp_test.txt', 'w+', encoding='utf-8')
        f_temp_dev = open('temp_dev.txt', 'w+', encoding='utf-8')
        
        early_stopping = EarlyStopping(patience=patience)
        EARLY_STOP = False
        for i in range(num_epochs):
            if EARLY_STOP:
                break
            
            # put model to training mode
            model.train()

            print(20 * '*', 'epoch', i+1, 20 * '*')
            f_output.write('********************epoch')
            f_output.write(str(i+1))
            f_output.write('********************\n')
            
            start_time = time.time()
            s = 0
            while s < 103:
                model.train()

                batch_l1, batch_l2, batch_r, batch_len_r, batch_first_mention_bit, batch_second_mention_bit, batch_focus_mention_length, batch_other_mention_length, batch_coreference_chain_length, batch_used_bit_seql_1, batch_same_bit_seql_1, batch_used_bit_seql_2, batch_same_bit_seql_2, batch_is_subject_list, batch_is_object_list, batch_label, max_seql_length = \
                    self.get_batch(training_data_path, s+1)
            
                rep_focus_mention_length_array = self.get_mention_distance_embedding(batch_focus_mention_length)
                rep_other_mention_length_array = self.get_mention_distance_embedding(batch_other_mention_length)
                rep_coreference_chain_length_array = self.get_coreference_chain_distance_embedding(batch_coreference_chain_length)

                rep_r = Variable(torch.from_numpy(batch_r),
                                requires_grad=False).type(self.float_type)
                len_r = Variable(torch.from_numpy(batch_len_r),
                                requires_grad=False).type(self.long_type)
                rep_first_mention_bit = Variable(torch.from_numpy(batch_first_mention_bit),
                                requires_grad=False).type(self.float_type)
                rep_second_mention_bit = Variable(torch.from_numpy(batch_second_mention_bit),
                                requires_grad=False).type(self.float_type)
                rep_focus_mention_length = Variable(torch.from_numpy(rep_focus_mention_length_array),
                                requires_grad=False).type(self.float_type)
                rep_other_mention_length = Variable(torch.from_numpy(rep_other_mention_length_array),
                                requires_grad=False).type(self.float_type)
                rep_coreference_chain_length = Variable(torch.from_numpy(rep_coreference_chain_length_array),
                                requires_grad=False).type(self.float_type)
                rep_used_bit_seql_1 = Variable(torch.from_numpy(batch_used_bit_seql_1),
                                requires_grad=False).type(self.float_type)
                rep_same_bit_seql_1 = Variable(torch.from_numpy(batch_same_bit_seql_1),
                                requires_grad=False).type(self.float_type)
                rep_used_bit_seql_2 = Variable(torch.from_numpy(batch_used_bit_seql_2),
                                requires_grad=False).type(self.float_type)
                rep_same_bit_seql_2 = Variable(torch.from_numpy(batch_same_bit_seql_2),
                                requires_grad=False).type(self.float_type)

                rep_is_subject_list = Variable(torch.from_numpy(batch_is_subject_list),
                                requires_grad=False).type(self.float_type)
                rep_is_object_list = Variable(torch.from_numpy(batch_is_object_list),
                                requires_grad=False).type(self.float_type)
                label = Variable(torch.from_numpy(batch_label),
                                 requires_grad=False).type(self.float_type)
                
                # Forward pass: predict scores
                score_l1, score_l2 = model(batch_l1, batch_l2, max_seql_length, rep_r, len_r, rep_first_mention_bit, rep_second_mention_bit, rep_focus_mention_length, rep_other_mention_length, rep_coreference_chain_length, rep_used_bit_seql_1, rep_same_bit_seql_1, rep_used_bit_seql_2, rep_same_bit_seql_2, rep_is_subject_list, rep_is_object_list)

                # Zero gradients, perform a backward pass, and update the weights.      
                cost = NN.MarginRankingLoss(margin=margin)(score_l1, score_l2, label)             
                optimizer.zero_grad()            
                cost.backward(retain_graph=True)
                optimizer.step()

                s = s+1

            end_time = time.time()
            print ('the training took: %d(s)' % (end_time - start_time))
            f_output.write('the training took: ')
            f_output.write(str(end_time - start_time))
            f_output.write('(s)\n')
            
            start_time = time.time()
            file_name_gold = 'output_f_m1_dev_conll_gold_test_pov_gold' + str(i+1) + ".txt"
            f_gold = open(file_name_gold, 'w+', encoding='utf-8')
            file_name_dev = 'output_f_m1_dev_conll_gold_test_pov_dev' + str(i+1) + ".txt"
            f_dev = open(file_name_dev, 'w+', encoding='utf-8')
            file_name_test = 'output_f_m1_dev_conll_gold_test_pov_test' + str(i+1) + ".txt"
            f_test = open(file_name_test, 'w+', encoding='utf-8')
            
            valid_losses, total_correct_dev, total_size_dev = self.get_model_development_data_scenario2(development_data_dir, model, margin)
            valid_loss = numpy.average(valid_losses)       
            print_msg = (f'valid_loss: {valid_loss:.5f}')
            print(print_msg)                     
            valid_losses = []
                
            print ('Accuracy of the trained model on validation set %f' % (total_correct_dev / total_size_dev))
            print('dev size: ')
            print(total_size_dev)

            
            best_score = total_correct_dev * 1.0 / total_size_dev
            early_stopping(best_score, model)
            if early_stopping.early_stop:
                print("Early stopping")
                f_output.write("Early stopping\n")
                EARLY_STOP = True
            
            end_time = time.time()
            print ('the validation took: %d(s)' % (end_time - start_time))
            f_output.write('the validation took: ')
            f_output.write(str(end_time - start_time))
            f_output.write('(s)\n')

            if EARLY_STOP:
                start_time = time.time()
                valid_losses, total_size, total_correct, total_annotated_mention, total_annotated_mention_correctly_identified, total_annotated_mention_correctly_identified_correctly_converted, total_annotated_mention_incorrectly_identified, total_annotated_mention_incorrectly_identified_incorrectly_converted, total_annotated_mention_not_identified, total_annotated_mention_not_identified_incorrectly_converted, total_identified_mention_not_annotated, total_identified_mention_not_annotated_incorrectly_converted, total_annotated_verb, total_annotated_verb_correctly_identified, total_annotated_verb_correctly_identified_correctly_converted, total_annotated_verb_not_identified, total_annotated_verb_not_identified_incorrectly_converted, total_identified_verb_not_annotated, total_identified_verb_not_annotated_incorrectly_converted, total_se_error =self.get_model_testing_data_scenario2(f_gold, f_temp_gold, pov_gold_testing_data_dir, model, margin, True, True)
                valid_losses = []
            
                valid_losses, total_size, total_correct, total_annotated_mention, total_annotated_mention_correctly_identified, total_annotated_mention_correctly_identified_correctly_converted, total_annotated_mention_incorrectly_identified, total_annotated_mention_incorrectly_identified_incorrectly_converted, total_annotated_mention_not_identified, total_annotated_mention_not_identified_incorrectly_converted, total_identified_mention_not_annotated, total_identified_mention_not_annotated_incorrectly_converted, total_annotated_verb, total_annotated_verb_correctly_identified, total_annotated_verb_correctly_identified_correctly_converted, total_annotated_verb_not_identified, total_annotated_verb_not_identified_incorrectly_converted, total_identified_verb_not_annotated, total_identified_verb_not_annotated_incorrectly_converted, total_se_error=self.get_model_testing_data_scenario2(f_dev, f_temp_dev, pov_auto_dev_data_dir, model, margin, False, True)
                valid_losses = []
                        
                valid_losses, total_size, total_correct, total_annotated_mention, total_annotated_mention_correctly_identified, total_annotated_mention_correctly_identified_correctly_converted, total_annotated_mention_incorrectly_identified, total_annotated_mention_incorrectly_identified_incorrectly_converted, total_annotated_mention_not_identified, total_annotated_mention_not_identified_incorrectly_converted, total_identified_mention_not_annotated, total_identified_mention_not_annotated_incorrectly_converted, total_annotated_verb, total_annotated_verb_correctly_identified, total_annotated_verb_correctly_identified_correctly_converted, total_annotated_verb_not_identified, total_annotated_verb_not_identified_incorrectly_converted, total_identified_verb_not_annotated, total_identified_verb_not_annotated_incorrectly_converted, total_se_error=self.get_model_testing_data_scenario2(f_test, f_temp_test, pov_auto_testing_data_dir, model, margin, False, True)
                valid_losses = []
                
                end_time = time.time()
                print ('the testing took: %d(s)' % (end_time - start_time))
                 
            f_gold.close()
            f_test.close()
            f_dev.close()
        f_temp_gold.close()
        f_temp_test.close()
        f_temp_dev.close()
        f_output.close()
    
def main():    
    parser = argparse.ArgumentParser('LSTM models')

    parser.add_argument('--conll_data_dir',
                    type=str,
                    default='output_training_array/conll_padding',
                    help='Directory to the conll training data.')
    parser.add_argument('--development_data_dir',
                    type=str,
                    default='conll_data/dev/',
                    help='Directory of the development data.')
    parser.add_argument('--pov_gold_testing_data_dir',
                    type=str,
                    default='pov_data_gold/',
                    help='Path of the testing data.')
    parser.add_argument('--pov_auto_dev_data_dir',
                    type=str,
                    default='pov_data_auto/dev/',
                    help='Path of the testing data.')
    parser.add_argument('--pov_auto_testing_data_dir',
                    type=str,
                    default='pov_data_auto/test/',
                    help='Path of the testing data.')
    parser.add_argument('--loaded_model_path',
                    type=str,
                    default=None,
                    help='Path of the loaded model.')
    parser.add_argument('--is_training',
                    type=int,
                    default=1, 
                    help='If it is in training mode.')
    parser.add_argument('--num_epochs',
                    type=int,
                    default=200, 
                    help='Number of epochs to run trainer.')
    parser.add_argument('--batch_size', 
                    type=int,
                    default=512, 
                    help='Batch size. Must divide evenly into the dataset sizes.')
    parser.add_argument('--learning_rate', 
                    type=float,
                    default=12e-5,
                    help='Initial learning rate.')
    parser.add_argument('--dropout_rate_1',
                    type=float,
                    default=0.3,
                    help='Dropout rate.')
    parser.add_argument('--dropout_rate_2',
                    type=float,
                    default=0.2,
                    help='Dropout rate.')
    parser.add_argument('--lstm_size',
                    type=int,
                    default=50,
                    help='Size of lstm cell.')
    parser.add_argument('--hidden_size',
                    type=int,
                    default=100,
                    help='Size of hidden layer of FFN.')
    parser.add_argument('--margin',
                    type=float,
                    default=0.05,
                    help='Margin for the ranking loss.')
    parser.add_argument('--patience',
                    type=int,
                    default=10,
                    help='Patience for early stopping.')

    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()

    rnn = RNNNet1(1)
    rnn.train_and_evaluate_scenario2(FLAGS)
    
if __name__ == "__main__":
    main()
