# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 10:51:41 2019

@author: chenm
"""

import argparse
from os import listdir
from os.path import isfile, join
import time
import gc 
import os

import numpy
import pickle

import torch

from modeling_bert_coref import BertForSequenceClassificationCoref
from transformers import BertTokenizer
from transformers import AdamW

from pytorchtools import EarlyStopping
    
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
        
###############################################################
# The Transformer model
#
class POVTransformer(object):
    def __init__(self, mode):
        self.mode = mode
        self.raw_document_data_list = []
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_basic_tokenize=False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.old_lr = 1e-5
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        
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
    
    # Get a batch of data from given data set.
    def get_batch(self, training_data_path, index):
        training_data_path = training_data_path+"_"+str(index)+".pkl"
        f = open(training_data_path, 'rb')
        
        data_set = pickle.load(f)
        
        seq_list_1 = data_set[0]
        Y_list_1 = data_set[1]  
        mention_pos_list_1 = data_set[2]
        seq_list_2 = data_set[3]
        Y_list_2 = data_set[4]
        mention_pos_list_2 = data_set[5]
        label_list = data_set[6]
        
        data_set = None
        f.close()
        
        return seq_list_1, Y_list_1, mention_pos_list_1, seq_list_2, Y_list_2, mention_pos_list_2, label_list

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
    
    def get_updated_sequence(self, raw_sequence, related_mentions, mention_parts=None, focus_mention_len=-1, offset=0, related_mention_cluster_ids=None, focus_mention_cluster_id=-100):
        updated_sequence = []
        coreference_info_matrix = []
        pos_to_cluster_id = {}
        
        related_offset = 1
        
        for i in range(len(related_mentions)):
            related_mention = related_mentions[i]
            start_pos = related_mention[0]
            end_pos = related_mention[1]

            if start_pos < 100:
                related_offset = related_offset + 2
                
        for i in range(len(raw_sequence)):
            if mention_parts != None:
                if i >= 100 + offset and i <= 100 + offset + focus_mention_len - 1:
                    if i == 100 + offset:
                        updated_sequence.append('[unused0]')
                    if i-100 - offset < len(mention_parts):
                        updated_sequence.append(mention_parts[i-100-offset])
                        pos_to_cluster_id[len(updated_sequence)-1]=focus_mention_cluster_id
                        if i-100-offset == len(mention_parts)-1:
                            updated_sequence.append('[unused0]')
                        if i-99-offset < len(mention_parts) and i == 100 + offset + focus_mention_len - 1:
                            for j in range(i-99-offset, len(mention_parts)):
                                updated_sequence.append(mention_parts[j])
                                pos_to_cluster_id[len(updated_sequence)-1]=focus_mention_cluster_id
                                if j == len(mention_parts)-1:
                                    updated_sequence.append('[unused0]')
                            
                    continue
            else:
                if i >= 100 + offset and i <= 100 + offset + focus_mention_len - 1:
                    if i == 100 + offset and i-100-offset == focus_mention_len-1:
                        updated_sequence.append('[unused0]')
                        updated_sequence.append(raw_sequence[i])
                        pos_to_cluster_id[len(updated_sequence)-1]=focus_mention_cluster_id
                        updated_sequence.append('[unused0]')
                    elif i == 100 + offset:
                        updated_sequence.append('[unused0]')
                        updated_sequence.append(raw_sequence[i])
                        pos_to_cluster_id[len(updated_sequence)-1]=focus_mention_cluster_id
                    elif i-100-offset == focus_mention_len-1:
                        updated_sequence.append(raw_sequence[i])
                        pos_to_cluster_id[len(updated_sequence)-1]=focus_mention_cluster_id
                        updated_sequence.append('[unused0]')
                    else:
                        updated_sequence.append(raw_sequence[i])
                        pos_to_cluster_id[len(updated_sequence)-1]=focus_mention_cluster_id
                    continue
                
            is_in_related_mention = False
            for j in range(len(related_mentions)):
                related_mention = related_mentions[j]
                related_mention_cluster_id = related_mention_cluster_ids[j]
                start_pos = related_mention[0]
                end_pos = related_mention[1]
                
                if start_pos <= 100 + offset:
                    if i == start_pos and i == end_pos:
                        updated_sequence.append('[unused0]')
                        updated_sequence.append(raw_sequence[i])
                        pos_to_cluster_id[len(updated_sequence)-1]=related_mention_cluster_id
                        updated_sequence.append('[unused0]')
                        is_in_related_mention = True
                        break
                    elif i == start_pos:
                        updated_sequence.append('[unused0]')
                        updated_sequence.append(raw_sequence[i])
                        pos_to_cluster_id[len(updated_sequence)-1]=related_mention_cluster_id
                        is_in_related_mention = True
                        break
                    elif i == end_pos:
                        updated_sequence.append(raw_sequence[i])
                        pos_to_cluster_id[len(updated_sequence)-1]=related_mention_cluster_id
                        updated_sequence.append('[unused0]')
                        is_in_related_mention = True
                        break
                    elif i > start_pos and i < end_pos:
                        updated_sequence.append(raw_sequence[i])
                        pos_to_cluster_id[len(updated_sequence)-1]=related_mention_cluster_id
                        is_in_related_mention = True
                        break
                else:
                    if i == start_pos:
                        updated_sequence.append('[UNK]')
                        pos_to_cluster_id[len(updated_sequence)-1]=related_mention_cluster_id
                        is_in_related_mention = True
                        break
                    elif i > start_pos and i <= end_pos:
                        is_in_related_mention = True
                        break
                    
            if not is_in_related_mention:
                updated_sequence.append(raw_sequence[i])
        
        for i in range(len(updated_sequence)):
            v = list(numpy.zeros(len(updated_sequence), dtype=numpy.float32))
            coreference_info_matrix.append(v)
            
        for i in range(len(coreference_info_matrix)):
            v = coreference_info_matrix[i]
            pos_list = []
            if i not in pos_to_cluster_id:
                continue
            key_cluster_id = pos_to_cluster_id[i]
            for pos in pos_to_cluster_id:
                cluster_id = pos_to_cluster_id[pos]
                if cluster_id == key_cluster_id:
                    pos_list.append(pos)
                    
            for pos in pos_list:
                v[pos] = 1
                
            coreference_info_matrix[i]=v
            
        return updated_sequence, coreference_info_matrix, related_offset
    
    def prune_mention_set_dev(self, mention_set, focus_mention):
        updated_mention_set=[]
        list1 = ['his','her','our','their']
        list2 = ['himself','herself','ourselves','themselves']
        list3 = ['you','he','she','we','they']
        list4 = ['you','him','her','us','them']
        focus_mention_parts = focus_mention.lower().split()
        focus_mention_contains_reflexive = False
        for focus_mention_part in focus_mention_parts:
            if focus_mention_part in list2:
                focus_mention_contains_reflexive = True
                break

        if (focus_mention.lower() in list1) or (focus_mention.lower().endswith('\'s')) or (focus_mention.lower().endswith('’s')):
            for mention in mention_set:
                if (mention.lower() in list1) or (mention.lower().endswith('\'s')) or (mention.lower().endswith('’s')):
                    updated_mention_set.append(mention.lower())
        elif focus_mention_contains_reflexive:
            for mention in mention_set:
                if mention.lower() in list2:
                    updated_mention_set.append(mention.lower())        
        elif focus_mention.lower() in list3:
            for mention in mention_set:
                mention_parts = mention.lower().split()
                mention_contains_reflexive = False
                for mention_part in mention_parts:
                    if mention_part in list2:
                        mention_contains_reflexive = True
                        break
                if (mention.lower() in list1) or (mention.lower().endswith('\'s')) or (mention.lower().endswith('’s')) or (mention.lower() in list2) or (mention.lower() in list4):
                    continue
                elif mention_contains_reflexive:
                    continue
                else:
                    updated_mention_set.append(mention.lower())
        elif focus_mention.lower() in list4:
            for mention in mention_set:
                if (mention.lower() in list1) or (mention.lower().endswith('\'s')) or (mention.lower().endswith('’s')) or (mention.lower() in list2) or (mention.lower() in list3):
                    continue
                else:
                    updated_mention_set.append(mention.lower())
        else:
            has_processed = False
            if (focus_mention.lower() not in list1) and (not focus_mention.lower().endswith('\'s')) and (not focus_mention.lower().endswith('’s')):
                has_processed = True
                for mention in mention_set:
                    if (mention.lower() in list1) or (mention.lower().endswith('\'s')) or (mention.lower().endswith('’s')):
                        continue
            
                    if not focus_mention_contains_reflexive:
                        mention_parts = mention.lower().split()
                        mention_contains_reflexive = False
                        for mention_part in mention_parts:
                            if mention_part in list2:
                                mention_contains_reflexive = True
                                break
                        if mention_contains_reflexive:
                            continue
                        else:
                            updated_mention_set.append(mention.lower())
            if not focus_mention_contains_reflexive:
                has_processed = True
                for mention in mention_set:
                    mention_parts = mention.lower().split()
                    mention_contains_reflexive = False
                    for mention_part in mention_parts:
                        if mention_part  in list2:
                            mention_contains_reflexive = True
                            break
                    if mention_contains_reflexive:
                        continue
                    else:
                        if (focus_mention.lower() not in list1) and (not focus_mention.lower().endswith('\'s')) and (not focus_mention.lower().endswith('’s')):
                            if (mention.lower() in list1) or (mention.lower().endswith('\'s')) or (mention.lower().endswith('’s')):
                                continue
                            else:
                                if mention.lower() not in updated_mention_set:
                                    updated_mention_set.append(mention.lower())
            if not has_processed:
                updated_mention_set = mention_set
            
        return updated_mention_set
    
    def prune_mention_set_test(self, mention_set, focus_mention):
        updated_mention_set=[]
        list1 = ['my','your','his', 'her','our','their']
        list2 = ['myself','yourself','himself','herself','ourselves','themselves']
        list3 = ['i','you','he','she','we','they']
        list4 = ['me','you','him','her','us','them']
        focus_mention_parts = focus_mention.lower().split()
        focus_mention_contains_reflexive = False
        for focus_mention_part in focus_mention_parts:
            if focus_mention_part in list2:
                focus_mention_contains_reflexive = True
                break

        if (focus_mention.lower() in list1) or (focus_mention.lower().endswith('\'s')) or (focus_mention.lower().endswith('’s')):
            for mention in mention_set:
                if (mention.lower() in list1) or (mention.lower().endswith('\'s')) or (mention.lower().endswith('’s')):
                    updated_mention_set.append(mention.lower())
        elif focus_mention_contains_reflexive:
            for mention in mention_set:
                if mention.lower() in list2:
                    updated_mention_set.append(mention.lower())        
        elif focus_mention.lower() in list3:
            for mention in mention_set:
                mention_parts = mention.lower().split()
                mention_contains_reflexive = False
                for mention_part in mention_parts:
                    if mention_part in list2:
                        mention_contains_reflexive = True
                        break
                if (mention.lower() in list1) or (mention.lower().endswith('\'s')) or (mention.lower().endswith('’s')) or (mention.lower() in list2) or (mention.lower() in list4):
                    continue
                elif mention_contains_reflexive:
                    continue
                else:
                    updated_mention_set.append(mention.lower())
        elif focus_mention.lower() in list4:
            for mention in mention_set:
                if (mention.lower() in list1) or (mention.lower().endswith('\'s')) or (mention.lower().endswith('’s')) or (mention.lower() in list2) or (mention.lower() in list3):
                    continue
                else:
                    updated_mention_set.append(mention.lower())
        else:
            has_processed = False
            if (focus_mention.lower() not in list1) and (not focus_mention.lower().endswith('\'s')) and (not focus_mention.lower().endswith('’s')):
                has_processed = True
                for mention in mention_set:
                    if (mention.lower() in list1) or (mention.lower().endswith('\'s')) or (mention.lower().endswith('’s')):
                        continue
            
                    if not focus_mention_contains_reflexive:
                        mention_parts = mention.lower().split()
                        mention_contains_reflexive = False
                        for mention_part in mention_parts:
                            if mention_part in list2:
                                mention_contains_reflexive = True
                                break
                        if mention_contains_reflexive:
                            continue
                        else:
                            updated_mention_set.append(mention.lower())
            if not focus_mention_contains_reflexive:
                has_processed = True
                for mention in mention_set:
                    mention_parts = mention.lower().split()
                    mention_contains_reflexive = False
                    for mention_part in mention_parts:
                        if mention_part  in list2:
                            mention_contains_reflexive = True
                            break
                    if mention_contains_reflexive:
                        continue
                    else:
                        if (focus_mention.lower() not in list1) and (not focus_mention.lower().endswith('\'s')) and (not focus_mention.lower().endswith('’s')):
                            if (mention.lower() in list1) or (mention.lower().endswith('\'s')) or (mention.lower().endswith('’s')):
                                continue
                            else:
                                if mention.lower() not in updated_mention_set:
                                    updated_mention_set.append(mention.lower())
            if not has_processed:
                updated_mention_set = mention_set
            
        return updated_mention_set
    
    def create_padding(self, data_list, dic_list_1, dic_list_2):
        updated_data_list = []
        for i in range(len(data_list)): 
            Y = data_list[i]
            dic_1 = dic_list_1[i]
            dic_2 = dic_list_2[i]
            updated_Y = []
            max_length = len(dic_1.keys())
            for key in sorted(dic_1):
                value = dic_1[key]
                if value == -1:
                    y = list(numpy.zeros(max_length, dtype=numpy.float32))
                    updated_Y.append(y)
                else:
                    y = list(numpy.zeros(max_length, dtype=numpy.float32))
                    original_y = Y[value]
                    related_index_list = []
                    for j in range(len(original_y)):
                        if original_y[j] == 1:
                            related_index_list.append(j)
                            
                    mapped_index_list = []
                    for j in related_index_list:
                        current_mapped_index_list = dic_2[j]
                        mapped_index_list.extend(current_mapped_index_list)
                        
                    for mapped_index in mapped_index_list:
                        y[mapped_index] = 1
                    updated_Y.append(y)
                
            updated_data_list.append(updated_Y)
        
        
        return updated_data_list
    
    def validate_list(self, L):
        current_dim_1 = len(L[0])
        current_dim_2 = len(L[0][0])
        is_dim_valid = True
        for l in L:
            if len(l) != current_dim_1:
                is_dim_valid = False
                break
            if len(l[0]) != current_dim_2:
                is_dim_valid = False
                break
            
        return is_dim_valid
            
    def get_model_development_data(self, data_dir, model, margin=0.2, dropout_rate=0):
        # Put model to evaluation mode
        model.eval()

        with torch.no_grad():
            # Evaluate the trained model
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
                    for raw_sequence_data in raw_sequence_data_list:
                        focus_mention_cluster_id = raw_sequence_data._focus_mention_cluster_id
                        focus_mention = raw_sequence_data._original_focus_mention
                        focus_mention_parts = focus_mention.split()
                        if focus_mention_cluster_id not in cluster_id_to_cluster_data:
                            print('eee')
                        focus_mention_cluster_data = cluster_id_to_cluster_data[focus_mention_cluster_id]
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

                        mentions_to_replace = []
                        temp_dict = {}
                        for related_mention in related_mentions:                  
                            if related_mention[0] < 100:
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

                        seq_list_1 = []     
                        seq_list_2 = []
                        label_list = []
                        Y_list_1 = []
                        Y_list_2 = []
                        mention_pos_list_1 = []
                        mention_pos_list_2 = []
                        
                        if len(mentions_to_replace) == 0:
                            updated_sequence_focus, focus_coreference_info_matrix, related_offset = self.get_updated_sequence(raw_sequence, related_mentions, mention_parts=None, focus_mention_len=len(focus_mention_parts), offset=0, related_mention_cluster_ids=related_mention_cluster_ids, focus_mention_cluster_id=focus_mention_cluster_id)
                            
                            previous_mention_cluster_ids = []
                            correct_previous_mentions = []
                            previous_mentions = []
                            previous_mention_index_in_sentence_list = []
                            previous_mention_sentence_num_list = []
                        
                            focus_sequence_text = ''
                            for usf in updated_sequence_focus:
                                focus_sequence_text = focus_sequence_text + ' ' + usf
                            focus_sequence_text = focus_sequence_text.strip()
                            focus_sequence_text = focus_sequence_text.lower()
                            focus_sequence_text = focus_sequence_text.replace("[unk]", "[UNK]")
                            focus_sequence_text = focus_sequence_text.replace("[pad]", "[unused1]")
                            
                            
                            for i in range(len(mention_set)):
                                mention = mention_set[i]

                                if mention.lower() != focus_mention.lower():                                
                                    mention_parts = mention.split()

                                    updated_sequence_other, other_coreference_info_matrix, related_offset = self.get_updated_sequence(raw_sequence, related_mentions, mention_parts, len(focus_mention_parts), offset=0, related_mention_cluster_ids=related_mention_cluster_ids, focus_mention_cluster_id=focus_mention_cluster_id)
                                    
                                    other_sequence_text = ''
                                    for uso in updated_sequence_other:
                                        other_sequence_text = other_sequence_text + ' ' + uso
                                    other_sequence_text = other_sequence_text.strip()
                                    other_sequence_text = other_sequence_text.lower()
                                    other_sequence_text = other_sequence_text.replace("[unk]", "[UNK]")
                                    other_sequence_text = other_sequence_text.replace("[pad]", "[unused1]")
                            
                                    seq_list_1.append(focus_sequence_text)
                                    focus_coreference_info_matrix_copy = focus_coreference_info_matrix.copy()
                                    Y_list_1.append(focus_coreference_info_matrix_copy)
                                    mention_pos_list_1.append(99+related_offset+len(focus_mention_parts))
                                    seq_list_2.append(other_sequence_text)
                                    Y_list_2.append(other_coreference_info_matrix)
                                    mention_pos_list_2.append(99+related_offset+len(mention_parts))
                                    label_list.append(1)
                                else:
                                    seq_list_1.append(focus_sequence_text)
                                    focus_coreference_info_matrix_copy_1 = focus_coreference_info_matrix.copy()
                                    focus_coreference_info_matrix_copy_2 = focus_coreference_info_matrix.copy()
                                    Y_list_1.append(focus_coreference_info_matrix_copy_1)
                                    mention_pos_list_1.append(99+related_offset+len(focus_mention_parts))
                                    seq_list_2.append(focus_sequence_text)
                                    Y_list_2.append(focus_coreference_info_matrix_copy_2)
                                    mention_pos_list_2.append(99+related_offset+len(focus_mention_parts))
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
                            
                            updated_sequence_1 = raw_sequence
                            updated_sequence_2 = []
                            from_index = -1
                            for i in range(len(related_mentions)):      
                                related_mention = related_mentions[i]
                                if related_mention[0] > 100:
                                    from_index = i - 1
                                    break
                            
                            decrease_count = from_index
                            total_offset = 0
                            for sorted_key in temp_dict_sorted_keys:
                                previous_mention_index = sorted_key_to_previous_mention_index[sorted_key]
                                previous_mention = previous_mentions[previous_mention_index]
                                previous_mention_parts = previous_mention.split()
                                correct_previous_mention = correct_previous_mentions[previous_mention_index]
                                correct_previous_mention_parts = correct_previous_mention.split()
                                  
                                has_updated = False
                                previous_mention_cluster_id = previous_mention_cluster_ids[previous_mention_index]
                                if (previous_mention_cluster_id == focus_mention_cluster_id) or (previous_mention_cluster_id in cluster_id_to_he_clusters and focus_mention_cluster_id in cluster_id_to_he_clusters) or (previous_mention_cluster_id in cluster_id_to_she_clusters and focus_mention_cluster_id in cluster_id_to_she_clusters) or (previous_mention_cluster_id in cluster_id_to_it_clusters and focus_mention_cluster_id in cluster_id_to_it_clusters) or (previous_mention_cluster_id in cluster_id_to_they_clusters):
                                    if previous_mention.lower() != correct_previous_mention.lower():
                                        has_updated = True
                                        updated_sequence_2 = ['']*(len(updated_sequence_1)+len(previous_mention_parts)-len(correct_previous_mention_parts))
                                        offset = len(previous_mention_parts) - len(correct_previous_mention_parts) 
                                        total_offset = total_offset + offset
                                        updated_related_mention = related_mentions[decrease_count]
                                        for j in range(updated_related_mention[0]):
                                            updated_sequence_2[j]=updated_sequence_1[j]
                                        for j in range(updated_related_mention[0], updated_related_mention[0]+len(previous_mention_parts)):
                                            updated_sequence_2[j]=previous_mention_parts[j-updated_related_mention[0]]
                                        x = 0
                                        for j in range(updated_related_mention[1]+1,len(updated_sequence_1)):
                                            updated_sequence_2[updated_related_mention[0]+len(previous_mention_parts)+x]=updated_sequence_1[j]
                                            x = x + 1
                                            
                                        updated_sequence_1 = updated_sequence_2.copy()
                                        
                                        if offset != 0:
                                            for j in range(from_index,len(related_mentions)):
                                                related_mention = related_mentions[j]
                                                if j != from_index:
                                                    related_mention[0] = related_mention[0] + offset
                                                related_mention[1] = related_mention[1] + offset
                                                related_mentions[j] = related_mention
                                                
                                if not has_updated:
                                    updated_sequence_2 = updated_sequence_1.copy()
                                    
                                decrease_count = decrease_count - 1
                    
                            updated_sequence_focus, focus_coreference_info_matrix, related_offset = self.get_updated_sequence(updated_sequence_2, related_mentions, mention_parts=None, focus_mention_len=len(focus_mention_parts), offset=total_offset, related_mention_cluster_ids=related_mention_cluster_ids, focus_mention_cluster_id=focus_mention_cluster_id)  
                            
                            focus_sequence_text = ''
                            for usf in updated_sequence_focus:
                                focus_sequence_text = focus_sequence_text + ' ' + usf
                            focus_sequence_text = focus_sequence_text.strip()
                            focus_sequence_text = focus_sequence_text.lower()
                            focus_sequence_text = focus_sequence_text.replace("[unk]", "[UNK]")
                            focus_sequence_text = focus_sequence_text.replace("[pad]", "[unused1]")
                            
                            for i in range(len(mention_set)):
                                mention = mention_set[i]

                                if mention.lower() != focus_mention.lower():                                                                    
                                    mention_parts = mention.split()
                                    
                                    updated_sequence_other, other_coreference_info_matrix, related_offset = self.get_updated_sequence(updated_sequence_2, related_mentions, mention_parts, len(focus_mention_parts), offset=total_offset, related_mention_cluster_ids=related_mention_cluster_ids, focus_mention_cluster_id=focus_mention_cluster_id)                                    
                                    
                                    other_sequence_text = ''
                                    for uso in updated_sequence_other:
                                        other_sequence_text = other_sequence_text + ' ' + uso
                                    other_sequence_text = other_sequence_text.strip()
                                    other_sequence_text = other_sequence_text.lower()
                                    other_sequence_text = other_sequence_text.replace("[unk]", "[UNK]")
                                    other_sequence_text = other_sequence_text.replace("[pad]", "[unused1]")
                                    
                                    seq_list_1.append(focus_sequence_text)
                                    focus_coreference_info_matrix_copy = focus_coreference_info_matrix.copy()
                                    Y_list_1.append(focus_coreference_info_matrix_copy)
                                    mention_pos_list_1.append(99+related_offset+len(focus_mention_parts))
                                    seq_list_2.append(other_sequence_text)
                                    Y_list_2.append(other_coreference_info_matrix)
                                    mention_pos_list_2.append(99+related_offset+len(mention_parts))
                                    label_list.append(1)
                                else:
                                    seq_list_1.append(focus_sequence_text)
                                    focus_coreference_info_matrix_copy_1 = focus_coreference_info_matrix.copy()
                                    focus_coreference_info_matrix_copy_2 = focus_coreference_info_matrix.copy()
                                    Y_list_1.append(focus_coreference_info_matrix_copy_1)
                                    mention_pos_list_1.append(99+related_offset+len(focus_mention_parts))
                                    seq_list_2.append(focus_sequence_text)
                                    Y_list_2.append(focus_coreference_info_matrix_copy_2)
                                    mention_pos_list_2.append(99+related_offset+len(focus_mention_parts))
                                    label_list.append(1)
                                    
                        total_size = total_size + 1
                        if focus_mention_cluster_id not in appeared_focus_mention_cluster_id_list:
                            appeared_focus_mention_cluster_id_list.append(focus_mention_cluster_id)

                            
                        encoding_1 = self.tokenizer(seq_list_1, return_tensors='pt', padding=True, truncation=True)
                        input_ids_1 = encoding_1['input_ids']
                        pre_padding_list_1, post_padding_list_1 = self.get_padding(input_ids_1)
                        encoding_2 = self.tokenizer(seq_list_2, return_tensors='pt', padding=True, truncation=True)
                        input_ids_2 = encoding_2['input_ids']
                        pre_padding_list_2, post_padding_list_2 = self.get_padding(input_ids_2)
                        
                        dic_list_1_1, dic_list_2_1 = self.map_input_ids(pre_padding_list_1, post_padding_list_1, seq_list_1, input_ids_1)
                        dic_list_1_2, dic_list_2_2 = self.map_input_ids(pre_padding_list_2, post_padding_list_2, seq_list_2, input_ids_2)
                        
                        input_ids_1 = input_ids_1.to(self.device)
                        attention_mask_1 = encoding_1['attention_mask']
                        attention_mask_1 = attention_mask_1.to(self.device)

                        input_ids_2 = input_ids_2.to(self.device)
                        attention_mask_2 = encoding_2['attention_mask']
                        attention_mask_2 = attention_mask_2.to(self.device)
                        
                        labels = torch.tensor(label_list).unsqueeze(0)
                        labels = labels.to(self.device)
                        
                        Y_list_1 = self.create_padding(Y_list_1,dic_list_1_1, dic_list_2_1)
                        is_dim_valid_1 = self.validate_list(Y_list_1)
                        Y_list_2 = self.create_padding(Y_list_2,dic_list_1_2, dic_list_2_2)
                        is_dim_valid_2 = self.validate_list(Y_list_2)
                        if not is_dim_valid_1 or not is_dim_valid_2:
                            total_correct = total_correct + 1
                            previous_mentions.append(focus_mention)
                            previous_mention_cluster_ids.append(focus_mention_cluster_id)
                            correct_previous_mentions.append(focus_mention)
                            previous_mention_index_in_sentence_list.append(focus_mention_index_in_sentence)
                            previous_mention_sentence_num_list.append(focus_mention_sentence_num)
                            chosen_mention = focus_mention
                            print("skipping")
                            print(len(input_ids_1))
                            print(len(input_ids_2))
                            continue
                        
                        Y_1 = torch.tensor(Y_list_1)
                        Y_1 = Y_1.to(self.device)
                        Y_2 = torch.tensor(Y_list_2)
                        Y_2 = Y_2.to(self.device)
                        
                        mapped_mention_pos_list_1 = []
                        for h in range(len(mention_pos_list_1)):
                            mention_pos = mention_pos_list_1[h]
                            current_dic = dic_list_2_1[h]
                            mapped_mention_pos_list = current_dic[mention_pos]
                            mapped_mention_pos = mapped_mention_pos_list[-1]
                            mapped_mention_pos_list_1.append(mapped_mention_pos)
                        
                        mapped_mention_pos_list_2 = []
                        for h in range(len(mention_pos_list_2)):
                            mention_pos = mention_pos_list_2[h]
                            current_dic = dic_list_2_2[h]
                            mapped_mention_pos_list = current_dic[mention_pos]
                            mapped_mention_pos = mapped_mention_pos_list[-1]
                            mapped_mention_pos_list_2.append(mapped_mention_pos)
                        
                        outputs = model(input_ids_1=input_ids_1, input_ids_2=input_ids_2, Y_1=Y_1, Y_2=Y_2, attention_mask_1=attention_mask_1, attention_mask_2=attention_mask_2, labels=labels, mention_pos_list_1=mapped_mention_pos_list_1, mention_pos_list_2=mapped_mention_pos_list_2,margin=margin)
                        loss = outputs[0]
                        output1 = outputs[1]
                        output2 = outputs[2]
                        valid_losses.append(loss.item())
                      
                        s1 = output1
                        s2 = output2
                        if self.use_gpu:
                            s1 = output1.cpu()
                            s2 = output2.cpu()
                        s1 = s1.data.numpy()
                        s2 = s2.data.numpy()

                        max_index = -1
                        max_value = -10000000
                        for ii in range(len(s2)):
                            score_2 = s2[ii]
                            if score_2 > max_value:
                                max_value = score_2
                                max_index = ii
                                
                    
                        chosen_mention = ""           
                        for ii in range(len(mention_set)):
                            mention = mention_set[ii]
                            if ii == max_index:                                    
                                previous_mentions.append(mention)
                                previous_mention_cluster_ids.append(focus_mention_cluster_id)
                                correct_previous_mentions.append(focus_mention)
                                previous_mention_index_in_sentence_list.append(focus_mention_index_in_sentence)
                                previous_mention_sentence_num_list.append(focus_mention_sentence_num)
                                chosen_mention = mention
                                    
                                if chosen_mention.lower() == focus_mention.lower():
                                    total_correct = total_correct+1
                                break                       
                    
                    total_entity = total_entity + len(appeared_focus_mention_cluster_id_list)
        
        return valid_losses, total_correct, total_size
    
    def get_model_testing_data(self, f, f_temp, data_dir, model, gold_setting=False, write_result=False, margin=0.2, dropout_rate=0):
        # Put model to evaluation mode
        model.eval()

        with torch.no_grad():
            # Evaluate the trained model
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
                   
                previous_mentions = []
                previous_mention_cluster_ids = []
                correct_previous_mentions = []
                previous_mention_index_in_sentence_list = []
                previous_mention_sentence_num_list = []

                for raw_document_data in self.raw_document_data_list:
                    current_total_correct = 0
                    current_total_size = 0
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

                    appeared_focus_mention_cluster_id_list = []
                    for raw_sequence_data in raw_sequence_data_list:  
                        is_identified_mention = raw_sequence_data._is_identified_mention
                        is_annotated_mention = raw_sequence_data._is_annotated_mention
                        is_identified_verb = raw_sequence_data._is_identified_verb
                        is_annotated_verb = raw_sequence_data._is_annotated_verb

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
                    
                        mentions_to_replace = []
                        temp_dict = {}
                        for related_mention in related_mentions:                  
                            if related_mention[0] < 100:
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

                        seq_list_1 = []     
                        seq_list_2 = []
                        label_list = []
                        Y_list_1 = []
                        Y_list_2 = []
                        mention_pos_list_1 = []
                        mention_pos_list_2 = []
                        
                        if len(mentions_to_replace) == 0:
                            updated_sequence_focus, focus_coreference_info_matrix, related_offset = self.get_updated_sequence(raw_sequence, related_mentions, mention_parts=converted_focus_mention_parts, focus_mention_len=len(original_focus_mention_parts), offset=0, related_mention_cluster_ids=related_mention_cluster_ids, focus_mention_cluster_id=focus_mention_cluster_id)
                            
                            previous_mention_cluster_ids = []
                            correct_previous_mentions = []
                            previous_mentions = []
                            previous_mention_index_in_sentence_list = []
                            previous_mention_sentence_num_list = []

                            focus_sequence_text = ''
                            for usf in updated_sequence_focus:
                                focus_sequence_text = focus_sequence_text + ' ' + usf
                            focus_sequence_text = focus_sequence_text.strip()
                            focus_sequence_text = focus_sequence_text.lower()
                            focus_sequence_text = focus_sequence_text.replace("[unk]", "[UNK]")
                            focus_sequence_text = focus_sequence_text.replace("[pad]", "[unused1]")
                            
                            if converted_focus_mention.lower() not in mention_set:
                                no_correct_string_in_se = True
                            
                            for i in range(len(mention_set)):
                                mention = mention_set[i]

                                if mention.lower() != converted_focus_mention.lower():                                                                   
                                    mention_parts = mention.split()
                    
                                    updated_sequence_other, other_coreference_info_matrix, related_offset = self.get_updated_sequence(raw_sequence, related_mentions, mention_parts, len(original_focus_mention_parts), offset=0, related_mention_cluster_ids=related_mention_cluster_ids, focus_mention_cluster_id=focus_mention_cluster_id)
                                    
                                    other_sequence_text = ''
                                    for uso in updated_sequence_other:
                                        other_sequence_text = other_sequence_text + ' ' + uso
                                    other_sequence_text = other_sequence_text.strip()
                                    other_sequence_text = other_sequence_text.lower()
                                    other_sequence_text = other_sequence_text.replace("[unk]", "[UNK]")
                                    other_sequence_text = other_sequence_text.replace("[pad]", "[unused1]")
                                    
                                    seq_list_1.append(focus_sequence_text)
                                    focus_coreference_info_matrix_copy = focus_coreference_info_matrix.copy()
                                    Y_list_1.append(focus_coreference_info_matrix_copy)
                                    mention_pos_list_1.append(99+related_offset+len(converted_focus_mention_parts))
                                    seq_list_2.append(other_sequence_text)
                                    Y_list_2.append(other_coreference_info_matrix)
                                    mention_pos_list_2.append(99+related_offset+len(mention_parts))
                                    label_list.append(1)
                                else:
                                    seq_list_1.append(focus_sequence_text)
                                    focus_coreference_info_matrix_copy_1 = focus_coreference_info_matrix.copy()
                                    focus_coreference_info_matrix_copy_2 = focus_coreference_info_matrix.copy()
                                    Y_list_1.append(focus_coreference_info_matrix_copy_1)
                                    mention_pos_list_1.append(99+related_offset+len(converted_focus_mention_parts))
                                    seq_list_2.append(focus_sequence_text)
                                    Y_list_2.append(focus_coreference_info_matrix_copy_2)
                                    mention_pos_list_2.append(99+related_offset+len(converted_focus_mention_parts))
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

                            updated_sequence_1 = raw_sequence
                            updated_sequence_2 = []
                            from_index = -1
                            for i in range(len(related_mentions)):      
                                related_mention = related_mentions[i]
                                if related_mention[0] > 100:
                                    from_index = i - 1
                                    break
                            
                            decrease_count = from_index
                            total_offset = 0
                            for sorted_key in temp_dict_sorted_keys:
                                previous_mention_index = sorted_key_to_previous_mention_index[sorted_key]
                                previous_mention = previous_mentions[previous_mention_index]
                                previous_mention_parts = previous_mention.split()
                                correct_previous_mention = correct_previous_mentions[previous_mention_index]
                                correct_previous_mention_parts = correct_previous_mention.split()
                                  
                                has_updated = False
                                previous_mention_cluster_id = previous_mention_cluster_ids[previous_mention_index]
                                if (previous_mention_cluster_id == focus_mention_cluster_id) or (previous_mention_cluster_id in cluster_id_to_he_clusters and focus_mention_cluster_id in cluster_id_to_he_clusters) or (previous_mention_cluster_id in cluster_id_to_she_clusters and focus_mention_cluster_id in cluster_id_to_she_clusters) or (previous_mention_cluster_id in cluster_id_to_it_clusters and focus_mention_cluster_id in cluster_id_to_it_clusters) or (previous_mention_cluster_id in cluster_id_to_they_clusters):
                                    if previous_mention.lower() != correct_previous_mention.lower():
                                        has_updated = True
                                        updated_sequence_2 = ['']*(len(updated_sequence_1)+len(previous_mention_parts)-len(correct_previous_mention_parts))
                                        offset = len(previous_mention_parts) - len(correct_previous_mention_parts)
                                        total_offset = total_offset + offset
                                        updated_related_mention = related_mentions[decrease_count]
                                        for j in range(updated_related_mention[0]):
                                            updated_sequence_2[j]=updated_sequence_1[j]
                                        for j in range(updated_related_mention[0], updated_related_mention[0]+len(previous_mention_parts)):
                                            updated_sequence_2[j]=previous_mention_parts[j-updated_related_mention[0]]
                                        x = 0
                                        for j in range(updated_related_mention[1]+1,len(updated_sequence_1)):
                                            updated_sequence_2[updated_related_mention[0]+len(previous_mention_parts)+x]=updated_sequence_1[j]
                                            x = x + 1
                                            
                                        updated_sequence_1 = updated_sequence_2.copy()
                                        
                                        if offset != 0:
                                            for j in range(from_index,len(related_mentions)):
                                                related_mention = related_mentions[j]
                                                if j != from_index:
                                                    related_mention[0] = related_mention[0] + offset
                                                related_mention[1] = related_mention[1] + offset
                                                related_mentions[j] = related_mention
                                                
                                if not has_updated:
                                    updated_sequence_2 = updated_sequence_1.copy()
                                    
                                decrease_count = decrease_count - 1
                             
                            updated_sequence_focus, focus_coreference_info_matrix, related_offset = self.get_updated_sequence(updated_sequence_2, related_mentions, mention_parts=converted_focus_mention_parts, focus_mention_len=len(original_focus_mention_parts), offset=total_offset, related_mention_cluster_ids=related_mention_cluster_ids, focus_mention_cluster_id=focus_mention_cluster_id)

                            focus_sequence_text = ''
                            for usf in updated_sequence_focus:
                                focus_sequence_text = focus_sequence_text + ' ' + usf
                            focus_sequence_text = focus_sequence_text.strip()
                            focus_sequence_text = focus_sequence_text.lower()
                            focus_sequence_text = focus_sequence_text.replace("[unk]", "[UNK]")
                            focus_sequence_text = focus_sequence_text.replace("[pad]", "[unused1]")
                            
                            if converted_focus_mention.lower() not in mention_set:
                                no_correct_string_in_se = True
                            
                            for i in range(len(mention_set)):
                                mention = mention_set[i]

                                if mention.lower() != converted_focus_mention.lower():                                
                                    mention_parts = mention.split()
                                    
                                    updated_sequence_other, other_coreference_info_matrix, related_offset = self.get_updated_sequence(updated_sequence_2, related_mentions, mention_parts, len(original_focus_mention_parts), offset=total_offset, related_mention_cluster_ids=related_mention_cluster_ids, focus_mention_cluster_id=focus_mention_cluster_id)  
                                    
                                    other_sequence_text = ''
                                    for uso in updated_sequence_other:
                                        other_sequence_text = other_sequence_text + ' ' + uso
                                    other_sequence_text = other_sequence_text.strip()
                                    other_sequence_text = other_sequence_text.lower()
                                    other_sequence_text = other_sequence_text.replace("[unk]", "[UNK]")
                                    other_sequence_text = other_sequence_text.replace("[pad]", "[unused1]")
                                    
                                    seq_list_1.append(focus_sequence_text)
                                    focus_coreference_info_matrix_copy = focus_coreference_info_matrix.copy()
                                    Y_list_1.append(focus_coreference_info_matrix_copy)
                                    mention_pos_list_1.append(99+related_offset+len(converted_focus_mention_parts))
                                    seq_list_2.append(other_sequence_text)
                                    Y_list_2.append(other_coreference_info_matrix)
                                    mention_pos_list_2.append(99+related_offset+len(mention_parts))
                                    label_list.append(1)
                                else:
                                    seq_list_1.append(focus_sequence_text)
                                    focus_coreference_info_matrix_copy_1 = focus_coreference_info_matrix.copy()
                                    focus_coreference_info_matrix_copy_2 = focus_coreference_info_matrix.copy()
                                    Y_list_1.append(focus_coreference_info_matrix_copy_1)
                                    mention_pos_list_1.append(99+related_offset+len(converted_focus_mention_parts))
                                    seq_list_2.append(focus_sequence_text)
                                    Y_list_2.append(focus_coreference_info_matrix_copy_2)
                                    mention_pos_list_2.append(99+related_offset+len(converted_focus_mention_parts))
                                    label_list.append(1)
                        
                        encoding_1 = self.tokenizer(seq_list_1, return_tensors='pt', padding=True, truncation=True)
                        input_ids_1 = encoding_1['input_ids']
                        pre_padding_list_1, post_padding_list_1 = self.get_padding(input_ids_1)
                        encoding_2 = self.tokenizer(seq_list_2, return_tensors='pt', padding=True, truncation=True)
                        input_ids_2 = encoding_2['input_ids']
                        pre_padding_list_2, post_padding_list_2 = self.get_padding(input_ids_2)
                        
                        dic_list_1_1, dic_list_2_1 = self.map_input_ids(pre_padding_list_1, post_padding_list_1, seq_list_1, input_ids_1)
                        dic_list_1_2, dic_list_2_2 = self.map_input_ids(pre_padding_list_2, post_padding_list_2, seq_list_2, input_ids_2)
                        
                        input_ids_1 = input_ids_1.to(self.device)
                        attention_mask_1 = encoding_1['attention_mask']
                        attention_mask_1 = attention_mask_1.to(self.device)
                        
                        input_ids_2 = input_ids_2.to(self.device)
                        attention_mask_2 = encoding_2['attention_mask']
                        attention_mask_2 = attention_mask_2.to(self.device)
                        
                        labels = torch.tensor(label_list).unsqueeze(0)
                        labels = labels.to(self.device)
                        
                        Y_list_1 = self.create_padding(Y_list_1,dic_list_1_1, dic_list_2_1)
                        is_dim_valid_1 = self.validate_list(Y_list_1)
                        Y_list_2 = self.create_padding(Y_list_2,dic_list_1_2, dic_list_2_2)
                        is_dim_valid_2 = self.validate_list(Y_list_2)
                        
                        if not is_dim_valid_1 or not is_dim_valid_2:
                            total_correct = total_correct + 1
                            previous_mentions.append(converted_focus_mention)
                            previous_mention_cluster_ids.append(focus_mention_cluster_id)
                            correct_previous_mentions.append(original_focus_mention)
                            previous_mention_index_in_sentence_list.append(focus_mention_index_in_sentence)
                            previous_mention_sentence_num_list.append(focus_mention_sentence_num)
                            chosen_mention = converted_focus_mention
                            print("skipping")
                            print(len(input_ids_1))
                            print(len(input_ids_2))
                            continue
                        
                        Y_1 = torch.tensor(Y_list_1)
                        Y_1 = Y_1.to(self.device)
                        Y_2 = torch.tensor(Y_list_2)
                        Y_2 = Y_2.to(self.device)
                        
                        mapped_mention_pos_list_1 = []
                        for h in range(len(mention_pos_list_1)):
                            mention_pos = mention_pos_list_1[h]
                            current_dic = dic_list_2_1[h]
                            mapped_mention_pos_list = current_dic[mention_pos]
                            mapped_mention_pos = mapped_mention_pos_list[-1]
                            mapped_mention_pos_list_1.append(mapped_mention_pos)
                        
                        mapped_mention_pos_list_2 = []
                        for h in range(len(mention_pos_list_2)):
                            mention_pos = mention_pos_list_2[h]
                            current_dic = dic_list_2_2[h]
                            mapped_mention_pos_list = current_dic[mention_pos]
                            mapped_mention_pos = mapped_mention_pos_list[-1]
                            mapped_mention_pos_list_2.append(mapped_mention_pos)
                            
                        outputs = model(input_ids_1=input_ids_1, input_ids_2=input_ids_2, Y_1=Y_1, Y_2=Y_2, attention_mask_1=attention_mask_1, attention_mask_2=attention_mask_2, labels=labels, mention_pos_list_1=mapped_mention_pos_list_1, mention_pos_list_2=mapped_mention_pos_list_2, margin=margin)
                        loss = outputs[0]
                        output1 = outputs[1]
                        output2 = outputs[2]
                        valid_losses.append(loss.item())

                        s1 = output1
                        s2 = output2
                        if self.use_gpu:
                            s1 = output1.cpu()
                            s2 = output2.cpu()
                        s1 = s1.data.numpy()
                        s2 = s2.data.numpy()
                        max_index = -1
                        max_value = -10000000

                        for ii in range(len(s2)):
                            score_2 = s2[ii]

                            if score_2 > max_value:
                                max_value = score_2
                                max_index = ii
                
                        chosen_mention = ""
                        for ii in range(len(mention_set)):
                            mention = mention_set[ii]
                            if ii == max_index:
                                previous_mentions.append(mention)
                                previous_mention_cluster_ids.append(focus_mention_cluster_id)
                                correct_previous_mentions.append(original_focus_mention)
                                previous_mention_index_in_sentence_list.append(focus_mention_index_in_sentence)
                                previous_mention_sentence_num_list.append(focus_mention_sentence_num)
                                chosen_mention = mention
                                break
                            
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
                            for x in range(100):
                                f.write("%s " %raw_sequence[x])
                            f.write(chosen_mention)
                            f.write("\n")
                    
                    total_entity = total_entity + len(appeared_focus_mention_cluster_id_list)
                    
                    f_temp.write(data_path)
                    f_temp.write("\n")
                   
                    f_temp.write("current total correct: ")
                    f_temp.write(str(current_total_correct))
                    f_temp.write('\t')
                    f_temp.write("current total size: ")
                    f_temp.write(str(current_total_size))
                    f_temp.write('\t')
                    f_temp.write("current total annotated mention: ")
                    f_temp.write(str(current_total_annotated_mention))
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
                    f_temp.write("current total annotated mention incorrectly identified incorrectly converted: ")
                    f_temp.write(str(current_total_annotated_mention_incorrectly_identified_incorrectly_converted))
                    f_temp.write('\t')
                    f_temp.write("current total annotated mention not identified: ")
                    f_temp.write(str(current_total_annotated_mention_not_identified))
                    f_temp.write('\t')
                    f_temp.write("current total annotated mention not identified incorrectly converted: ")
                    f_temp.write(str(current_total_annotated_mention_not_identified_incorrectly_converted))
                    f_temp.write('\t')
                    f_temp.write("current total identified mention not annotated: ")
                    f_temp.write(str(current_total_identified_mention_not_annotated))
                    f_temp.write('\t')
                    f_temp.write("current total identified mention not annotated incorrectly converted: ")
                    f_temp.write(str(current_total_identified_mention_not_annotated_incorrectly_converted))
                    f_temp.write('\t')
                    f_temp.write("current total annoated verb: ")
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
                    f_temp.write("current total annotated verb not identified incorrectly converted: ")
                    f_temp.write(str(current_total_annotated_verb_not_identified_incorrectly_converted))
                    f_temp.write('\t')
                    f_temp.write("current total identified verb not annotated: ")
                    f_temp.write(str(current_total_identified_verb_not_annotated))
                    f_temp.write('\t')
                    f_temp.write("current total identified verb not annotated incorrectly converted: ")
                    f_temp.write(str(current_total_identified_verb_not_annotated_incorrectly_converted))
                    f_temp.write('\t')
                    f_temp.write("current total se error: ")
                    f_temp.write(str(current_total_se_error))
                    f_temp.write('\n')
        
        annotated_mention_correctly_identified_incorrectly_converted_count = total_annotated_mention_correctly_identified - total_annotated_mention_correctly_identified_correctly_converted
        annotated_mention_incorrectly_identified_count = total_annotated_mention_not_identified_incorrectly_converted + total_annotated_mention_incorrectly_identified_incorrectly_converted
        annotated_verb_correctly_identified_incorrectly_converted_count = total_annotated_verb_correctly_identified - total_annotated_verb_correctly_identified_correctly_converted
        annotated_verb_incorrectly_identified_count = total_annotated_verb_not_identified_incorrectly_converted
        identified_mention_count = total_annotated_mention_correctly_identified+total_annotated_mention_incorrectly_identified+total_identified_mention_not_annotated_incorrectly_converted
        identified_verb_count=total_annotated_verb_correctly_identified+total_identified_verb_not_annotated_incorrectly_converted
        
        return valid_losses, total_size, total_correct, total_annotated_mention, total_annotated_verb, annotated_mention_correctly_identified_incorrectly_converted_count, annotated_mention_incorrectly_identified_count, annotated_verb_correctly_identified_incorrectly_converted_count, annotated_verb_incorrectly_identified_count, identified_mention_count, identified_verb_count
     

    def evaluate(self, model, FLAGS):
        start_time = time.time()
        
        margin = FLAGS.margin
        dropout_rate = FLAGS.dropout_rate
        development_data_dir = FLAGS.development_data_dir
        pov_gold_testing_data_dir = FLAGS.pov_gold_testing_data_dir
        pov_auto_testing_data_dir = FLAGS.pov_auto_testing_data_dir

        output_file = 'result_f_transformer_dev_conll_gold_test_three' + ".txt"
        f_output = open(output_file, 'w+', encoding='utf-8')
        f_temp_gold = open('temp_gold_transformer.txt', 'w+', encoding='utf-8')
        f_temp_test = open('temp_test_transformer.txt', 'w+', encoding='utf-8')
        file_name_gold = 'output_f_transformer_dev_conll_gold_test_pov_gold.txt'
        f_gold = open(file_name_gold, 'w+', encoding='utf-8')
        file_name_test = 'output_f_transformer_dev_conll_gold_test_pov_test.txt'
        f_test = open(file_name_test, 'w+', encoding='utf-8')
            
        valid_losses, total_correct_dev, total_size_dev = self.get_model_development_data(development_data_dir, model, margin, dropout_rate)
        valid_loss = numpy.average(valid_losses)       
        print_msg = (f'valid_loss: {valid_loss:.5f}')
        print(print_msg)                     
        valid_losses = []                
        print ('Accuracy of the trained model on the CoNLL data %f' % (total_correct_dev / total_size_dev))
        
        valid_losses, total_size, total_correct, total_annotated_mention, total_annotated_verb, annotated_mention_correctly_identified_incorrectly_converted_count, annotated_mention_incorrectly_identified_count, annotated_verb_correctly_identified_incorrectly_converted_count, annotated_verb_incorrectly_identified_count, identified_mention_count, identified_verb_count =self.get_model_testing_data(f_gold, f_temp_gold, pov_gold_testing_data_dir, model, True, True, margin, dropout_rate)
        valid_losses = []
        print('Results on pov data - gold setting: ')                
        print("accuracy: ")
        print(1.0*total_correct/total_size)
          
        valid_losses, total_size, total_correct, total_annotated_mention, total_annotated_verb, annotated_mention_correctly_identified_incorrectly_converted_count, annotated_mention_incorrectly_identified_count, annotated_verb_correctly_identified_incorrectly_converted_count, annotated_verb_incorrectly_identified_count, identified_mention_count, identified_verb_count=self.get_model_testing_data(f_test, f_temp_test, pov_auto_testing_data_dir, model, False, True, margin, dropout_rate)
        valid_losses = []            
        correct_mention = total_annotated_mention - annotated_mention_correctly_identified_incorrectly_converted_count - annotated_mention_incorrectly_identified_count
        correct_verb = total_annotated_verb - annotated_verb_correctly_identified_incorrectly_converted_count - annotated_verb_incorrectly_identified_count 
        recall = 1.0 * (correct_mention + correct_verb) / (total_annotated_mention + total_annotated_verb)
        precision = 1.0 * (correct_mention + correct_verb) / (identified_mention_count + identified_verb_count)
        F1 = 2*recall*precision/(recall+precision)
        print('Results on pov data - auto setting: ')   
        print('Recall: ')
        print(recall)
        print('Precision: ')
        print(precision)
        print('F1 score: ')
        print(F1)  
                 
        f_gold.close()
        f_test.close()
        f_temp_gold.close()
        f_temp_test.close()
        f_output.close()
       
        end_time = time.time()
        print ('the testing took: %d(s)' % (end_time - start_time))
     
    def get_padding(self, input_ids_list):
        pre_padding_list = []
        post_padding_list = []
        
        for input_ids in input_ids_list:
            pre_padding = 0
            post_padding = -1
            for i in range(len(input_ids)):
                if input_ids[i] == 102:
                    post_padding = i
                    break
            pre_padding_list.append(pre_padding)
            post_padding_list.append(post_padding)
            
        return pre_padding_list, post_padding_list
    
    def map_input_ids(self, pre_padding_list, post_padding_list, t_text_batch, input_ids_list):
        dic_list_1 = []
        dic_list_2 = []
        
        for i in range(len(t_text_batch)):
            dic_1 = {}
            dic_2 = {}
            pre_padding = pre_padding_list[i]
            post_padding = post_padding_list[i]
            text = t_text_batch[i]
            text_parts = text.split()
            input_ids = input_ids_list[i]
            
            index = 0
            word_index = 0
            if pre_padding != -1:
                index = pre_padding + 1
                for j in range(index):
                    dic_1[j] = -1

            for text_part in text_parts:
                tokens = self.tokenizer.tokenize(text_part)
                for j in range(index, index+len(tokens)):
                    dic_1[j] = word_index
                    if word_index not in dic_2:
                        temp_list = []
                        temp_list.append(j)
                        dic_2[word_index] = temp_list
                    else:
                        temp_list = dic_2[word_index]
                        temp_list.append(j)
                        dic_2[word_index] = temp_list
                        
                index = index + len(tokens)
                word_index = word_index + 1
            
            if post_padding != -1:
                for j in range(post_padding, len(input_ids)):
                    dic_1[j] = -1
                    
            dic_list_1.append(dic_1)
            dic_list_2.append(dic_2)
            
        return dic_list_1, dic_list_2 
    
    def update_learning_rate(self, niter_decay, optimizer):
        lrd = self.old_lr / niter_decay
        lr = self.old_lr - lrd
        for param_group in optimizer.param_groups:
            param_group['lr']=lr
            
        self.old_lr = lr
        
        
    # Train and evaluate models
    def train_and_evaluate(self, FLAGS):       
        num_epochs    = FLAGS.num_epochs
        learning_rate = FLAGS.learning_rate
        patience = FLAGS.patience
        
        training_data_path = FLAGS.conll_data_dir
        development_data_dir = FLAGS.development_data_dir

        loaded_model_path = FLAGS.loaded_model_path
        is_training = FLAGS.is_training
        
        margin = FLAGS.margin
        dropout_rate = FLAGS.dropout_rate
        
        model = BertForSequenceClassificationCoref.from_pretrained('bert-base-uncased')       
        optimizer = AdamW(model.parameters(), lr=learning_rate)       
        self.old_lr = learning_rate
        
        if loaded_model_path is not None:
            model.load_state_dict(torch.load(loaded_model_path))
            
        # If GPU is availabel, then run experiments on GPU
        if self.use_gpu:
            model.cuda()

        if is_training == 0:
            model.eval()
            self.evaluate(model,FLAGS)
            return
            
        # ======================================================================
        # define training operation

        # put model to training mode
        model.train()
        early_stopping = EarlyStopping(patience=patience)
        EARLY_STOP = False
        for i in range(num_epochs):
            if EARLY_STOP:
                break
            
            print(20 * '*', 'epoch', i+1, 20 * '*')
            
            model.train()
            start_time = time.time()
            s = 0
            while s < 3273:
                text_batch_1, batch_Y_1, mention_pos_list_1, text_batch_2, batch_Y_2, mention_pos_list_2, batch_label = self.get_batch(training_data_path, s+1)
                for j in range(4):
                    optimizer.zero_grad()
                    t_text_batch_1 = text_batch_1[j*4:j*4+4]
                    t_text_batch_2 = text_batch_2[j*4:j*4+4]
                    t_mention_pos_list_1 = mention_pos_list_1[j*4:j*4+4]
                    t_mention_pos_list_2 = mention_pos_list_2[j*4:j*4+4]
                    for k in range(len(t_text_batch_1)):
                        k_text_batch = t_text_batch_1[k]
                        k_text_batch = k_text_batch.lower()
                        k_text_batch = k_text_batch.replace("[unk]", "[UNK]")
                        k_text_batch = k_text_batch.replace("[pad]", "[unused1]")
                        t_text_batch_1[k]=k_text_batch
                    
                    for k in range(len(t_text_batch_2)):
                        k_text_batch = t_text_batch_2[k]
                        k_text_batch = k_text_batch.lower()
                        k_text_batch = k_text_batch.replace("[unk]", "[UNK]")
                        k_text_batch = k_text_batch.replace("[pad]", "[unused1]")
                        t_text_batch_2[k]=k_text_batch
                        
                    encoding_1 = self.tokenizer(t_text_batch_1, return_tensors='pt', padding=True, truncation=True)
                    input_ids_1 = encoding_1['input_ids']
                    pre_padding_list_1, post_padding_list_1 = self.get_padding(input_ids_1)
                    dic_list_1_1, dic_list_2_1 = self.map_input_ids(pre_padding_list_1, post_padding_list_1, t_text_batch_1, input_ids_1)
                    
                    encoding_2 = self.tokenizer(t_text_batch_2, return_tensors='pt', padding=True, truncation=True)
                    input_ids_2 = encoding_2['input_ids']
                    pre_padding_list_2, post_padding_list_2 = self.get_padding(input_ids_2)
                    dic_list_1_2, dic_list_2_2 = self.map_input_ids(pre_padding_list_2, post_padding_list_2, t_text_batch_2, input_ids_2)
                    
                    input_ids_1 = input_ids_1.to(self.device)
                    attention_mask_1 = encoding_1['attention_mask']
                    attention_mask_1 = attention_mask_1.to(self.device)
                    input_ids_2 = input_ids_2.to(self.device)
                    attention_mask_2 = encoding_2['attention_mask']
                    attention_mask_2 = attention_mask_2.to(self.device)
                    
                    t_batch_label = batch_label[j*4:j*4+4]
                    labels = torch.tensor(t_batch_label).unsqueeze(0)                
                    labels = labels.to(self.device)
                    
                    t_batch_Y_1 = batch_Y_1[j*4:j*4+4]
                    t_batch_Y_1 = self.create_padding(t_batch_Y_1,dic_list_1_1, dic_list_2_1)
                    Y_1 = torch.tensor(t_batch_Y_1)
                    Y_1 = Y_1.to(self.device)
                    
                    t_batch_Y_2 = batch_Y_2[j*4:j*4+4]
                    t_batch_Y_2 = self.create_padding(t_batch_Y_2,dic_list_1_2, dic_list_2_2)
                    Y_2 = torch.tensor(t_batch_Y_2)
                    Y_2 = Y_2.to(self.device)
                    
                    mapped_mention_pos_list_1 = []
                    for h in range(len(t_mention_pos_list_1)):
                        mention_pos = t_mention_pos_list_1[h]
                        current_dic = dic_list_2_1[h]
                        mapped_mention_pos_list = current_dic[mention_pos]
                        mapped_mention_pos = mapped_mention_pos_list[-1]
                        mapped_mention_pos_list_1.append(mapped_mention_pos)
                        
                    mapped_mention_pos_list_2 = []
                    for h in range(len(t_mention_pos_list_2)):
                        mention_pos = t_mention_pos_list_2[h]
                        current_dic = dic_list_2_2[h]
                        mapped_mention_pos_list = current_dic[mention_pos]
                        mapped_mention_pos = mapped_mention_pos_list[-1]
                        mapped_mention_pos_list_2.append(mapped_mention_pos)
                        
                    outputs = model(input_ids_1=input_ids_1, input_ids_2=input_ids_2, Y_1=Y_1, Y_2=Y_2, attention_mask_1=attention_mask_1, attention_mask_2=attention_mask_2, labels=labels, mention_pos_list_1=mapped_mention_pos_list_1, mention_pos_list_2=mapped_mention_pos_list_2, margin=margin)
                    loss = outputs[0]                      
                    loss.backward()                
                    optimizer.step()
                    torch.cuda.empty_cache()
                    del encoding_1
                    del encoding_2
                    del input_ids_1
                    del input_ids_2
                    del t_text_batch_1
                    del t_text_batch_2
                    del t_batch_Y_1
                    del t_batch_Y_2
                    del t_batch_label
                    del attention_mask_1
                    del attention_mask_2
                    del labels
                    del Y_1
                    del Y_2
                    del outputs
                    del pre_padding_list_1
                    del post_padding_list_1
                    del pre_padding_list_2
                    del post_padding_list_2
                    del dic_list_1_1
                    del dic_list_2_1
                    del dic_list_1_2
                    del dic_list_2_2
                    del t_mention_pos_list_1
                    del t_mention_pos_list_2
                    del mapped_mention_pos_list_1
                    del mapped_mention_pos_list_2
                    gc.collect()
                
                torch.cuda.empty_cache()
                del text_batch_1
                del text_batch_2
                del batch_Y_1
                del batch_Y_2
                del mention_pos_list_1
                del mention_pos_list_2
                del batch_label
                gc.collect()
                
                s = s+1

            end_time = time.time()
            print ('the training took: %d(s)' % (end_time - start_time))
            
            if i > 1:
                self.update_learning_rate(2, optimizer)
                
            start_time = time.time()
            
            valid_losses, total_correct_dev, total_size_dev = self.get_model_development_data(development_data_dir, model, margin, dropout_rate)
            valid_loss = numpy.average(valid_losses)       
            print_msg = (f'valid_loss: {valid_loss:.5f}')
            print(print_msg)                     
            valid_losses = []
                
            print ('Accuracy of the trained model on validation set %f' % (total_correct_dev / total_size_dev))
            
            best_score = total_correct_dev * 1.0 / total_size_dev
            early_stopping(best_score, model)
            if early_stopping.early_stop:
                print("Early stopping")
                EARLY_STOP = True
            
            end_time = time.time()
            print ('the validation took: %d(s)' % (end_time - start_time))

    
def main():    
    parser = argparse.ArgumentParser('LSTM models')

    parser.add_argument('--conll_data_dir',
                    type=str,
                    default='change_pov/preprocessing/generate_training_data/output_training_array/conll_padding',
                    help='Directory to the conll training data.')
    parser.add_argument('--development_data_dir',
                    type=str,
                    default='conll_data/dev_transformer/',
                    help='Directory of the development data.')
    parser.add_argument('--pov_gold_testing_data_dir',
                    type=str,
                    default='pov_data_transformer_gold/',
                    help='Path of the testing data.')
    parser.add_argument('--pov_auto_dev_data_dir',
                    type=str,
                    default='pov_data_transformer_auto/dev/',
                    help='Path of the testing data.')
    parser.add_argument('--pov_auto_testing_data_dir',
                    type=str,
                    default='pov_data_transformer_auto/test/',
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
    parser.add_argument('--learning_rate', 
                    type=float,
                    default=12e-5,
                    help='Initial learning rate.')
    parser.add_argument('--margin',
                    type=float,
                    default=0.1,
                    help='Margin for the ranking loss.')
    parser.add_argument('--dropout_rate',
                    type=float,
                    default=0.1,
                    help='Dropout rate.')
    parser.add_argument('--patience',
                    type=int,
                    default=10,
                    help='Patience for early stopping.')

    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()

    pov_transformer = POVTransformer(0)
    pov_transformer.train_and_evaluate(FLAGS)
    
if __name__ == "__main__":
    main()
