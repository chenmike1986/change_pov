# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 08:38:44 2021

@author: jayzohio
"""

from openprompt.pipeline_base import PromptForGeneration
from openprompt.prompts.generation_verbalizer import GenerationVerbalizer
from tokenizers import PreTokenizedString
from tqdm import tqdm
from my_huggingface_dataset import PROCESSORS
import torch
from openprompt.data_utils.utils import InputExample
import argparse
import numpy as np

from openprompt import PromptDataLoader
from openprompt.prompts import ManualVerbalizer
from my_soft_template import SoftTemplate
from openprompt import PromptForClassification
import time
import os
import re
from openprompt.utils.crossfit_metrics import evaluate as crossfit_evaluate

from os import listdir
from os.path import isfile, join

import random
from openprompt.utils.reproduciblity import set_seed
from openprompt.plms.seq2seq import T5TokenizerWrapper, T5LMTokenizerWrapper
from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration
from openprompt.data_utils.data_sampler import FewShotSampler
from openprompt.plms import load_plm
from transformers import  AdamW, get_linear_schedule_with_warmup,get_constant_schedule_with_warmup  # use AdamW is a standard practice for transformer 
from transformers.optimization import Adafactor, AdafactorSchedule  # use Adafactor is the default setting for T5

from pytorchtools import EarlyStopping
from sys import exit

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
    
def get_cluster_data_from_line(line):       
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
  
def get_invalid_cluster_data_from_line(line):       
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

def get_focus_mention_from_line(line):
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

def get_distance_info_from_line(line):
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

def get_related_mention_from_line(line):
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

def get_raw_sequence_from_line(line):
    raw_sequence = []
    line = line.strip()
    line_parts = line.split()
    if line.startswith("<raw_sequence>:"):
        for i in range(1, len(line_parts)):
            raw_sequence.append(line_parts[i])
    else:
        print("There is an error in the dataset 27.")
        
    return raw_sequence

def get_postag_sequence_from_line(line):
    postag_sequence = []
    line = line.strip()
    line_parts = line.split()
    if line.startswith("<postags>:"):
        for i in range(1, len(line_parts)):
            postag_sequence.append(line_parts[i])
    else:
        print("There is an error in the dataset 28.")
        
    return postag_sequence

def get_related_sentence_from_line(line):
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

def get_token_index_info_from_line(line):
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

def get_original_pre_mention_sequence_from_line(line):
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

def get_converted_pre_mention_sequence_from_line(line):
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

def get_pre_mention_info_from_line(line):
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

def get_pre_mention_cluster_id_sequence_from_line(line):
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

def get_pre_mention_distance_sequence_from_line(line):
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

def get_original_post_mention_sequence_from_line(line):
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

def get_converted_post_mention_sequence_from_line(line):
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

def get_post_mention_info_from_line(line):
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

def get_post_mention_cluster_id_sequence_from_line(line):
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

def get_post_mention_distance_sequence_from_line(line):
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

def read_file(file_path):
    raw_document_data_list = []
    
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
                raw_document_data_list.append(raw_document_data)
                document_text = ""
                gold_cluster_id_to_cluster_data = {}
                auto_cluster_id_to_cluster_data = {}
                gold_invalid_cluster_id_to_cluster_data = {}
                auto_invalid_cluster_id_to_cluster_data = {}
                raw_sequence_data_list = []
            raw_cluster_data = get_cluster_data_from_line(line)
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
                raw_cluster_data = get_invalid_cluster_data_from_line(line)
                if line.startswith("<invalid_cluster_id>:") or line.startswith("<gold_invalid_cluster_id>:"):
                    gold_invalid_cluster_id_to_cluster_data[raw_cluster_data._cluster_id] = raw_cluster_data
                else:
                    auto_invalid_cluster_id_to_cluster_data[raw_cluster_data._cluster_id]=raw_cluster_data
        elif line.startswith("<focus_mention>:") or line.startswith("<original_focus_mention>:") or line.startswith("<identified_mention>:") or line.startswith("<identified_verb>:"):
            is_identified_mention, is_annotated_mention, is_identified_verb, is_annotated_verb, start_pos, identified_original_focus_mention, annotated_original_focus_mention, identified_converted_focus_mention, annotated_converted_focus_mention, identified_original_verb, annotated_original_verb, identified_converted_verb, annotated_converted_verb, original_focus_mention, converted_focus_mention, focus_mention_sentence_num, focus_mention_cluster_id, focus_mention_index_in_sentence, is_subject, is_object = get_focus_mention_from_line(line)
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
            result = get_related_mention_from_line(line)
            related_mention = result[0]
            related_mention_cluster_id = result[1]
            original_related_mention_text = result[2]
            converted_related_mention_text = result[3]
            related_mentions.append(related_mention)
            related_mention_cluster_ids.append(related_mention_cluster_id)
            related_original_mention_text_list.append(original_related_mention_text)
            related_converted_mention_text_list.append(converted_related_mention_text)
        elif line.startswith("<raw_sequence>:"):
            raw_sequence = get_raw_sequence_from_line(line)
        elif line.startswith("<postags>:"):
            postag_sequence = get_postag_sequence_from_line(line)       
        elif line.startswith("<sentence_num>:"):
            sentence_result = get_related_sentence_from_line(line)
            related_sentence_num_to_sentence_text[sentence_result[0]] = sentence_result[1]
        elif line.startswith("<start_token_index_in_sentence>:"):
            token_index_result = get_token_index_info_from_line(line)
            start_token_index_in_sentence = token_index_result[0]
            end_token_index_in_sentence = token_index_result[1]
        elif line.startswith("<pre_mention_sequence>:") or line.startswith("<original_pre_mention_sequence>:"):
            original_pre_mention_sequence = get_original_pre_mention_sequence_from_line(line)
        elif line.startswith("<converted_pre_mention_sequence>:"):
            converted_pre_mention_sequence = get_converted_pre_mention_sequence_from_line(line)
        elif line.startswith("<pre_mention_text>:"):
            pre_mention_info = get_pre_mention_info_from_line(line)
            pre_mention_info_list.append(pre_mention_info)
        elif line.startswith("<pre_mention_cluster_id_sequence>:"):
            pre_mention_cluster_id_sequence = get_pre_mention_cluster_id_sequence_from_line(line)
        elif line.startswith("<pre_mention_distance_sequence>:"):
            pre_mention_distance_sequence = get_pre_mention_distance_sequence_from_line(line)
        elif line.startswith("<post_mention_sequence>:") or line.startswith("<original_post_mention_sequence>:"):
            original_post_mention_sequence = get_original_post_mention_sequence_from_line(line)
        elif line.startswith("<converted_post_mention_sequence>:"):
            converted_post_mention_sequence = get_converted_post_mention_sequence_from_line(line)
        elif line.startswith("<post_mention_text>:"):
            post_mention_info = get_post_mention_info_from_line(line)
            post_mention_info_list.append(post_mention_info)
        elif line.startswith("<post_mention_cluster_id_sequence>:"):
            post_mention_cluster_id_sequence = get_post_mention_cluster_id_sequence_from_line(line)
        elif line.startswith("<post_mention_distance_sequence>:"):
            post_mention_distance_sequence = get_post_mention_distance_sequence_from_line(line)
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
        raw_document_data_list.append(raw_document_data)
            
    f.close()
    
    return raw_document_data_list
 
def prune_mention_set_dev(mention_set, focus_mention):   
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


def prune_mention_set_test(mention_set, focus_mention):    
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

def get_similar_clusters(cluster_id_to_cluster_data):
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

def get_updated_sentence(raw_sequence, index_in_sentence, mention, original_mention_len):
    updated_raw_sequence = []

    mention_parts = mention.split()
    
    for i in range(index_in_sentence):
        updated_raw_sequence.append(raw_sequence[i])
        
    for i in range(len(mention_parts)):
        updated_raw_sequence.append(mention_parts[i])
        
    for i in range(index_in_sentence + original_mention_len, len(raw_sequence)):
        updated_raw_sequence.append(raw_sequence[i])

    return updated_raw_sequence

def get_updated_sequence(raw_sequence, related_mentions, related_mention_cluster_ids, focus_mention_cluster_id, before_pos):
    updated_raw_sequence = []
    
    for i in range(len(raw_sequence)):
        
        in_related_mention = False
        for j in range(len(related_mentions)):
            related_mention = related_mentions[j]
            related_mention_cluster_id = related_mention_cluster_ids[j]
            start_pos = related_mention[0]
            end_pos = related_mention[1]
            if start_pos >= before_pos:
                continue
            
            if i == start_pos and i == end_pos:
                if related_mention_cluster_id == focus_mention_cluster_id:
                    updated_raw_sequence.append('<F>')
                    updated_raw_sequence.append(raw_sequence[i])
                    updated_raw_sequence.append('</F>')
                else:
                    updated_raw_sequence.append('<E>')
                    updated_raw_sequence.append(raw_sequence[i])
                    updated_raw_sequence.append('</E>')
                in_related_mention = True
            elif i == start_pos:
                if related_mention_cluster_id == focus_mention_cluster_id:
                    updated_raw_sequence.append('<F>')
                    updated_raw_sequence.append(raw_sequence[i])
                else:
                    updated_raw_sequence.append('<E>')
                    updated_raw_sequence.append(raw_sequence[i])
                in_related_mention = True
            elif i > start_pos and i < end_pos:
                updated_raw_sequence.append(raw_sequence[i])
                in_related_mention = True
            elif i == end_pos:
                if related_mention_cluster_id == focus_mention_cluster_id:                       
                    updated_raw_sequence.append(raw_sequence[i])
                    updated_raw_sequence.append('</F>')
                else:                       
                    updated_raw_sequence.append(raw_sequence[i])
                    updated_raw_sequence.append('</E>')
                in_related_mention = True
                
            if in_related_mention:
                break
        
        if not in_related_mention:
            updated_raw_sequence.append(raw_sequence[i])
            
    return updated_raw_sequence

def get_updated_sequence_2(raw_sequence, related_mentions, related_mention_cluster_ids, related_converted_mention_text_list, focus_mention_cluster_id, before_pos):
    updated_raw_sequence = []
    
    for i in range(len(raw_sequence)):
        
        in_related_mention = False
        for j in range(len(related_mentions)):
            related_mention = related_mentions[j]
            related_mention_cluster_id = related_mention_cluster_ids[j]
            related_converted_mention_text = related_converted_mention_text_list[j]
            start_pos = related_mention[0]
            end_pos = related_mention[1]
            if start_pos >= before_pos:
                continue
            
            if i == start_pos and i == end_pos:
                if related_mention_cluster_id == focus_mention_cluster_id:
                    updated_raw_sequence.append('<F>')
                    updated_raw_sequence.append(related_converted_mention_text)
                    updated_raw_sequence.append('</F>')
                else:
                    updated_raw_sequence.append('<E>')
                    updated_raw_sequence.append(related_converted_mention_text)
                    updated_raw_sequence.append('</E>')
                in_related_mention = True
            elif i == start_pos:
                if related_mention_cluster_id == focus_mention_cluster_id:
                    updated_raw_sequence.append('<F>')
                    updated_raw_sequence.append(related_converted_mention_text)
                else:
                    updated_raw_sequence.append('<E>')
                    updated_raw_sequence.append(related_converted_mention_text)
                in_related_mention = True
            elif i > start_pos and i < end_pos:
                in_related_mention = True
            elif i == end_pos:
                if related_mention_cluster_id == focus_mention_cluster_id:                       
                    updated_raw_sequence.append('</F>')
                else:                       
                    updated_raw_sequence.append('</E>')
                in_related_mention = True
                
            if in_related_mention:
                break
        
        if not in_related_mention:
            updated_raw_sequence.append(raw_sequence[i])
            
    return updated_raw_sequence

parser = argparse.ArgumentParser("")
parser.add_argument("--shot", type=int, default=-1)
parser.add_argument("--seed", type=int, default=144)
parser.add_argument("--plm_eval_mode", default=False, help="whether to turn off the dropout in the freezed model. Set to true to turn off.")
parser.add_argument("--tune_plm", default=True)
parser.add_argument("--model", type=str, default='t5', help="We test both t5 and t5-lm in this scripts, the corresponding tokenizerwrapper will be automatically loaded.")
parser.add_argument("--model_name_or_path", default='t5/')
parser.add_argument("--project_root", default="/users/mchen40/mention_selection/", help="The project root in the file system, i.e. the absolute path of OpenPrompt")
parser.add_argument("--template_id", type=int, default=0)
parser.add_argument("--verbalizer_id", type=int, default=0)
parser.add_argument("--data_dir", type=str, default="conll_json/") # sometimes, huggingface datasets can not be automatically downloaded due to network issue, please refer to 0_basic.py line 15 for solutions. 
parser.add_argument("--dev_data_dir", type=str, default="conll_data/dev/")
parser.add_argument("--conll_test_data_dir", type=str, default="conll_data/test/")
parser.add_argument("--pov_gold_test_data_dir", type=str, default="pov_data_gold/")
parser.add_argument("--pov_auto_test_data_dir", type=str, default="pov_data_auto/test/")
parser.add_argument("--dataset",type=str, default="conll")
parser.add_argument("--result_file", type=str, default="results.txt")
parser.add_argument("--max_steps", default=500000, type=int)
parser.add_argument("--prompt_lr", type=float, default=0.3)
parser.add_argument("--warmup_step_prompt", type=int, default=500)
parser.add_argument("--init_from_vocab", default=True)
parser.add_argument("--eval_every_steps", type=int, default=2900)
parser.add_argument("--soft_token_num", type=int, default=20)
parser.add_argument("--optimizer", type=str, default="Adafactor")
parser.add_argument("--is_training", default=False)
parser.add_argument("--model_path", type=str, default='plm-fine-tune-748.pt')

args = parser.parse_args()

args.result_file = os.path.join(args.project_root, args.result_file)

content_write = "="*20+"\n"
content_write += f"dataset {args.dataset}\t"
content_write += f"temp {args.template_id}\t"
content_write += f"verb {args.verbalizer_id}\t"
content_write += f"model {args.model}\t"
content_write += f"seed {args.seed}\t"
content_write += f"shot {args.shot}\t"
content_write += f"plm_eval_mode {args.plm_eval_mode}\t"
content_write += f"init_from_vocab {args.init_from_vocab}\t"
content_write += f"eval_every_steps {args.eval_every_steps}\t"
content_write += f"prompt_lr {args.prompt_lr}\t"
content_write += f"optimizer {args.optimizer}\t"
content_write += f"warmup_step_prompt {args.warmup_step_prompt}\t"
content_write += f"soft_token_num {args.soft_token_num}\t"
content_write += "\n"
print(content_write)

this_run_unicode = str(random.randint(0, 1e10))
set_seed(args.seed)

plm, tokenizer, model_config, WrapperClass = load_plm(args.model, args.model_name_or_path)
dataset = {}
Processor = PROCESSORS["super_glue.conll"]
dataset['train'] = Processor().get_train_examples(args.data_dir)
# dataset['validation'] = Processor().get_dev_examples(args.data_dir)
# dataset['test'] = Processor().get_test_examples(args.data_dir)
class_labels =Processor().get_labels()
scriptsbase = "CoNLL"
scriptformat = "txt"
max_seq_l = 480
dataset_decoder_max_length = 80
if args.tune_plm:
    batchsize_t = 4
    batchsize_e = 4 
    gradient_accumulation_steps = 8
    model_parallelize = True
else:
    batchsize_t = 8
    batchsize_e = 4 
    gradient_accumulation_steps = 4
    model_parallelize = False

#mytemplate = SoftTemplate(model=plm, tokenizer=tokenizer, num_tokens=args.soft_token_num, initialize_from_vocab=args.init_from_vocab).from_file(f"scripts/{scriptsbase}/soft_template.txt", choice=args.template_id)
mytemplate = SoftTemplate(model=plm, tokenizer=tokenizer, num_tokens=args.soft_token_num, initialize_from_vocab=args.init_from_vocab)
if os.path.exists(f"scripts/{scriptsbase}/generation_verbalizer.{scriptformat}"):
    myverbalizer = GenerationVerbalizer(tokenizer, classes=class_labels, is_rule=True).from_file(f"scripts/{scriptsbase}/generation_verbalizer.{scriptformat}", choice=args.verbalizer_id)
else:
    myverbalizer = GenerationVerbalizer(tokenizer, classes=class_labels, is_rule=False).from_file(f"scripts/{scriptsbase}/manual_verbalizer.{scriptformat}", choice=args.verbalizer_id)

use_cuda = True
prompt_model = PromptForGeneration(plm=plm,template=mytemplate, freeze_plm=(not args.tune_plm), plm_eval_mode=args.plm_eval_mode)
if use_cuda:
    prompt_model=  prompt_model.cuda()

if model_parallelize:
    prompt_model.parallelize()

if args.model_path is not None:
    prompt_model.load_state_dict(torch.load(args.model_path))
    
train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, verbalizer=myverbalizer, tokenizer=tokenizer, # be sure to add verbalizer 
    tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l, decoder_max_length=dataset_decoder_max_length,  # be sure to use larger decoder_max_length for teacher forcing.
    batch_size=batchsize_t,shuffle=False, teacher_forcing=True, predict_eos_token=True,  # be sure to use teacher_forcing and predict_eos_token=True
    truncate_method="tail")

# validation_dataloader = PromptDataLoader(dataset=dataset["validation"], template=mytemplate, verbalizer=myverbalizer, tokenizer=tokenizer, 
#     tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l, decoder_max_length=3, 
#     batch_size=batchsize_e,shuffle=False, teacher_forcing=False, predict_eos_token=False, # predict_eos_token=True or False are both ok 
#     truncate_method="tail")

# test_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, verbalizer=myverbalizer, tokenizer=tokenizer, 
#     tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l, decoder_max_length=3, 
#     batch_size=batchsize_e,shuffle=False, teacher_forcing=False, predict_eos_token=False,
#     truncate_method="tail")

# print("truncate rate: {}".format(test_dataloader.tokenizer_wrapper.truncate_rate), flush=True)

generation_arguments = {
    "max_length": dataset_decoder_max_length,
}

def evaluate_1(data_dir, prompt_model, mytemplate, myverbalizer, tokenizer, WrapperClass):
    total_correct = 0
    total_size = 0
    dataloader = None
    f = open('result-prompt-tuning-conll.txt', 'w+', encoding='utf-8')
    
    files = [join(data_dir,ff) for ff in listdir(data_dir) if isfile(join(data_dir,ff))]
    for data_path in files:
        raw_document_data_list = read_file(data_path)
        
        previous_mentions = []
        previous_mention_cluster_ids = []
        correct_previous_mentions = []
    
        COUNT = 0
        count = 0
        for raw_document_data in raw_document_data_list:
            COUNT = COUNT + 1
            cluster_id_to_cluster_data = raw_document_data._gold_cluster_id_to_cluster_data
            invalid_cluster_id_to_cluster_data = raw_document_data._gold_invalid_cluster_id_to_cluster_data
            raw_sequence_data_list = raw_document_data._raw_sequence_data_list
            results = get_similar_clusters(cluster_id_to_cluster_data)
            cluster_id_to_he_clusters = results[0]
            cluster_id_to_she_clusters = results[1]
            cluster_id_to_it_clusters = results[2]
            cluster_id_to_they_clusters = results[3]
                  
            for raw_sequence_data in raw_sequence_data_list:    
                count = count + 1
                focus_mention_cluster_id = raw_sequence_data._focus_mention_cluster_id
                focus_mention = raw_sequence_data._original_focus_mention
                focus_mention_parts = focus_mention.split()
                focus_mention_cluster_data = cluster_id_to_cluster_data[focus_mention_cluster_id]
                temp_mention_set = focus_mention_cluster_data._mention_list
                mention_set = prune_mention_set_dev(temp_mention_set, focus_mention)
            
                related_mentions = raw_sequence_data._related_mentions
                related_mention_cluster_ids = raw_sequence_data._related_mention_cluster_ids
            
                updated_related_mentions = []
                clean_related_mentions = []
                clean_related_mention_cluster_ids = []
                effective_count = 0
                for i in range(len(related_mention_cluster_ids)):
                    related_mention_cluster_id = related_mention_cluster_ids[i]
                    if related_mention_cluster_id == focus_mention_cluster_id:
                        updated_related_mentions.append(related_mentions[i])
                        clean_related_mentions.append(related_mentions[i])
                        clean_related_mention_cluster_ids.append(related_mention_cluster_id)
                        if related_mentions[i][0] <= 50:
                            effective_count = effective_count + 1
                    elif (related_mention_cluster_id in cluster_id_to_he_clusters and focus_mention_cluster_id in cluster_id_to_he_clusters) or (related_mention_cluster_id in cluster_id_to_she_clusters and focus_mention_cluster_id in cluster_id_to_she_clusters) or (related_mention_cluster_id in cluster_id_to_it_clusters and focus_mention_cluster_id in cluster_id_to_it_clusters) or (related_mention_cluster_id in cluster_id_to_they_clusters):
                        clean_related_mentions.append(related_mentions[i])
                        clean_related_mention_cluster_ids.append(related_mention_cluster_id)
                        if related_mentions[i][0] <= 50:
                            effective_count = effective_count + 1
        
                raw_sequence = raw_sequence_data._raw_sequence
                related_sentence_num_to_sentence_text = raw_sequence_data._related_sentence_num_to_sentence_text
                start_token_index_in_sentence = raw_sequence_data._start_token_index_in_sentence
                end_token_index_in_sentence = raw_sequence_data._end_token_index_in_sentence
                sorted_sentence_num_list = sorted(related_sentence_num_to_sentence_text.keys())
                first_sentence_num = sorted_sentence_num_list[0]
                last_sentence_num = sorted_sentence_num_list[len(sorted_sentence_num_list)-1]
                first_sentence_text = related_sentence_num_to_sentence_text[first_sentence_num]
                first_sentence_text_parts = first_sentence_text.split()
                last_sentence_text = related_sentence_num_to_sentence_text[last_sentence_num]
                last_sentence_text_parts = last_sentence_text.split()
                
                updated_raw_text = ''
                for i in range(len(raw_sequence)):
                    rrss = raw_sequence[i]
                    rrss = rrss.replace("{", "")
                    rrss = rrss.replace("}","")
                    raw_sequence[i]=rrss
                    
                # same_entity_mentions = []
                # different_entity_mentions = []
                
                mentions_to_replace = []
                temp_dict = {}
                for related_mention in related_mentions:                  
                    if related_mention[0] <= 50:
                        mentions_to_replace.append(related_mention)
                        temp_dict[related_mention[0]] = related_mention
                    
                if len(mention_set) < 2:
                    if len(mentions_to_replace) == 0:
                        previous_mentions = []
                        previous_mention_cluster_ids = []
                        correct_previous_mentions = []
                    
                    previous_mentions.append(focus_mention)
                    previous_mention_cluster_ids.append(focus_mention_cluster_id)
                    correct_previous_mentions.append(focus_mention)
                   
                    total_size =  total_size + 1
                    total_correct = total_correct + 1
                    continue
                
                if len(mentions_to_replace) == 0:
                    previous_mentions = []
                    previous_mention_cluster_ids = []
                    correct_previous_mentions = []
                    
                    # same_entity_mentions = []
                    # different_entity_mentions = []
                    to_replace_mentions = []
                    to_replace_mention_start_pos_to_cluster_id={}
                    for i in range(len(related_mentions)):   
                        related_mention = related_mentions[i]
                        related_mention_cluster_id = related_mention_cluster_ids[i]
                        start_pos = related_mention[0]
                        end_pos = related_mention[1]

                        if start_pos <= 50:
                            # related_mention_text = ""
                            # for j in range(start_pos, end_pos+1):
                            #     related_mention_text = related_mention_text + ' ' + raw_sequence[j]
                            # related_mention_text = related_mention_text.strip()
                            # if related_mention_cluster_id == focus_mention_cluster_id:
                            #     same_entity_mentions.append(related_mention_text)
                            # else:
                            #     different_entity_mentions.append(related_mention_text)

                            continue
                        else:
                            to_replace_mentions.append(related_mention)
                            to_replace_mention_start_pos_to_cluster_id[start_pos]=related_mention_cluster_id
                            
                    sorted_mention_list = sorted(to_replace_mentions, key=lambda x: x[0], reverse=True)
                    updated_raw_sequence = raw_sequence
                    for sorted_mention in sorted_mention_list:
                        start_pos = sorted_mention[0]
                        end_pos = sorted_mention[1]
                        to_replace_mention_cluster_id = to_replace_mention_start_pos_to_cluster_id[start_pos]
                        if to_replace_mention_cluster_id == focus_mention_cluster_id:
                            updated_raw_sequence = get_updated_sentence(updated_raw_sequence, start_pos, '<F> <unk> </F>', end_pos - start_pos + 1)
                        else:
                            updated_raw_sequence = get_updated_sentence(updated_raw_sequence, start_pos, '<E> <unk> </E>', end_pos - start_pos + 1)
                            
                    updated_raw_sequence = get_updated_sentence(updated_raw_sequence, 50, '<F> @placeholder </F>', len(focus_mention_parts))

                    for i in range(start_token_index_in_sentence-1, -1, -1):
                        ttpp = first_sentence_text_parts[i]
                        ttpp = ttpp.replace("{", "")
                        ttpp = ttpp.replace("}", "")
                        updated_raw_sequence.insert(0,ttpp)
                        
                    for i in range(end_token_index_in_sentence, len(last_sentence_text_parts)):
                        ttpp = last_sentence_text_parts[i]
                        ttpp = ttpp.replace("{", "")
                        ttpp = ttpp.replace("}", "")
                        updated_raw_sequence.append(ttpp)
                     
                    pre_offset = start_token_index_in_sentence
                    
                    initial_related_mentions = raw_sequence_data._related_mentions
                    related_mentions = []
                    for initial_related_mention in initial_related_mentions:
                        final_related_mention = []
                        final_related_mention.append(initial_related_mention[0]+pre_offset)
                        final_related_mention.append(initial_related_mention[1]+pre_offset)
                        related_mentions.append(final_related_mention)
                    
                    final_raw_sequence = get_updated_sequence(updated_raw_sequence, related_mentions, related_mention_cluster_ids, focus_mention_cluster_id, 50+pre_offset)
                    updated_raw_text = ''
                    for piece in final_raw_sequence:
                        updated_raw_text = updated_raw_text + ' ' + piece
                    updated_raw_text = updated_raw_text.strip()
                else:     
                    # same_entity_mentions = []
                    # different_entity_mentions = []
                    to_replace_mentions = []
                    not_replaced_mentions = []
                    to_replace_mention_start_pos_to_cluster_id={}
                    for i in range(len(related_mentions)):   
                        related_mention = related_mentions[i]
                        related_mention_cluster_id = related_mention_cluster_ids[i]
                        start_pos = related_mention[0]
                        end_pos = related_mention[1]

                        if start_pos <= 50:
                            not_replaced_mentions.append(related_mention)
                        else:
                            to_replace_mentions.append(related_mention)
                            to_replace_mention_start_pos_to_cluster_id[start_pos]=related_mention_cluster_id
                            
                    sorted_mention_list = sorted(to_replace_mentions, key=lambda x: x[0], reverse=True)
                    updated_raw_sequence = raw_sequence
                    for sorted_mention in sorted_mention_list:
                        start_pos = sorted_mention[0]
                        end_pos = sorted_mention[1]
                        to_replace_mention_cluster_id = to_replace_mention_start_pos_to_cluster_id[start_pos]
                        if to_replace_mention_cluster_id == focus_mention_cluster_id:
                            updated_raw_sequence = get_updated_sentence(updated_raw_sequence, start_pos, '<F> <unk> </F>', end_pos - start_pos + 1)
                        else:
                            updated_raw_sequence = get_updated_sentence(updated_raw_sequence, start_pos, '<E> <unk> </E>', end_pos - start_pos + 1)
                                                                  
                    updated_raw_sequence = get_updated_sentence(updated_raw_sequence, 50, '<F> @placeholder </F>', len(focus_mention_parts))
                    
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
                    
                    # related_mentions_to_update = {}
                    # related_mentions_to_update_cluster_ids = {}
                    y=0
                    z=0
                    #focus_mention_offset = 0
                    for sorted_key in temp_dict_sorted_keys:
                        previous_mention_index = sorted_key_to_previous_mention_index[sorted_key]
                        previous_mention = previous_mentions[previous_mention_index]
                        correct_previous_mention = correct_previous_mentions[previous_mention_index]
                        correct_previous_mention_parts = correct_previous_mention.split()
                                               
                        previous_mention_cluster_id = previous_mention_cluster_ids[previous_mention_index]
                        if (previous_mention_cluster_id == focus_mention_cluster_id) or (previous_mention_cluster_id in cluster_id_to_he_clusters and focus_mention_cluster_id in cluster_id_to_he_clusters) or (previous_mention_cluster_id in cluster_id_to_she_clusters and focus_mention_cluster_id in cluster_id_to_she_clusters) or (previous_mention_cluster_id in cluster_id_to_it_clusters and focus_mention_cluster_id in cluster_id_to_it_clusters) or (previous_mention_cluster_id in cluster_id_to_they_clusters):
                            temp_related_mention = clean_related_mentions[effective_count-y-1]
                            temp_related_mention_cluster_id = clean_related_mention_cluster_ids[effective_count-y-1]
                            y=y+1
                            
                            if previous_mention.lower() != correct_previous_mention.lower():
                                updated_previous_mention = ''
                                if previous_mention_cluster_id == focus_mention_cluster_id:
                                    updated_previous_mention = '<F> ' + previous_mention + ' </F>'
                                else:
                                    updated_previous_mention = '<E> ' + previous_mention + ' </E>'
                                updated_raw_sequence = get_updated_sentence(updated_raw_sequence, temp_related_mention[0], updated_previous_mention, len(correct_previous_mention_parts))
                                #focus_mention_offset = focus_mention_offset + len(updated_previous_mention.split())- len(correct_previous_mention_parts)
                                # related_mentions_to_update[temp_related_mention[0]]=previous_mention
                                # related_mentions_to_update_cluster_ids[temp_related_mention[0]]=temp_related_mention_cluster_id
                            else:
                                updated_correct_previous_mention = ''
                                if previous_mention_cluster_id == focus_mention_cluster_id:
                                    updated_correct_previous_mention = '<F> ' + correct_previous_mention + ' </F>'
                                else:
                                    updated_correct_previous_mention = '<E> ' + correct_previous_mention + ' </E>'
                                updated_raw_sequence = get_updated_sentence(updated_raw_sequence, temp_related_mention[0], updated_correct_previous_mention, len(correct_previous_mention_parts))
                                # related_mentions_to_update[temp_related_mention[0]]=previous_mention
                                # related_mentions_to_update_cluster_ids[temp_related_mention[0]]=temp_related_mention_cluster_id
                        else:
                            temp_related_mention = not_replaced_mentions[len(not_replaced_mentions)-1-y-z]
                            updated_correct_previous_mention = ''
                            if previous_mention_cluster_id == focus_mention_cluster_id:
                                updated_correct_previous_mention = '<F> ' + correct_previous_mention + ' </F>'
                            else:
                                updated_correct_previous_mention = '<E> ' + correct_previous_mention + ' </E>'
                            updated_raw_sequence = get_updated_sentence(updated_raw_sequence, temp_related_mention[0], updated_correct_previous_mention, len(correct_previous_mention_parts))
                            z=z+1
                            # related_mentions_to_update[sorted_key]=correct_previous_mention
                            # related_mentions_to_update_cluster_ids[sorted_key]=sorted_key_to_replaced_mention_cluster_id[sorted_key]
                            
                    # for i in range(len(related_mentions)):   
                    #     related_mention = related_mentions[i]
                    #     related_mention_cluster_id = related_mention_cluster_ids[i]
                    #     start_pos = related_mention[0]
                    #     end_pos = related_mention[1]

                    #     if start_pos <= 50:
                    #         if start_pos in related_mentions_to_update:
                    #             cluster_id_to_update = related_mentions_to_update_cluster_ids[start_pos]
                    #             mention_text_to_update = related_mentions_to_update[start_pos]
                    #             if cluster_id_to_update == focus_mention_cluster_id:
                    #                 same_entity_mentions.append(mention_text_to_update)
                    #             else:
                    #                 different_entity_mentions.append(mention_text_to_update)
                    #         else:       
                    #             print('wrong implementation')
                    
                    for i in range(start_token_index_in_sentence-1, -1, -1):
                        ttpp = first_sentence_text_parts[i]
                        ttpp = ttpp.replace("{", "")
                        ttpp = ttpp.replace("}", "")
                        updated_raw_sequence.insert(0,ttpp)
                            
                    for i in range(end_token_index_in_sentence, len(last_sentence_text_parts)):
                        ttpp = last_sentence_text_parts[i]
                        ttpp = ttpp.replace("{", "")
                        ttpp = ttpp.replace("}", "")
                        updated_raw_sequence.append(ttpp)
                            
                    updated_raw_text = ''
                    for piece in updated_raw_sequence:
                        updated_raw_text = updated_raw_text + ' ' + piece
                    updated_raw_text = updated_raw_text.strip()
                 
                total_size = total_size + 1
                idx = count
                guid = "{}".format(idx)
                
                meta1={}
                meta1['context']=updated_raw_text
                
                for i in range(len(mention_set)):
                    mention = mention_set[i]
                    mention = mention.replace("{", "")
                    mention = mention.replace("}", "")
                    mention_set[i]=mention
                    
                s1 = " <M> ".join(mention_set)
                s1 = "<M> " + s1
                meta1['entities']=s1
                # s2 = "| ".join(same_entity_mentions)
                # meta1['same_entity_mentions']=s2
                # s3 = "| ".join(different_entity_mentions)
                # meta1['different_entity_mentions']=s3
                
                focus_mention = focus_mention.replace("{","")
                focus_mention = focus_mention.replace("}", "")
                meta1['answer']=focus_mention
                
                example = InputExample(guid=guid, meta=meta1, label=0)
                examples = []
                examples.append(example)
                dataloader = PromptDataLoader(dataset=examples, template=mytemplate, verbalizer=myverbalizer, tokenizer=tokenizer, 
                    tokenizer_wrapper_class=WrapperClass, max_seq_length=1000, decoder_max_length=5, 
                    batch_size=1,shuffle=False, teacher_forcing=False, predict_eos_token=False, # predict_eos_token=True or False are both ok 
                    truncate_method="tail")
                
                predictions = []
                ground_truths = []
                updated_output_sentence=''
                for step, inputs in enumerate(dataloader):
                    if use_cuda:
                        inputs = inputs.cuda()
                    _, output_sentence = prompt_model.generate(inputs, **generation_arguments, verbose=False)
                    updated_output_sentence = output_sentence[0].replace("'s", " 's")
                    predictions.append(updated_output_sentence)
                    ground_truths.extend(inputs['tgt_text'])
                assert len(predictions)==len(ground_truths), (len(predictions), len(ground_truths))
                predictions = [prediction.strip() for prediction in predictions]
                ground_truths = [ground_truth.strip() for ground_truth in ground_truths]
                # shown one example
                for p in range(len(predictions)):
                    prediction = predictions[p]
                    prediction = prediction.replace(" ,", ",")
                    prediction = prediction.replace(" .", ".")
                    prediction = prediction.replace("``", "<unk>")
                    prediction = prediction.replace("Berry", "Barry")
                    prediction = prediction.replace("berry", "barry")
                    prediction = prediction.replace("o '", "o'")
                    prediction = prediction.replace("O '", "O'")
                    prediction = prediction.replace("did n't", "didn't")
                    prediction = prediction.replace("Did n't", "Didn't")
                    predictions[p]=prediction
                for p in range(len(ground_truths)):
                    ground_truth = ground_truths[p]
                    ground_truth = ground_truth.replace(" ,", ",")
                    ground_truth = ground_truth.replace(" .", ".")
                    ground_truth = ground_truth.replace("``", "<unk>")
                    ground_truth = ground_truth.replace("Berry", "Barry")
                    ground_truth = ground_truth.replace("berry", "barry")
                    ground_truth = ground_truth.replace("o '", "o'")
                    ground_truth = ground_truth.replace("O '", "O'")
                    ground_truth = ground_truth.replace("did n't", "didn't")
                    ground_truth = ground_truth.replace("Did n't", "Didn't")
                    ground_truths[p]=ground_truth

                score1 =  crossfit_evaluate(predictions, ground_truths, metric="ACC")
                score2 =  crossfit_evaluate(ground_truths, predictions, metric="ACC")
                if score1 == 0 and score2==0:
                    f.write(f"predictions {predictions[0]}, ground_truths {ground_truths[0]}")
                    f.write('\n')
                updated_output_sentence = updated_output_sentence.replace("{","")
                updated_output_sentence = updated_output_sentence.replace("}","")
                previous_mentions.append(updated_output_sentence)
                previous_mention_cluster_ids.append(focus_mention_cluster_id)
                correct_previous_mentions.append(focus_mention)
                if score1 == 1 or score2==1:
                    total_correct = total_correct + 1
    f.close()                
    return total_correct, total_size

def evaluate_2(data_dir, gold_setting, prompt_model, mytemplate, myverbalizer, tokenizer, WrapperClass):
    total_correct = 0
    total_size = 0
    
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
    
    gold_se_error = 0
    
    f = open('result.txt','w+',encoding='utf-8')
    files = [join(data_dir,ff) for ff in listdir(data_dir) if isfile(join(data_dir,ff))]
    for data_path in files:
        raw_document_data_list = read_file(data_path)
        
        previous_mentions = []
        previous_mention_cluster_ids = []
        correct_previous_mentions_1 = []
        correct_previous_mentions_2 = []
        
        count = 0
        for raw_document_data in raw_document_data_list:
            current_total_correct = 0
            current_total_size = 0
    
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
            
            gold_cluster_id_to_cluster_data = raw_document_data._gold_cluster_id_to_cluster_data
            auto_cluster_id_to_cluster_data = raw_document_data._auto_cluster_id_to_cluster_data
            gold_invalid_cluster_id_to_cluster_data = raw_document_data._gold_invalid_cluster_id_to_cluster_data
            auto_invalid_cluster_id_to_cluster_data = raw_document_data._auto_invalid_cluster_id_to_cluster_data
            
            raw_sequence_data_list = raw_document_data._raw_sequence_data_list
            results = None
            if gold_setting:
                results = get_similar_clusters(gold_cluster_id_to_cluster_data)
            else:
                results = get_similar_clusters(auto_cluster_id_to_cluster_data)
            cluster_id_to_he_clusters = results[0]
            cluster_id_to_she_clusters = results[1]
            cluster_id_to_it_clusters = results[2]
            cluster_id_to_they_clusters = results[3]
            
            for raw_sequence_data in raw_sequence_data_list: 
                count = count + 1
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
                        continue
                    elif is_identified_verb == 1 and is_annotated_verb == 0:
                        total_identified_verb_not_annotated = total_identified_verb_not_annotated + 1
                        current_total_identified_verb_not_annotated = current_total_identified_verb_not_annotated + 1
                        identified_original_verb = raw_sequence_data._identified_original_verb
                        identified_converted_verb = raw_sequence_data._identified_converted_verb
                        if identified_original_verb.lower() != identified_converted_verb.lower():
                            total_identified_verb_not_annotated_incorrectly_converted = total_identified_verb_not_annotated_incorrectly_converted + 1
                            current_total_identified_verb_not_annotated_incorrectly_converted = current_total_identified_verb_not_annotated_incorrectly_converted + 1
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
                        continue
                    
                original_focus_mention = raw_sequence_data._identified_original_focus_mention
                converted_focus_mention = raw_sequence_data._converted_focus_mention
                original_focus_mention_parts = original_focus_mention.split()
                converted_focus_mention_parts = converted_focus_mention.split()
                gold_focus_mention_cluster_data = gold_cluster_id_to_cluster_data[focus_mention_cluster_id]
                auto_focus_mention_cluster_data = auto_cluster_id_to_cluster_data[focus_mention_cluster_id]
                temp_mention_set = []
                if gold_setting:
                    temp_mention_set = gold_focus_mention_cluster_data._mention_list
                else:
                    temp_mention_set = auto_focus_mention_cluster_data._mention_list
                mention_set = prune_mention_set_test(temp_mention_set, original_focus_mention)
                
                if gold_setting:
                    mmm_se = False
                    for mmm in auto_focus_mention_cluster_data._mention_list:
                        if mmm.lower() == converted_focus_mention.lower():
                            mmm_se = True
                            break
                    if not mmm_se:
                        gold_se_error = gold_se_error + 1
                        
                related_mentions = raw_sequence_data._related_mentions
                related_mention_cluster_ids = raw_sequence_data._related_mention_cluster_ids
                related_converted_mention_text_list = raw_sequence_data._related_converted_mention_text_list
                
                updated_related_mentions = []
                clean_related_mentions = []
                clean_related_mention_cluster_ids = []
                effective_count = 0
                for i in range(len(related_mention_cluster_ids)):
                    related_mention_cluster_id = related_mention_cluster_ids[i]
                    if related_mention_cluster_id == focus_mention_cluster_id:
                        updated_related_mentions.append(related_mentions[i])
                        clean_related_mentions.append(related_mentions[i])
                        clean_related_mention_cluster_ids.append(related_mention_cluster_id)
                        if related_mentions[i][0] <= 50:
                            effective_count = effective_count + 1
                    elif (related_mention_cluster_id in cluster_id_to_he_clusters and focus_mention_cluster_id in cluster_id_to_he_clusters) or (related_mention_cluster_id in cluster_id_to_she_clusters and focus_mention_cluster_id in cluster_id_to_she_clusters) or (related_mention_cluster_id in cluster_id_to_it_clusters and focus_mention_cluster_id in cluster_id_to_it_clusters) or (related_mention_cluster_id in cluster_id_to_they_clusters):
                        clean_related_mentions.append(related_mentions[i])
                        clean_related_mention_cluster_ids.append(related_mention_cluster_id)
                        if related_mentions[i][0] <= 50:
                            effective_count = effective_count + 1
                            
                raw_sequence = raw_sequence_data._raw_sequence
                related_sentence_num_to_sentence_text = raw_sequence_data._related_sentence_num_to_sentence_text
                start_token_index_in_sentence = raw_sequence_data._start_token_index_in_sentence
                end_token_index_in_sentence = raw_sequence_data._end_token_index_in_sentence
                sorted_sentence_num_list = sorted(related_sentence_num_to_sentence_text.keys())
                first_sentence_num = sorted_sentence_num_list[0]
                last_sentence_num = sorted_sentence_num_list[len(sorted_sentence_num_list)-1]
                first_sentence_text = related_sentence_num_to_sentence_text[first_sentence_num]
                first_sentence_text_parts = first_sentence_text.split()
                last_sentence_text = related_sentence_num_to_sentence_text[last_sentence_num]
                last_sentence_text_parts = last_sentence_text.split()
                
                updated_raw_text = ''
                for i in range(len(raw_sequence)):
                    rrss = raw_sequence[i]
                    rrss = rrss.replace("{", "")
                    rrss = rrss.replace("}","")
                    raw_sequence[i]=rrss
                    
                # same_entity_mentions = []
                # different_entity_mentions = []
                
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
                        correct_previous_mentions_1 = []
                        correct_previous_mentions_2 = []
                        
                    previous_mentions.append(converted_focus_mention)
                    previous_mention_cluster_ids.append(focus_mention_cluster_id)
                    correct_previous_mentions_1.append(converted_focus_mention)                    
                    correct_previous_mentions_2.append(original_focus_mention)
                    
                    if gold_setting:
                        total_size = total_size + 1
                        current_total_size = current_total_size + 1
                        total_correct = total_correct + 1
                        current_total_correct = current_total_correct + 1
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

                        elif is_identified_mention == 1 and is_annotated_mention == 0:
                            total_identified_mention_not_annotated = total_identified_mention_not_annotated + 1
                            current_total_identified_mention_not_annotated = current_total_identified_mention_not_annotated + 1
                            identified_original_mention = raw_sequence_data._identified_original_focus_mention
                            if mention_set[0].lower() != identified_original_mention.lower():
                                total_identified_mention_not_annotated_incorrectly_converted = total_identified_mention_not_annotated_incorrectly_converted + 1
                                current_total_identified_mention_not_annotated_incorrectly_converted = current_total_identified_mention_not_annotated_incorrectly_converted + 1
                    continue
                
                no_correct_string_in_se = False
            
                if len(mentions_to_replace) == 0:
                    previous_mention_cluster_ids = []
                    correct_previous_mentions_1 = []
                    correct_previous_mentions_2 = []
                    previous_mentions = []
                    
                    # same_entity_mentions = []
                    # different_entity_mentions = []
                    to_replace_mentions = []
                    to_replace_mention_start_pos_to_cluster_id={}
                    for i in range(len(related_mentions)):   
                        related_mention = related_mentions[i]
                        related_mention_cluster_id = related_mention_cluster_ids[i]
                        start_pos = related_mention[0]
                        end_pos = related_mention[1]

                        if start_pos <= 50:
                            # related_mention_text = ""
                            # for j in range(start_pos, end_pos+1):
                            #     related_mention_text = related_mention_text + ' ' + raw_sequence[j]
                            # related_mention_text = related_mention_text.strip()
                            # if related_mention_cluster_id == focus_mention_cluster_id:
                            #     same_entity_mentions.append(related_mention_text)
                            # else:
                            #     different_entity_mentions.append(related_mention_text)

                            continue
                        else:
                            to_replace_mentions.append(related_mention)
                            to_replace_mention_start_pos_to_cluster_id[start_pos]=related_mention_cluster_id
                            
                    sorted_mention_list = sorted(to_replace_mentions, key=lambda x: x[0], reverse=True)
                    updated_raw_sequence = raw_sequence
                    for sorted_mention in sorted_mention_list:
                        start_pos = sorted_mention[0]
                        end_pos = sorted_mention[1]
                        to_replace_mention_cluster_id = to_replace_mention_start_pos_to_cluster_id[start_pos]
                        if to_replace_mention_cluster_id == focus_mention_cluster_id:
                            updated_raw_sequence = get_updated_sentence(updated_raw_sequence, start_pos, '<F> <unk> </F>', end_pos - start_pos + 1)
                        else:
                            updated_raw_sequence = get_updated_sentence(updated_raw_sequence, start_pos, '<E> <unk> </E>', end_pos - start_pos + 1)
                                  
                    updated_raw_sequence = get_updated_sentence(updated_raw_sequence, 50, '<F> @placeholder </F>', len(original_focus_mention_parts))

                    # for i in range(start_token_index_in_sentence-1, -1, -1):                        
                    #     ttpp = first_sentence_text_parts[i]
                    #     ttpp = ttpp.replace("{", "")
                    #     ttpp = ttpp.replace("}", "")
                    #     updated_raw_sequence.insert(0,ttpp)
                        
                    # for i in range(end_token_index_in_sentence, len(last_sentence_text_parts)):
                    #     ttpp = last_sentence_text_parts[i]
                    #     ttpp = ttpp.replace("{", "")
                    #     ttpp = ttpp.replace("}", "")
                    #     updated_raw_sequence.append(ttpp)
                   
                    # pre_offset = start_token_index_in_sentence
                    
                    # initial_related_mentions = raw_sequence_data._related_mentions
                    # related_mentions = []
                    # for initial_related_mention in initial_related_mentions:
                    #     final_related_mention = []
                    #     final_related_mention.append(initial_related_mention[0]+pre_offset)
                    #     final_related_mention.append(initial_related_mention[1]+pre_offset)
                    #     related_mentions.append(final_related_mention)
                    
                    #final_raw_sequence = get_updated_sequence(updated_raw_sequence, related_mentions, related_mention_cluster_ids, focus_mention_cluster_id, 50+pre_offset)
                    
                    final_raw_sequence = get_updated_sequence_2(updated_raw_sequence, related_mentions, related_mention_cluster_ids, related_converted_mention_text_list, focus_mention_cluster_id, 50)
                    updated_raw_text = ''
                    for piece in final_raw_sequence:
                        updated_raw_text = updated_raw_text + ' ' + piece
                    updated_raw_text = updated_raw_text.strip()
                    
                    if converted_focus_mention.lower() not in mention_set:
                        no_correct_string_in_se = True
                else:
                    # same_entity_mentions = []
                    # different_entity_mentions = []
                    to_replace_mentions = []
                    to_replace_mention_start_pos_to_cluster_id={}
                    for i in range(len(related_mentions)):   
                        related_mention = related_mentions[i]
                        related_mention_cluster_id = related_mention_cluster_ids[i]
                        start_pos = related_mention[0]
                        end_pos = related_mention[1]

                        if start_pos <= 50:
                            continue
                        else:
                            to_replace_mentions.append(related_mention)
                            to_replace_mention_start_pos_to_cluster_id[start_pos]=related_mention_cluster_id
                            
                    sorted_mention_list = sorted(to_replace_mentions, key=lambda x: x[0], reverse=True)
                    updated_raw_sequence = raw_sequence
                    for sorted_mention in sorted_mention_list:
                        start_pos = sorted_mention[0]
                        end_pos = sorted_mention[1]
                        to_replace_mention_cluster_id = to_replace_mention_start_pos_to_cluster_id[start_pos]
                        if to_replace_mention_cluster_id == focus_mention_cluster_id:
                            updated_raw_sequence = get_updated_sentence(updated_raw_sequence, start_pos, '<F> <unk> </F>', end_pos - start_pos + 1)
                        else:
                            updated_raw_sequence = get_updated_sentence(updated_raw_sequence, start_pos, '<E> <unk> </E>', end_pos - start_pos + 1)                                           
                    updated_raw_sequence = get_updated_sentence(updated_raw_sequence, 50, '<F> @placeholder </F>', len(original_focus_mention_parts))
                            
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

                    # related_mentions_to_update = {}
                    # related_mentions_to_update_cluster_ids = {}
                    y=0
                    # focus_mention_offset = 0
                    for sorted_key in temp_dict_sorted_keys:
                        previous_mention_index = sorted_key_to_previous_mention_index[sorted_key]
                        previous_mention = previous_mentions[previous_mention_index]
                        correct_previous_mention_1 = correct_previous_mentions_1[previous_mention_index]
                        correct_previous_mention_parts_1 = correct_previous_mention_1.split()
                        correct_previous_mention_2 = correct_previous_mentions_2[previous_mention_index]
                        correct_previous_mention_parts_2 = correct_previous_mention_2.split()
                        
                        previous_mention_cluster_id = previous_mention_cluster_ids[previous_mention_index]
                        if (previous_mention_cluster_id == focus_mention_cluster_id) or (previous_mention_cluster_id in cluster_id_to_he_clusters and focus_mention_cluster_id in cluster_id_to_he_clusters) or (previous_mention_cluster_id in cluster_id_to_she_clusters and focus_mention_cluster_id in cluster_id_to_she_clusters) or (previous_mention_cluster_id in cluster_id_to_it_clusters and focus_mention_cluster_id in cluster_id_to_it_clusters) or (previous_mention_cluster_id in cluster_id_to_they_clusters):
                            temp_related_mention = clean_related_mentions[effective_count-y-1]
                            temp_related_mention_cluster_id = clean_related_mention_cluster_ids[effective_count-y-1]
                            y=y+1
                            
                            updated_previous_mention = ''
                            if previous_mention_cluster_id == focus_mention_cluster_id:
                                updated_previous_mention = '<F> ' + previous_mention + ' </F>'
                            else:
                                updated_previous_mention = '<E> ' + previous_mention + ' </E>'
                                
                            updated_raw_sequence = get_updated_sentence(updated_raw_sequence, temp_related_mention[0], updated_previous_mention, len(correct_previous_mention_parts_2))
                            # related_mentions_to_update[temp_related_mention[0]]=previous_mention
                            # related_mentions_to_update_cluster_ids[temp_related_mention[0]]=temp_related_mention_cluster_id
                        else:
                            updated_correct_previous_mention = ''
                            if previous_mention_cluster_id == focus_mention_cluster_id:
                                updated_correct_previous_mention = '<F> ' + correct_previous_mention_1 + ' </F>'
                            else:
                                updated_correct_previous_mention = '<E> ' + correct_previous_mention_1 + ' </E>'
                                
                            updated_raw_sequence = get_updated_sentence(updated_raw_sequence, sorted_key, updated_correct_previous_mention, len(correct_previous_mention_parts_2))
                            # related_mentions_to_update[sorted_key]=correct_previous_mention_1
                            # related_mentions_to_update_cluster_ids[sorted_key]=sorted_key_to_replaced_mention_cluster_id[sorted_key]
                            
                    # for i in range(len(related_mentions)):   
                    #     related_mention = related_mentions[i]
                    #     related_mention_cluster_id = related_mention_cluster_ids[i]
                    #     start_pos = related_mention[0]
                    #     end_pos = related_mention[1]

                    #     if start_pos <= 50:
                    #         if start_pos in related_mentions_to_update:
                    #             cluster_id_to_update = related_mentions_to_update_cluster_ids[start_pos]
                    #             mention_text_to_update = related_mentions_to_update[start_pos]
                    #             if cluster_id_to_update == focus_mention_cluster_id:
                    #                 same_entity_mentions.append(mention_text_to_update)
                    #             else:
                    #                 different_entity_mentions.append(mention_text_to_update)
                    #         else:       
                    #             print('wrong implementation')
                    
                    if converted_focus_mention.lower() not in mention_set:
                        no_correct_string_in_se = True
                    
                    # for i in range(start_token_index_in_sentence-1, -1, -1):
                    #     ttpp = first_sentence_text_parts[i]
                    #     ttpp = ttpp.replace("{", "")
                    #     ttpp = ttpp.replace("}", "")
                    #     updated_raw_sequence.insert(0,ttpp)
                            
                    # for i in range(end_token_index_in_sentence, len(last_sentence_text_parts)):
                    #     ttpp = last_sentence_text_parts[i]
                    #     ttpp = ttpp.replace("{", "")
                    #     ttpp = ttpp.replace("}", "")
                    #     updated_raw_sequence.append(ttpp)
                        
                    updated_raw_text = ''
                    for piece in updated_raw_sequence:
                        updated_raw_text = updated_raw_text + ' ' + piece
                    updated_raw_text = updated_raw_text.strip()
                    
                idx = count
                guid = "{}".format(idx)

                meta1={}
                meta1['context']=updated_raw_text
                
                for i in range(len(mention_set)):
                    mention = mention_set[i]
                    mention = mention.replace("{", "")
                    mention = mention.replace("}", "")
                    mention_set[i]=mention
                    
                s1 = " <M> ".join(mention_set)
                s1 = "<M> " + s1
                meta1['entities']=s1
                # s2 = "| ".join(same_entity_mentions)
                # meta1['same_entity_mentions']=s2
                # s3 = "| ".join(different_entity_mentions)
                # meta1['different_entity_mentions']=s3
                
                converted_focus_mention = converted_focus_mention.replace("{","")
                converted_focus_mention = converted_focus_mention.replace("}", "")
                meta1['answer']=converted_focus_mention
                
                example = InputExample(guid=guid, meta=meta1, label=0)
                examples = []
                examples.append(example)
                dataloader = PromptDataLoader(dataset=examples, template=mytemplate, verbalizer=myverbalizer, tokenizer=tokenizer, 
                    tokenizer_wrapper_class=WrapperClass, max_seq_length=300, decoder_max_length=5, 
                    batch_size=1,shuffle=False, teacher_forcing=False, predict_eos_token=False, # predict_eos_token=True or False are both ok 
                    truncate_method="tail")
                
                predictions = []
                ground_truths = []
                updated_output_sentence=''
                for step, inputs in enumerate(dataloader):
                    if use_cuda:
                        inputs = inputs.cuda()
                    _, output_sentence = prompt_model.generate(inputs, **generation_arguments, verbose=False)
                    updated_output_sentence = output_sentence[0].replace("'s", " 's")
                    predictions.append(updated_output_sentence)
                    ground_truths.extend(inputs['tgt_text'])
                assert len(predictions)==len(ground_truths), (len(predictions), len(ground_truths))
                predictions = [prediction.strip() for prediction in predictions]
                ground_truths = [ground_truth.strip() for ground_truth in ground_truths]

                for p in range(len(predictions)):
                    prediction = predictions[p]
                    prediction = prediction.replace(" ,", ",")
                    prediction = prediction.replace(" .", ".")
                    prediction = prediction.replace("``", "<unk>")
                    prediction = prediction.replace("Berry", "Barry")
                    prediction = prediction.replace("berry", "barry")
                    prediction = prediction.replace("o '", "o'")
                    prediction = prediction.replace("O '", "O'")
                    prediction = prediction.replace("did n't", "didn't")
                    prediction = prediction.replace("Did n't", "Didn't")
                    predictions[p]=prediction
                for p in range(len(ground_truths)):
                    ground_truth = ground_truths[p]
                    ground_truth = ground_truth.replace(" ,", ",")
                    ground_truth = ground_truth.replace(" .", ".")
                    ground_truth = ground_truth.replace("``", "<unk>")
                    ground_truth = ground_truth.replace("Berry", "Barry")
                    ground_truth = ground_truth.replace("berry", "barry")
                    ground_truth = ground_truth.replace("o '", "o'")
                    ground_truth = ground_truth.replace("O '", "O'")
                    ground_truth = ground_truth.replace("did n't", "didn't")
                    ground_truth = ground_truth.replace("Did n't", "Didn't")
                    ground_truths[p]=ground_truth

                score1 =  crossfit_evaluate(predictions, ground_truths, metric="ACC")
                score2 = crossfit_evaluate(ground_truths, predictions, metric="ACC")
                if score1 == 0 and score2 == 0:
                    f.write(f"predictions {predictions[0]}, ground_truths {ground_truths[0]}")
                    f.write('\n')
                updated_output_sentence = updated_output_sentence.replace("{","")
                updated_output_sentence = updated_output_sentence.replace("}","")
                previous_mentions.append(updated_output_sentence)
                previous_mention_cluster_ids.append(focus_mention_cluster_id)
                correct_previous_mentions_1.append(converted_focus_mention)
                original_focus_mention = original_focus_mention.replace("{","")
                original_focus_mention = original_focus_mention.replace("}", "")
                correct_previous_mentions_2.append(original_focus_mention)
                
                if gold_setting:
                    total_size = total_size + 1
                    current_total_size = current_total_size + 1
                    if score1==1 or score2==1:
                        total_correct = total_correct + 1
                        current_total_correct = current_total_correct + 1
                    elif no_correct_string_in_se:
                        total_se_error = total_se_error + 1
                        current_total_se_error = current_total_se_error + 1
                    elif len(mention_set) >= 9:
                        total_correct = total_correct + 1
                        current_total_correct = current_total_correct + 1
                else:
                    annotated_original_mention = raw_sequence_data._annotated_original_focus_mention
                    identified_original_mention = raw_sequence_data._identified_original_focus_mention
                    if is_identified_mention == 1 and is_annotated_mention == 1:
                        total_annotated_mention = total_annotated_mention + 1
                        current_total_annotated_mention = current_total_annotated_mention + 1
                        if annotated_original_mention.lower() == identified_original_mention.lower():
                            total_annotated_mention_correctly_identified = total_annotated_mention_correctly_identified + 1
                            current_total_annotated_mention_correctly_identified = current_total_annotated_mention_correctly_identified + 1
                            if score1 == 1 or score2==1:
                                total_annotated_mention_correctly_identified_correctly_converted = total_annotated_mention_correctly_identified_correctly_converted + 1
                                current_total_annotated_mention_correctly_identified_correctly_converted = current_total_annotated_mention_correctly_identified_correctly_converted + 1
                            else:
                                if no_correct_string_in_se:
                                    total_se_error = total_se_error + 1
                                    current_total_se_error = current_total_se_error + 1
                        else:
                            total_annotated_mention_incorrectly_identified = total_annotated_mention_incorrectly_identified + 1
                            current_total_annotated_mention_incorrectly_identified = current_total_annotated_mention_incorrectly_identified + 1
                            if (annotated_original_mention.lower() != converted_focus_mention.lower()) or (identified_original_mention.lower() != updated_output_sentence.lower()):
                                total_annotated_mention_incorrectly_identified_incorrectly_converted = total_annotated_mention_incorrectly_identified_incorrectly_converted + 1
                                current_total_annotated_mention_incorrectly_identified_incorrectly_converted = current_total_annotated_mention_incorrectly_identified_incorrectly_converted + 1
                    elif is_identified_mention == 1 and is_annotated_mention == 0:
                        total_identified_mention_not_annotated = total_identified_mention_not_annotated + 1
                        current_total_identified_mention_not_annotated = current_total_identified_mention_not_annotated + 1
                        if updated_output_sentence.lower() != identified_original_mention.lower():
                            total_identified_mention_not_annotated_incorrectly_converted = total_identified_mention_not_annotated_incorrectly_converted + 1
                            current_total_identified_mention_not_annotated_incorrectly_converted = current_total_identified_mention_not_annotated_incorrectly_converted + 1
     
    annotated_mention_correctly_identified_incorrectly_converted_count = total_annotated_mention_correctly_identified - total_annotated_mention_correctly_identified_correctly_converted
    annotated_mention_incorrectly_identified_count = total_annotated_mention_not_identified_incorrectly_converted + total_annotated_mention_incorrectly_identified_incorrectly_converted
    annotated_verb_correctly_identified_incorrectly_converted_count = total_annotated_verb_correctly_identified - total_annotated_verb_correctly_identified_correctly_converted
    annotated_verb_incorrectly_identified_count = total_annotated_verb_not_identified_incorrectly_converted
    identified_mention_count = total_annotated_mention_correctly_identified+total_annotated_mention_incorrectly_identified+total_identified_mention_not_annotated_incorrectly_converted
    identified_verb_count=total_annotated_verb_correctly_identified+total_identified_verb_not_annotated_incorrectly_converted
     
    f.close()
    return total_size, total_correct, total_annotated_mention, total_annotated_verb, annotated_mention_correctly_identified_incorrectly_converted_count, annotated_mention_incorrectly_identified_count, annotated_verb_correctly_identified_incorrectly_converted_count, annotated_verb_incorrectly_identified_count, identified_mention_count, identified_verb_count
                
if not args.is_training:
    total_correct, total_size = evaluate_1(args.conll_test_data_dir, prompt_model, mytemplate, myverbalizer, tokenizer, WrapperClass)
    val_acc = 1.0 * total_correct / total_size
    print("val_acc of CoNLL test: {}".format(val_acc), flush=True)
    total_size, total_correct, total_annotated_mention, total_annotated_verb, annotated_mention_correctly_identified_incorrectly_converted_count, annotated_mention_incorrectly_identified_count, annotated_verb_correctly_identified_incorrectly_converted_count, annotated_verb_incorrectly_identified_count, identified_mention_count, identified_verb_count = evaluate_2(args.pov_gold_test_data_dir, True, prompt_model, mytemplate, myverbalizer, tokenizer, WrapperClass)
    val_acc = 1.0 * total_correct / total_size
    print("val_acc of PoV gold: {}".format(val_acc), flush=True)
    total_size, total_correct, total_annotated_mention, total_annotated_verb, annotated_mention_correctly_identified_incorrectly_converted_count, annotated_mention_incorrectly_identified_count, annotated_verb_correctly_identified_incorrectly_converted_count, annotated_verb_incorrectly_identified_count, identified_mention_count, identified_verb_count = evaluate_2(args.pov_auto_test_data_dir, False, prompt_model, mytemplate, myverbalizer, tokenizer, WrapperClass)
    correct_mention = total_annotated_mention - annotated_mention_correctly_identified_incorrectly_converted_count - annotated_mention_incorrectly_identified_count
    correct_verb = total_annotated_verb - annotated_verb_correctly_identified_incorrectly_converted_count - annotated_verb_incorrectly_identified_count 
    recall = 1.0 * (correct_mention + correct_verb) / (total_annotated_mention + total_annotated_verb)
    precision = 1.0 * (correct_mention + correct_verb) / (identified_mention_count + identified_verb_count)
    F1 = 2*recall*precision/(recall+precision)
    print("recall of PoV auto: {}".format(recall), flush=True)
    print("precision of PoV auto: {}".format(precision), flush=True)
    print("F1 of PoV auto: {}".format(F1), flush=True)
    
    exit()


loss_func = torch.nn.CrossEntropyLoss()
tot_step = args.max_steps

if args.tune_plm: # normally we freeze the model when using soft_template. However, we keep the option to tune plm
    no_decay = ['bias', 'LayerNorm.weight'] # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters1 = [
        {'params': [p for n, p in prompt_model.plm.named_parameters() if (not any(nd in n for nd in no_decay))], 'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer1 = AdamW(optimizer_grouped_parameters1, lr=3e-5)
    scheduler1 = get_linear_schedule_with_warmup(
        optimizer1, 
        num_warmup_steps=500, num_training_steps=tot_step)
else:
    optimizer1 = None
    scheduler1 = None

optimizer_grouped_parameters2 = [{'params': [p for name, p in prompt_model.template.named_parameters() if 'raw_embedding' not in name]}] # note that you have to remove the raw_embedding manually from the optimization
if args.optimizer.lower() == "adafactor":
    optimizer2 = Adafactor(optimizer_grouped_parameters2,  
                            lr=args.prompt_lr,
                            relative_step=False,
                            scale_parameter=False,
                            warmup_init=False)  # when lr is 0.3, it is the same as the configuration of https://arxiv.org/abs/2104.08691
    scheduler2 = get_constant_schedule_with_warmup(optimizer2, num_warmup_steps=args.warmup_step_prompt) # when num_warmup_steps is 0, it is the same as the configuration of https://arxiv.org/abs/2104.08691
elif args.optimizer.lower() == "adamw":
    optimizer2 = AdamW(optimizer_grouped_parameters2, lr=args.prompt_lr) # usually lr = 0.5
    scheduler2 = get_linear_schedule_with_warmup(
                    optimizer2, 
                    num_warmup_steps=args.warmup_step_prompt, num_training_steps=tot_step) # usually num_warmup_steps is 500
    
tot_loss = 0 
log_loss = 0
best_val_acc = 0
glb_step = 0
actual_step = 0
leave_training = False

acc_traces = []
tot_train_time = 0
pbar_update_freq = 10
prompt_model.train()

early_stopping = EarlyStopping(patience=50)

pbar = tqdm(total=tot_step, desc="Train")
for epoch in range(200):
    print(f"Begin epoch {epoch}")
    for step, inputs in enumerate(train_dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        tot_train_time -= time.time()       
        loss = prompt_model(inputs)
        loss.backward()
        tot_loss += loss.item()
        actual_step += 1

        if actual_step % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(prompt_model.parameters(), 1.0)
            glb_step += 1
            if glb_step % pbar_update_freq == 0:
                aveloss = (tot_loss - log_loss)/pbar_update_freq
                pbar.update(10)
                pbar.set_postfix({'loss': aveloss})
                log_loss = tot_loss

        
        if optimizer1 is not None:
            optimizer1.step()
            optimizer1.zero_grad()
        if scheduler1 is not None:
            scheduler1.step()
        if optimizer2 is not None:
            optimizer2.step()
            optimizer2.zero_grad()
        if scheduler2 is not None:
            scheduler2.step()

        tot_train_time += time.time()

        # evaluate_2(args.dev_data_dir, True, prompt_model, mytemplate, myverbalizer, tokenizer, WrapperClass)
        
        if actual_step > args.max_steps:
            leave_training = True
            break
    
    total_correct, total_size = evaluate_1(args.dev_data_dir, prompt_model, mytemplate, myverbalizer, tokenizer, WrapperClass)
    val_acc = 1.0 * total_correct / total_size
    torch.save(prompt_model.state_dict(),f"{args.project_root}/ckpts/{epoch}.ckpt")
    if val_acc >= best_val_acc:               
        best_val_acc = val_acc
        
    #acc_traces.append(val_acc)
    print("Glb_step {}, val_acc {}, average time {}".format(glb_step, val_acc, tot_train_time/actual_step ), flush=True)
    prompt_model.train()

    early_stopping(val_acc, prompt_model)
    if early_stopping.early_stop:
        print("Early stopping")
        leave_training = True
        break
           
    if leave_training:
        break  
 
# prompt_model.load_state_dict(torch.load(f"{args.project_root}/ckpts/{this_run_unicode}.ckpt"))
# prompt_model = prompt_model.cuda()
# test_acc = evaluate(prompt_model, test_dataloader, desc="Test")
# test_acc = evaluate(prompt_model, test_dataloader, desc="Test")

# thres99 = 0.99*best_val_acc
# thres98 = 0.98*best_val_acc
# thres100 = best_val_acc
# step100=step98=step99=args.max_steps
# for val_time, acc in enumerate(acc_traces):
#     if acc>=thres98:
#         step98 = min(val_time*args.eval_every_steps, step98)
#         if acc>=thres99:
#             step99 = min(val_time*args.eval_every_steps, step99)
#             if acc>=thres100:
#                 step100 = min(val_time*args.eval_every_steps, step100)

# content_write += f"BestValAcc:{best_val_acc}\tEndValAcc:{acc_traces[-1]}\tcritical_steps:{[step98,step99,step100]}\n"
# content_write += "\n"
# print(content_write)

# with open(f"{args.result_file}", "a") as fout:
#     fout.write(content_write)

#os.remove(f"ckpts/{this_run_unicode}.ckpt")
