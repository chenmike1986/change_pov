# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 15:04:09 2019

@author: chenm
"""

import numpy
import pickle
import sys

from pytorch_pretrained_bert import BertTokenizer, BertModel
import torch
import argparse

class MentionInfo(object):
    def __init__(self, text, index_in_sentence, sentence_text, sentence_num):
        self._text = text
        self._index_in_sentence = index_in_sentence
        self._sentence_text = sentence_text
        self._sentence_num = sentence_num
        
class RawSequenceData(object):
    def __init__(self, 
                 in_annotated_mc,
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
        self._in_annotated_mc = in_annotated_mc
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
    def __init__(self, gold_cluster_id_to_cluster_data, auto_cluster_id_to_cluster_data, gold_invalid_cluster_id_to_cluster_data, auto_invalid_cluster_id_to_cluster_data, raw_sequence_data_list):
        self._gold_cluster_id_to_cluster_data = gold_cluster_id_to_cluster_data
        self._auto_cluster_id_to_cluster_data = auto_cluster_id_to_cluster_data
        self._gold_invalid_cluster_id_to_cluster_data = gold_invalid_cluster_id_to_cluster_data
        self._auto_invalid_cluster_id_to_cluster_data = auto_invalid_cluster_id_to_cluster_data
        self._raw_sequence_data_list = raw_sequence_data_list
        
################################################################
# Read sequences from CoNLL2012 data set
#
class CONLL(object):
    def __init__(self, conll_path, training_data, output_file, scenario):
        self.scenario = scenario
        
        self._special_count = 0
        
        self.raw_document_data_list = []
        
        train_file = conll_path + '/' + training_data

        self.max_seql_length = 0
        self.max_seqr_length = 0
        self.max_ent_seql_length = 0
        self.max_ent_seqr_length = 0
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', max_len=450)
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.model.eval()
        
        self.train_set = self.load_data(train_file, output_file, False)
    
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
     
    def get_bert_embedding(self, text):
        token_embeddings_for_sentence = []
        tokenized_text = self.tokenizer.tokenize(text)
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens_ids_tensor = torch.tensor([tokens_ids])
        segments_ids = [0] * len(tokenized_text)
        segments_tensors = torch.tensor([segments_ids])
        input_mask = [1]*len(tokenized_text)
        input_mask = torch.tensor([input_mask], dtype=torch.long)

        all_encoder_layers, _ = self.model(tokens_ids_tensor, token_type_ids=segments_tensors, attention_mask=input_mask)
        
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
    
    def get_embedding_sequence_from_left(self, sorted_sentence_num_list, sentence_num_to_sentence_embedding, start_token_index_in_sentence, focus_mention_sentence_num, focus_mention_index_in_sentence, focus_mention_parts, from_sentence=-1):
        result_sequence = []
        
        if from_sentence == -1:
            from_sentence = sorted_sentence_num_list[0]
            
        for sentence_num in sorted_sentence_num_list:
            sentence_embedding = sentence_num_to_sentence_embedding[sentence_num]
            if sentence_num < from_sentence:
                continue
            elif sentence_num == from_sentence and sentence_num != focus_mention_sentence_num:
                for i in range(start_token_index_in_sentence, len(sentence_embedding)):
                    result_sequence.append(sentence_embedding[i])
            elif sentence_num == focus_mention_sentence_num and sentence_num != from_sentence:
                for i in range(focus_mention_index_in_sentence + len(focus_mention_parts)):
                    result_sequence.append(sentence_embedding[i])
                break
            elif sentence_num == from_sentence and sentence_num == focus_mention_sentence_num:
                for i in range(start_token_index_in_sentence, focus_mention_index_in_sentence + len(focus_mention_parts)):
                    result_sequence.append(sentence_embedding[i])
                break
            else:
                for i in range(len(sentence_embedding)):
                    result_sequence.append(sentence_embedding[i])
            
        return result_sequence
    
    def get_embedding_sequence_from_right(self, sorted_sentence_num_list, sentence_num_to_sentence_embedding, updated_sentence_num_to_sentence_text_2,related_sentence_num_to_sentence_text, end_token_index_in_sentence, focus_mention_sentence_num, focus_mention_index_in_sentence, focus_mention_parts, to_sentence=-1):
        result_sequence = []
        sequence_text = []
        
        if to_sentence == -1:
            to_sentence = sorted_sentence_num_list[len(sorted_sentence_num_list)-1]
        for i in range(len(sorted_sentence_num_list)-1, -1, -1):
            sentence_num = sorted_sentence_num_list[i]
            sentence_text=''
            if sentence_num in updated_sentence_num_to_sentence_text_2:
                sentence_text = updated_sentence_num_to_sentence_text_2[sentence_num]
            else:
                sentence_text = related_sentence_num_to_sentence_text[sentence_num]
            sentence_text_parts = sentence_text.split()   
            sentence_embedding = sentence_num_to_sentence_embedding[sentence_num]
            
            if sentence_num > to_sentence:
                continue
            elif sentence_num == to_sentence and sentence_num != focus_mention_sentence_num:
                for j in range(end_token_index_in_sentence, -1, -1):
                    result_sequence.append(sentence_embedding[j])
                    sequence_text.append(sentence_text_parts[j])
            elif sentence_num == focus_mention_sentence_num and sentence_num != to_sentence:
                for j in range(len(sentence_embedding)-1, focus_mention_index_in_sentence+len(focus_mention_parts)-1, -1):
                    result_sequence.append(sentence_embedding[j])
                    sequence_text.append(sentence_text_parts[j])
                break           
            elif sentence_num == to_sentence and sentence_num == focus_mention_sentence_num:
                for j in range(end_token_index_in_sentence, focus_mention_index_in_sentence+len(focus_mention_parts)-1, -1):
                    result_sequence.append(sentence_embedding[j])
                    sequence_text.append(sentence_text_parts[j])
                break
            else:
                for j in range(len(sentence_embedding)-1, -1, -1):
                    result_sequence.append(sentence_embedding[j])
                    sequence_text.append(sentence_text_parts[j])
                    
        return result_sequence,sequence_text
    
    def map_raw_sequence_to_sentence(self, pre_padding_count, sorted_sentence_num_list, sentence_num_to_sentence_text, start_token_index_in_sentence, end_token_index_in_sentence):
        token_index = pre_padding_count
        token_index_to_sentence_num = {}
        token_index_to_index_in_sentence = {}
        for i in range(len(sorted_sentence_num_list)):
            sentence_num = sorted_sentence_num_list[i]
            sentence_text = sentence_num_to_sentence_text[sentence_num]
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
    
    def find_max_length(self, l):
        max_len = 0
        for e in l:
            if len(e) > max_len:
                max_len = len(e)
                
        return max_len
    
    def find_max_ent_length(self, l, sep_pos_list):
        max_len = 0
        for i in range(len(l)):
            e = l[i]
            sep_pos = sep_pos_list[i]
            if len(e) + len(sep_pos) > max_len:
                max_len = len(e) + len(sep_pos)
                
        return max_len
    
    def find_postag_category(self, postags):
        postag_parts = postags.split()
        
        last_index_of_prp = -1
        last_index_of_nnp = -1
        last_index_of_nn = -1

        for i in range(len(postag_parts)):
            postag = postag_parts[i]
            if postag.find('PRP') != -1:
                last_index_of_prp = i
            elif postag.find('NNP') != -1:
                last_index_of_nnp = i
            elif postag == 'NN' or postag == 'NNS':
                last_index_of_nn = i
                
        result = -1
        if last_index_of_prp != -1:
            result = 0
        elif last_index_of_nnp != -1 and last_index_of_nnp > last_index_of_nn:
            result = 1
        elif last_index_of_nn != -1 and last_index_of_nn > last_index_of_nnp:
            result = 2
            
        return result
    
    # sequences and labels
    def load_data(self, input_file, output_file, verbose = False):
        self.raw_document_data_list = []
        
        seql_1_list = []
        seql_2_list = []
        seqr_list = []
        
        token_bit_seql_1_list = []
        token_bit_seql_2_list = []
        token_bit_seqr_list = []
                        
        same_token_bit_seql_1_list = []
        same_token_bit_seql_2_list = []
        same_token_bit_seqr_list = []
                        
        token_distance_seql_1_list = []
        token_distance_seql_2_list = []
        token_distance_seqr_list = []
                        
        ent_seql_1_list = []
        ent_seql_2_list = []
        ent_seqr_list = []
        
        ent_sep_pos_list_l = []
        ent_sep_pos_list_r = []
        
        mention_bit_seql_1_list = []
        mention_bit_seql_2_list = []
        mention_bit_seqr_list = []
        
        mention_distance_seql_1_list = []
        mention_distance_seql_2_list = []
        mention_distance_seqr_list = []
        
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
        
        self.read_file(input_file)
        
        special_count = 0
        for raw_document_data in self.raw_document_data_list:
            cluster_id_to_cluster_data = raw_document_data._gold_cluster_id_to_cluster_data
            results = self.get_similar_clusters(cluster_id_to_cluster_data)
            cluster_id_to_he_clusters = results[0]
            cluster_id_to_she_clusters = results[1]
            cluster_id_to_it_clusters = results[2]
            cluster_id_to_they_clusters = results[3]
            used_cluster_id_to_count = {}
            cluster_id_to_used_mentions = {}
            cluster_id_to_last_mention = {}
            original_sentence_num_to_embedding = {}
                    
            raw_sequence_data_list = raw_document_data._raw_sequence_data_list
            for raw_sequence_data in raw_sequence_data_list:    
                is_subject = raw_sequence_data._is_subject
                is_object = raw_sequence_data._is_object
                focus_mention_cluster_id = raw_sequence_data._focus_mention_cluster_id
                focus_mention = raw_sequence_data._original_focus_mention                
                focus_mention_parts = focus_mention.split()
                focus_mention_cluster_data = cluster_id_to_cluster_data[focus_mention_cluster_id]
                focus_mention_cluster_total_mention = focus_mention_cluster_data._total_count
                focus_mention_sentence_num = raw_sequence_data._focus_mention_sentence_num
                focus_mention_index_in_sentence = raw_sequence_data._focus_mention_index_in_sentence
                mention_set = focus_mention_cluster_data._mention_list
                
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
                pre_mention_sequence = raw_sequence_data._original_pre_mention_sequence
                pre_mention_info_list = raw_sequence_data._pre_mention_info_list
                pre_mention_cluster_id_sequence = raw_sequence_data._pre_mention_cluster_id_sequence
                pre_mention_distance_sequence = raw_sequence_data._pre_mention_distance_sequence
                post_mention_sequence= raw_sequence_data._original_post_mention_sequence
                post_mention_info_list = raw_sequence_data._post_mention_info_list
                post_mention_cluster_id_sequence = raw_sequence_data._post_mention_cluster_id_sequence
                post_mention_distance_sequence = raw_sequence_data._post_mention_distance_sequence
                
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
                
                ent_seql_focus = []
                mention_bit_seql_focus = []
                mention_distance_seql_focus = []
                pre_padding_count = 0               
                
                ent_sep_pos_l = []
                
                for i in range(len(pre_mention_sequence)):
                    ms = pre_mention_sequence[i]
                    ms_cluster_id = pre_mention_cluster_id_sequence[i]
                    ms_distance = pre_mention_distance_sequence[i]
                    token_embeddings_for_mention = []
                    if ms == '<pad>':
                        padding_embedding = [0] * 768
                        padding_embedding= numpy.asarray(padding_embedding, dtype=numpy.float32)
                        token_embeddings_for_mention.append(padding_embedding)                      
                        pre_padding_count = pre_padding_count + 1

                        mention_bit_seql_focus.append(0)                        
                        mention_distance_seql_focus.append(ms_distance)
                    else:
                        ent_sep_pos_l.append(len(ent_seql_focus))

                        if ms_cluster_id == focus_mention_cluster_id:
                            mention_bit_seql_focus.append(1)
                        else:
                            mention_bit_seql_focus.append(0)
                        
                        mention_distance_seql_focus.append(ms_distance)
                            
                        pre_mention_info = pre_mention_info_list[i-pre_padding_count]
                        pre_mention_text = pre_mention_info._text
                        pre_mention_index_in_sentence = pre_mention_info._index_in_sentence
                        pre_mention_sentence_text = pre_mention_info._sentence_text
                        pre_mention_sentence_num = pre_mention_info._sentence_num
                        
                        token_embeddings_for_sentence = None
                        if pre_mention_sentence_num not in original_sentence_num_to_embedding:
                            token_embeddings_for_sentence = self.get_bert_embedding(pre_mention_sentence_text)
                            original_sentence_num_to_embedding[pre_mention_sentence_num] = token_embeddings_for_sentence
                        else:
                            token_embeddings_for_sentence = original_sentence_num_to_embedding[pre_mention_sentence_num]
                            
                        pre_mention_text_parts = pre_mention_text.split()
                        for j in range(pre_mention_index_in_sentence, pre_mention_index_in_sentence + len(pre_mention_text_parts)):
                            token_embeddings_for_mention.append(token_embeddings_for_sentence[j])
                            
                            if ms_cluster_id == focus_mention_cluster_id:
                                mention_bit_seql_focus.append(1)
                            else:
                                mention_bit_seql_focus.append(0)
                        
                            mention_distance_seql_focus.append(ms_distance)
                        
                    ent_seql_focus.extend(token_embeddings_for_mention)
                    
                ent_focus_seq_length = len(ent_seql_focus) + len(ent_sep_pos_l)
                if self.max_ent_seql_length < ent_focus_seq_length:
                    self.max_ent_seql_length = ent_focus_seq_length
                    
                ent_seqr = []
                mention_bit_seqr = []
                mention_distance_seqr = []
                
                ent_sep_pos_r = []

                for i in range(len(post_mention_sequence)-1, -1, -1):
                    ms = post_mention_sequence[i]
                    ms_cluster_id = post_mention_cluster_id_sequence[i]
                    ms_distance = post_mention_distance_sequence[i]
                    token_embeddings_for_mention = []
                     
                    if ms == '<pad>':
                        padding_embedding = [0] * 768
                        padding_embedding= numpy.asarray(padding_embedding, dtype=numpy.float32)
                        token_embeddings_for_mention.append(padding_embedding)
                        
                        mention_bit_seqr.append(0)
                        mention_distance_seqr.append(ms_distance)
                    else:
                        ent_sep_pos_r.append(len(ent_seqr))

                        if ms_cluster_id == focus_mention_cluster_id:
                            mention_bit_seqr.append(1)
                        else:
                            mention_bit_seqr.append(0)
                        mention_distance_seqr.append(ms_distance) 
                        
                        post_mention_info = post_mention_info_list[i]
                        post_mention_text = post_mention_info._text
                        post_mention_text_parts = post_mention_text.split()
                        post_mention_index_in_sentence = post_mention_info._index_in_sentence
                        post_mention_sentence_text = post_mention_info._sentence_text
                        post_mention_sentence_num = post_mention_info._sentence_num
                        
                        if ms_cluster_id == focus_mention_cluster_id:
                            updated_post_mention_sentence_text = self.get_updated_sentence(post_mention_sentence_text, post_mention_index_in_sentence, '[MASK]', len(post_mention_text_parts))
                            token_embeddings_for_sentence = self.get_bert_embedding(updated_post_mention_sentence_text)
                            token_embeddings_for_mention.append(token_embeddings_for_sentence[post_mention_index_in_sentence])
                            
                            mention_bit_seqr.append(1)
                            mention_distance_seqr.append(ms_distance)
                            token_embeddings_for_sentence = []
                        elif self.scenario == 2 and ((ms_cluster_id in cluster_id_to_he_clusters and focus_mention_cluster_id in cluster_id_to_he_clusters) or (ms_cluster_id in cluster_id_to_she_clusters and focus_mention_cluster_id in cluster_id_to_she_clusters) or (ms_cluster_id in cluster_id_to_it_clusters and focus_mention_cluster_id in cluster_id_to_it_clusters) or (ms_cluster_id in cluster_id_to_they_clusters)):
                            updated_post_mention_sentence_text = self.get_updated_sentence(post_mention_sentence_text, post_mention_index_in_sentence, '[MASK]', len(post_mention_text_parts))
                            token_embeddings_for_sentence = self.get_bert_embedding(updated_post_mention_sentence_text)
                            token_embeddings_for_mention.append(token_embeddings_for_sentence[post_mention_index_in_sentence])
                                    
                            mention_bit_seqr.append(0)
                            mention_distance_seqr.append(ms_distance)
                            token_embeddings_for_sentence = []
                        else:
                            token_embeddings_for_sentence = None
                            if post_mention_sentence_num not in original_sentence_num_to_embedding:
                                token_embeddings_for_sentence = self.get_bert_embedding(post_mention_sentence_text)
                                original_sentence_num_to_embedding[post_mention_sentence_num] = token_embeddings_for_sentence
                            else:
                                token_embeddings_for_sentence = original_sentence_num_to_embedding[post_mention_sentence_num]
                                
                            post_mention_text_parts = post_mention_text.split()
                            for j in range(post_mention_index_in_sentence, post_mention_index_in_sentence + len(post_mention_text_parts)):
                                token_embeddings_for_mention.append(token_embeddings_for_sentence[j])
                                    
                                mention_bit_seqr.append(0)
                                mention_distance_seqr.append(ms_distance)                     
                                    
                    ent_seqr.extend(token_embeddings_for_mention)
                        
                ent_seqr_length = len(ent_seqr) + len(ent_sep_pos_r)
                if self.max_ent_seqr_length < ent_seqr_length:
                    self.max_ent_seqr_length = ent_seqr_length
                 
                for i in range(len(mention_set)):
                    mention = mention_set[i]
                    mention_parts = mention.split()
                    
                    if mention.lower() != focus_mention.lower():
                        ent_seql_other = []
                        mention_bit_seql_other = []
                        mention_distance_seql_other = []
                        
                        for j in range(len(ent_seql_focus) - len(focus_mention_parts)):
                            ent_seql_other.append(ent_seql_focus[j])
                            
                        for j in range(len(mention_bit_seql_focus) - len(focus_mention_parts)):
                            mention_bit_seql_other.append(mention_bit_seql_focus[j])
                            mention_distance_seql_other.append(mention_distance_seql_focus[j])
                        
                        focus_mention_sentence = related_sentence_num_to_sentence_text[focus_mention_sentence_num]
                        updated_sentence_text = self.get_updated_sentence(focus_mention_sentence, focus_mention_index_in_sentence, mention, len(focus_mention_parts))
                            
                        token_embeddings_for_sentence = self.get_bert_embedding(updated_sentence_text)
                        for j in range(focus_mention_index_in_sentence, focus_mention_index_in_sentence + len(mention_parts)):
                            ent_seql_other.append(token_embeddings_for_sentence[j])
                                                        
                            mention_bit_seql_other.append(1)
                            mention_distance_seql_other.append(pre_mention_distance_sequence[len(pre_mention_distance_sequence)-1])
                        token_embeddings_for_sentence = []
                            
                        ent_other_seq_length = len(ent_seql_other) + len(ent_sep_pos_l)
                        if self.max_ent_seql_length < ent_other_seq_length:
                            self.max_ent_seql_length = ent_other_seq_length
                            
                        ent_seql_focus_copy = ent_seql_focus.copy()
                        mention_bit_seql_focus_copy = mention_bit_seql_focus.copy()
                        mention_distance_seql_focus_copy = mention_distance_seql_focus.copy()
                        ent_seqr_copy = ent_seqr.copy()
                        mention_bit_seqr_copy = mention_bit_seqr.copy()
                        mention_distance_seqr_copy = mention_distance_seqr.copy()
                        ent_sep_pos_l_copy = ent_sep_pos_l.copy()
                        ent_sep_pos_r_copy = ent_sep_pos_r.copy()
                        ent_seql_1_list.append(ent_seql_focus_copy)
                        ent_seql_2_list.append(ent_seql_other)
                        ent_seqr_list.append(ent_seqr_copy)
                        mention_bit_seql_1_list.append(mention_bit_seql_focus_copy)
                        mention_bit_seql_2_list.append(mention_bit_seql_other)
                        mention_bit_seqr_list.append(mention_bit_seqr_copy)
                        mention_distance_seql_1_list.append(mention_distance_seql_focus_copy)
                        mention_distance_seql_2_list.append(mention_distance_seql_other)
                        mention_distance_seqr_list.append(mention_distance_seqr_copy)
                        ent_sep_pos_list_l.append(ent_sep_pos_l_copy)
                        ent_sep_pos_list_r.append(ent_sep_pos_r_copy)
                                        
                seql_focus = []
                focus_seq_length = 50 + len(focus_mention_parts)           
                pre_padding_count = 0
                for i in range(len(raw_sequence)):
                    word = raw_sequence[i]
                    if word == "<pad>":
                        padding_embedding = [0] * 768
                        padding_embedding= numpy.asarray(padding_embedding, dtype=numpy.float32)
                        seql_focus.append(padding_embedding)
                        pre_padding_count = pre_padding_count + 1
                    else:
                        break
                    
                sorted_sentence_num_list = sorted(related_sentence_num_to_sentence_text.keys())
                token_index_to_sentence_num, token_index_to_index_in_sentence = self.map_raw_sequence_to_sentence(pre_padding_count, sorted_sentence_num_list, related_sentence_num_to_sentence_text, start_token_index_in_sentence, end_token_index_in_sentence)
                updated_sentence_num_to_sentence_text_1 = self.get_updated_sentence_num_to_sentence_text(related_sentence_num_to_sentence_text, token_index_to_sentence_num, token_index_to_index_in_sentence, updated_related_mentions)
                updated_sentence_num_to_sentence_text_2 = self.get_updated_sentence_num_to_sentence_text(related_sentence_num_to_sentence_text, token_index_to_sentence_num, token_index_to_index_in_sentence, clean_related_mentions)
                last_sentence_num = sorted_sentence_num_list[len(sorted_sentence_num_list)-1]
                end_token_index_offset_1 = self.get_end_token_offset(token_index_to_sentence_num, last_sentence_num, updated_related_mentions)
                end_token_index_offset_2 = self.get_end_token_offset(token_index_to_sentence_num, last_sentence_num, clean_related_mentions)
                
                sentence_num_to_sentence_embedding_1 = {}
                for sentence_num in related_sentence_num_to_sentence_text:
                    if sentence_num in updated_sentence_num_to_sentence_text_1:
                        sentence_text = updated_sentence_num_to_sentence_text_1[sentence_num]
                        token_embeddings_for_sentence = self.get_bert_embedding(sentence_text)
                        sentence_num_to_sentence_embedding_1[sentence_num] = token_embeddings_for_sentence
                    else:
                        if sentence_num in original_sentence_num_to_embedding:
                            token_embeddings_for_sentence = original_sentence_num_to_embedding[sentence_num]
                            sentence_num_to_sentence_embedding_1[sentence_num] = token_embeddings_for_sentence
                        else:
                            sentence_text = related_sentence_num_to_sentence_text[sentence_num]
                            token_embeddings_for_sentence = self.get_bert_embedding(sentence_text)
                            sentence_num_to_sentence_embedding_1[sentence_num] = token_embeddings_for_sentence
                            original_sentence_num_to_embedding[sentence_num] = token_embeddings_for_sentence
                 
                sentence_num_to_sentence_embedding_2 = {}
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
                
                temp_seql_focus = []
                if self.scenario == 1:
                    temp_seql_focus = self.get_embedding_sequence_from_left(sorted_sentence_num_list, sentence_num_to_sentence_embedding_1, start_token_index_in_sentence, focus_mention_sentence_num, focus_mention_index_in_sentence, focus_mention_parts)
                else:
                    temp_seql_focus = self.get_embedding_sequence_from_left(sorted_sentence_num_list, sentence_num_to_sentence_embedding_2, start_token_index_in_sentence, focus_mention_sentence_num, focus_mention_index_in_sentence, focus_mention_parts)

                seql_focus.extend(temp_seql_focus)
                        
                if self.max_seql_length < focus_seq_length:
                    self.max_seql_length = focus_seq_length
                 
                token_bit_seql_focus = [0] * focus_seq_length
                same_token_bit_seql_focus = [0] * focus_seq_length
                token_distance_seql_focus = [0] * focus_seq_length
                            
                previous_m_pos = 0
                for i in range(0,50):     
                    token_distance_seql_focus[i]=i-previous_m_pos
                    for j in range(0, len(related_mentions)):
                        related_mention = related_mentions[j]
                        related_mention_cluster_id = related_mention_cluster_ids[j]
                        rms = related_mention[0]
                        rme = related_mention[1]
                        if i >= rms and i <= rme:
                            token_bit_seql_focus[i] = 1
                            previous_m_pos = rms
                            if related_mention_cluster_id == focus_mention_cluster_id:
                                same_token_bit_seql_focus[i]=1
                            break
                                                                    
                for i in range(50,focus_seq_length):
                    token_bit_seql_focus[i]=1
                    same_token_bit_seql_focus[i]=1
                    token_distance_seql_focus[i]=i-previous_m_pos                                
                previous_m_pos = 50
                            
                seqr = []     
                seqr_text = []
                temp_seqr = []
                temp_seqr_text=[]
                for i in range(len(raw_sequence)-1, len(raw_sequence)-51, -1):
                    word = raw_sequence[i]
                    if word == "<pad>":
                        padding_embedding = [0] * 768
                        padding_embedding= numpy.asarray(padding_embedding, dtype=numpy.float32)
                        seqr.append(padding_embedding)
                        seqr_text.append("<pad>")
                    else:    
                        break
                 
                if self.scenario == 1:                           
                    temp_seqr = self.get_embedding_sequence_from_right(sorted_sentence_num_list, sentence_num_to_sentence_embedding_1, end_token_index_in_sentence+end_token_index_offset_1, focus_mention_sentence_num, focus_mention_index_in_sentence, focus_mention_parts)        
                elif self.scenario == 2:
                    temp_seqr,temp_seqr_text = self.get_embedding_sequence_from_right(sorted_sentence_num_list, sentence_num_to_sentence_embedding_2,updated_sentence_num_to_sentence_text_2,related_sentence_num_to_sentence_text, end_token_index_in_sentence+end_token_index_offset_2, focus_mention_sentence_num, focus_mention_index_in_sentence, focus_mention_parts)                                   
                else:
                    print("The scenario mode does not exist.")
                seqr.extend(temp_seqr)
                seqr_text.extend(temp_seqr_text)
                
                seqr_length = len(seqr)
                if self.max_seqr_length < seqr_length:
                    self.max_seqr_length = seqr_length
                
                token_bit_seqr = [0] * seqr_length
                same_token_bit_seqr = [0] * seqr_length
                token_distance_seqr = [0] * seqr_length
                                                   
                TEMP_COUNT = 0
                for i in range(seqr_length-1, -1, -1):    
                    tttt = seqr_text[i]
                    related_mention_cluster_id = -10
                    if TEMP_COUNT == len(related_mention_cluster_ids) and len(related_mention_cluster_ids) > 0:
                        related_mention_cluster_id = related_mention_cluster_ids[0]
                    elif len(related_mention_cluster_ids) > 0:
                        related_mention_cluster_id = related_mention_cluster_ids[len(related_mention_cluster_ids)-1-TEMP_COUNT]
                    
                    found_previous_mention = False
                    for j in range(i-1, -1, -1):
                        pppp = seqr_text[j]
                        if pppp == '[MASK]':
                            previous_m_pos = j
                            found_previous_mention = True
                    if tttt == '[MASK]':
                        TEMP_COUNT = TEMP_COUNT + 1
                        token_bit_seqr[i] = 1
                        if related_mention_cluster_id == focus_mention_cluster_id:
                            same_token_bit_seqr[i]=1
                                 
                    if found_previous_mention:
                        token_distance_seqr[i]=i-previous_m_pos
                    else:
                        token_distance_seqr[i]=i
                                    
                sentence_num_to_sentence_embedding_2 = {}

                used_bit_focus, same_bit_focus = self.get_used_bit_and_same_bit(focus_mention, focus_mention_cluster_id, cluster_id_to_used_mentions, cluster_id_to_last_mention)
                
                for i in range(len(mention_set)):
                    mention = mention_set[i]
                    
                    if mention.lower() != focus_mention.lower():                       
                        seql_other = []

                        mention_parts = mention.split()                    
                        other_seq_length = 50 + len(mention_parts)
                            
                        focus_mention_sentence = ""
                        if self.scenario == 1:
                            if focus_mention_sentence_num in updated_sentence_num_to_sentence_text_1:
                                focus_mention_sentence = updated_sentence_num_to_sentence_text_1[focus_mention_sentence_num]
                            else:
                                focus_mention_sentence = related_sentence_num_to_sentence_text[focus_mention_sentence_num]
                        else:
                            if focus_mention_sentence_num in updated_sentence_num_to_sentence_text_2:
                                focus_mention_sentence = updated_sentence_num_to_sentence_text_2[focus_mention_sentence_num]
                            else:
                                focus_mention_sentence = related_sentence_num_to_sentence_text[focus_mention_sentence_num]
                                
                        updated_sentence_text = self.get_updated_sentence(focus_mention_sentence, focus_mention_index_in_sentence, mention, len(focus_mention_parts))
                        token_embeddings_for_sentence = self.get_bert_embedding(updated_sentence_text)
                        sentence_num_to_sentence_embedding_1[focus_mention_sentence_num] = token_embeddings_for_sentence
                        
                        for j in range(len(raw_sequence)):
                            word = raw_sequence[j]
                            if word == "<pad>":
                                padding_embedding = [0] * 768
                                padding_embedding= numpy.asarray(padding_embedding, dtype=numpy.float32)
                                seql_other.append(padding_embedding)
                            else:
                                break
            
                        temp_seql_other = self.get_embedding_sequence_from_left(sorted_sentence_num_list, sentence_num_to_sentence_embedding_1, start_token_index_in_sentence, focus_mention_sentence_num, focus_mention_index_in_sentence, mention_parts)                        
                        seql_other.extend(temp_seql_other)
                        
                        if self.max_seql_length < other_seq_length:
                            self.max_seql_length = other_seq_length                                    
                        
                        token_bit_seql_other = [0] * other_seq_length
                        same_token_bit_seql_other = [0] * other_seq_length
                        token_distance_seql_other = [0] * other_seq_length
                               
                        previous_m_pos = 0
                        for i in range(0,50):              
                            token_distance_seql_other[i]=i-previous_m_pos
                            for j in range(0, len(related_mentions)):
                                related_mention = related_mentions[j]
                                related_mention_cluster_id = related_mention_cluster_ids[j]
                                rms = related_mention[0]
                                rme = related_mention[1]
                                if i >= rms and i <= rme:
                                    token_bit_seql_other[i] = 1
                                    previous_m_pos = rms
                                    if related_mention_cluster_id == focus_mention_cluster_id:
                                        same_token_bit_seql_other[i]=1
                                    break
                                
                        for i in range(50,other_seq_length):
                            token_bit_seql_other[i]=1
                            same_token_bit_seql_other[i]=1
                            token_distance_seql_other[i]=i-previous_m_pos
                        previous_m_pos = 50
                                    
                        seql_focus_copy = seql_focus.copy()
                        seqr_copy = seqr.copy()
                        seql_1_list.append(seql_focus_copy)
                        seql_2_list.append(seql_other)
                        seqr_list.append(seqr_copy)                        
                          
                        token_bit_seql_focus_copy = token_bit_seql_focus.copy()
                        token_bit_seqr_copy = token_bit_seqr.copy()
                        same_token_bit_seql_focus_copy = same_token_bit_seql_focus.copy()
                        same_token_bit_seqr_copy = same_token_bit_seqr.copy()
                        token_distance_seql_focus_copy = token_distance_seql_focus.copy()
                        token_distance_seqr_copy = token_distance_seqr.copy()
                        token_bit_seql_1_list.append(token_bit_seql_focus_copy)
                        token_bit_seql_2_list.append(token_bit_seql_other)
                        token_bit_seqr_list.append(token_bit_seqr_copy)
                        same_token_bit_seql_1_list.append(same_token_bit_seql_focus_copy)
                        same_token_bit_seql_2_list.append(same_token_bit_seql_other)
                        same_token_bit_seqr_list.append(same_token_bit_seqr_copy)
                        token_distance_seql_1_list.append(token_distance_seql_focus_copy)
                        token_distance_seql_2_list.append(token_distance_seql_other)
                        token_distance_seqr_list.append(token_distance_seqr_copy)
                                    
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
                        
                        
                if len(seql_1_list) >= 512:
                    special_count = special_count + 1
                    temp_data_set = [seql_1_list[:512], seql_2_list[:512], seqr_list[:512], ent_seql_1_list[:512], ent_seql_2_list[:512], ent_sep_pos_list_l[:512], ent_seqr_list[:512], ent_sep_pos_list_r[:512], mention_bit_seql_1_list[:512], mention_bit_seql_2_list[:512], mention_bit_seqr_list[:512], mention_distance_seql_1_list[:512], mention_distance_seql_2_list[:512], mention_distance_seqr_list[:512], first_mention_bit_seql_list[:512], second_mention_bit_seql_list[:512], focus_mention_length_list[:512], other_mention_length_list[:512], coreference_chain_length_list[:512], used_bit_seql_1_list[:512], same_bit_seql_1_list[:512], used_bit_seql_2_list[:512], same_bit_seql_2_list[:512], is_subject_list[:512], is_object_list[:512], token_bit_seql_1_list[:512], token_bit_seql_2_list[:512], token_bit_seqr_list[:512], same_token_bit_seql_1_list[:512], same_token_bit_seql_2_list[:512], same_token_bit_seqr_list[:512], token_distance_seql_1_list[:512], token_distance_seql_2_list[:512], token_distance_seqr_list[:512], label_list[:512]]
                    temp_data_set = self.create_padding(temp_data_set)
                    temp_file = output_file+ "_" + str(special_count) + ".pkl"
                    pickle.dump(temp_data_set, open(temp_file, 'wb'))
                    temp_data_set = []
                            
                    seql_1_list = seql_1_list[512:]
                    seql_2_list = seql_2_list[512:]
                    seqr_list = seqr_list[512:]
                    token_bit_seql_1_list = token_bit_seql_1_list[512:]
                    token_bit_seql_2_list = token_bit_seql_2_list[512:]
                    token_bit_seqr_list = token_bit_seqr_list[512:]
                    same_token_bit_seql_1_list = same_token_bit_seql_1_list[512:]
                    same_token_bit_seql_2_list = same_token_bit_seql_2_list[512:]
                    same_token_bit_seqr_list = same_token_bit_seqr_list[512:]
                    token_distance_seql_1_list = token_distance_seql_1_list[512:]
                    token_distance_seql_2_list = token_distance_seql_2_list[512:]
                    token_distance_seqr_list = token_distance_seqr_list[512:]
                    ent_seql_1_list = ent_seql_1_list[512:]
                    ent_seql_2_list = ent_seql_2_list[512:]
                    ent_sep_pos_list_l = ent_sep_pos_list_l[512:]
                    ent_seqr_list = ent_seqr_list[512:]
                    ent_sep_pos_list_r = ent_sep_pos_list_r[512:]
                    mention_bit_seql_1_list = mention_bit_seql_1_list[512:]
                    mention_bit_seql_2_list = mention_bit_seql_2_list[512:]
                    mention_bit_seqr_list = mention_bit_seqr_list[512:]
                    mention_distance_seql_1_list = mention_distance_seql_1_list[512:]
                    mention_distance_seql_2_list = mention_distance_seql_2_list[512:]
                    mention_distance_seqr_list = mention_distance_seqr_list[512:]
                    first_mention_bit_seql_list = first_mention_bit_seql_list[512:]
                    second_mention_bit_seql_list = second_mention_bit_seql_list[512:]
                    focus_mention_length_list  = focus_mention_length_list[512:]
                    other_mention_length_list = other_mention_length_list[512:]
                    coreference_chain_length_list = coreference_chain_length_list[512:]
                    used_bit_seql_1_list = used_bit_seql_1_list[512:]
                    same_bit_seql_1_list = same_bit_seql_1_list[512:]
                    used_bit_seql_2_list = used_bit_seql_2_list[512:]
                    same_bit_seql_2_list = same_bit_seql_2_list[512:]
                    is_subject_list = is_subject_list[512:]
                    is_object_list = is_object_list[512:]
                    label_list = label_list[512:]
                            
                    max_seql_length_1 = self.find_max_length(seql_1_list)
                    max_seql_length_2 = self.find_max_length(seql_2_list)
                    if max_seql_length_1 >= max_seql_length_2:
                        self.max_seql_length = max_seql_length_1
                    else:
                        self.max_seql_length = max_seql_length_2
                                
                    self.max_seqr_length = self.find_max_length(seqr_list)
                            
                    max_ent_seql_length_1 = self.find_max_ent_length(ent_seql_1_list, ent_sep_pos_list_l)
                    max_ent_seql_length_2 = self.find_max_ent_length(ent_seql_2_list, ent_sep_pos_list_l)
                    if max_ent_seql_length_1 >= max_ent_seql_length_2:
                        self.max_ent_seql_length = max_ent_seql_length_1
                    else:
                        self.max_ent_seql_length = max_ent_seql_length_2
                        
                    self.max_ent_seqr_length = self.find_max_ent_length(ent_seqr_list, ent_sep_pos_list_r)
                
                sentence_num_to_sentence_embedding_1 = {}
                
                if focus_mention_cluster_id not in cluster_id_to_used_mentions:
                    temp = []
                    temp.append(focus_mention.lower())
                    cluster_id_to_used_mentions[focus_mention_cluster_id] = temp
                else:
                    temp = cluster_id_to_used_mentions[focus_mention_cluster_id]
                    if focus_mention.lower() not in temp:
                        temp.append(focus_mention.lower())
                        cluster_id_to_used_mentions[focus_mention_cluster_id] = temp
                if focus_mention_cluster_id not in cluster_id_to_last_mention:
                    cluster_id_to_last_mention[focus_mention_cluster_id] = focus_mention.lower()
                else:
                    temp = cluster_id_to_last_mention[focus_mention_cluster_id]
                    if temp != focus_mention.lower():
                        cluster_id_to_last_mention[focus_mention_cluster_id] = focus_mention.lower()
        
        self._special_count = special_count                
        return [seql_1_list, seql_2_list, seqr_list, ent_seql_1_list, ent_seql_2_list, ent_sep_pos_list_l, ent_seqr_list, ent_sep_pos_list_r, mention_bit_seql_1_list, mention_bit_seql_2_list, mention_bit_seqr_list, mention_distance_seql_1_list, mention_distance_seql_2_list, mention_distance_seqr_list, first_mention_bit_seql_list, second_mention_bit_seql_list, focus_mention_length_list, other_mention_length_list, coreference_chain_length_list, used_bit_seql_1_list, same_bit_seql_1_list, used_bit_seql_2_list, same_bit_seql_2_list, is_subject_list, is_object_list, token_bit_seql_1_list, token_bit_seql_2_list, token_bit_seqr_list, same_token_bit_seql_1_list, same_token_bit_seql_2_list, same_token_bit_seqr_list, token_distance_seql_1_list, token_distance_seql_2_list, token_distance_seqr_list,label_list]

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
        in_annotated_mc = -1
        original_focus_mention = ""
        converted_focus_mention = ""
        cluster_id = -1
        sentence_num = -1
        index_in_sentence = -1
        is_subject = -1
        is_object = -1
        
        line = line.strip()
        line_parts = line.split("\t")
        in_annotated_mc_part = ''
        focus_mention_part = ''
        converted_mention_part = ''
        cluster_part = ''
        sentence_num_part = ''
        index_in_sentence_part = ''
        is_subject_part = ''
        is_object_part = ''
        
        has_converted_part = False
        has_in_annotated_mc = False
        if line.startswith("<in_annotated_mc>:"):
            in_annotated_mc_part = line_parts[0].strip() 
            focus_mention_part = line_parts[1].strip()
            temp_parts = in_annotated_mc_part.split()
            in_annotated_mc = int(temp_parts[1])   
            has_in_annotated_mc = True
        
        if not has_in_annotated_mc:
            focus_mention_part = line_parts[0].strip()
            
        if focus_mention_part.startswith("<focus_mention>:"):
            temp_parts = focus_mention_part.split()
            for i in range(1,len(temp_parts)):
                ttt = temp_parts[i].lower()
                original_focus_mention = original_focus_mention + " " + ttt
            original_focus_mention = original_focus_mention.strip()
            converted_focus_mention = original_focus_mention
        elif focus_mention_part.startswith("<original_focus_mention>:"):
            temp_parts = focus_mention_part.split()
            for i in range(1,len(temp_parts)):
                ttt = temp_parts[i].lower()
                original_focus_mention = original_focus_mention + " " + ttt
            original_focus_mention = original_focus_mention.strip()
            has_converted_part = True
        else:
            print("There is an error in the dataset 15.")
          
        if has_in_annotated_mc and has_converted_part:
            converted_mention_part = line_parts[2].strip()
            cluster_part = line_parts[3].strip()
            sentence_num_part = line_parts[4].strip()
            index_in_sentence_part = line_parts[5].strip()
            is_subject_part = line_parts[6].strip()
            is_object_part = line_parts[7].strip()
        elif not has_in_annotated_mc and has_converted_part:
            converted_mention_part = line_parts[1].strip()
            cluster_part = line_parts[2].strip()
            sentence_num_part = line_parts[3].strip()
            index_in_sentence_part = line_parts[4].strip()
            is_subject_part = line_parts[5].strip()
            is_object_part = line_parts[6].strip()
        elif not has_in_annotated_mc and not has_converted_part:
            cluster_part = line_parts[1].strip()
            sentence_num_part = line_parts[2].strip()
            index_in_sentence_part = line_parts[3].strip()
            is_subject_part = line_parts[4].strip()
            is_object_part = line_parts[5].strip()
            
        if has_converted_part:
            if converted_mention_part.startswith("<converted_focus_mention>:"):
                temp_parts = converted_mention_part.split()
                for i in range(1,len(temp_parts)):
                    ttt = temp_parts[i].lower()
                    converted_focus_mention = converted_focus_mention + " " + ttt
                converted_focus_mention = converted_focus_mention.strip()
            else:
                print("There is an error in the dataset 16.")
            
        if cluster_part.startswith("<cluster_id>:"):
            temp_parts = cluster_part.split()
            cluster_id = int(temp_parts[1])
        else:
            print("There is an error in the dataset 17.")
           
        if sentence_num_part.startswith("<sentence_num>:"):
            temp_parts = sentence_num_part.split()
            sentence_num = int(temp_parts[1])
        else:
            print("There is an error in the dataset 18.")
            
        if index_in_sentence_part.startswith("<index_in_sentence>:"):
            temp_parts = index_in_sentence_part.split()
            index_in_sentence = int(temp_parts[1])
        else:
            print("There is an error in the dataset 19.")
        
        if is_subject_part.startswith("<is_subject>:"):
            temp_parts = is_subject_part.split()
            is_subject = int(temp_parts[1])
        else:
            print("There is an error in the dataset I.")
            
        if is_object_part.startswith("<is_object>:"):
            temp_parts = is_object_part.split()
            is_object = int(temp_parts[1])
        else:
            print("There is an error in the dataset II.")
        
        return [in_annotated_mc, original_focus_mention, converted_focus_mention, cluster_id, sentence_num, index_in_sentence, is_subject, is_object]
    
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
        f = open(file_path, 'r', encoding='utf-8')
        lines = f.readlines()

        end_of_cluster = False
        encounter_valid_invalid_cluster = False
        
        in_annotated_mc = -1
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
            if line.startswith("<cluster_id>:") or line.startswith("<gold_cluster_id>:") or line.startswith("<auto_cluster_id>:"):
                # Start of another document
                if end_of_cluster:
                    raw_document_data = RawDocumentData(gold_cluster_id_to_cluster_data, auto_cluster_id_to_cluster_data, gold_invalid_cluster_id_to_cluster_data, auto_invalid_cluster_id_to_cluster_data, raw_sequence_data_list)
                    self.raw_document_data_list.append(raw_document_data)
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
            elif line.startswith("<focus_mention>:") or line.startswith("<original_focus_mention>:") or line.startswith("<in_annotated_mc>:"):
                result = self.get_focus_mention_from_line(line)
                in_annotated_mc = result[0]
                original_focus_mention = result[1]
                converted_focus_mention = result[2]
                focus_mention_cluster_id = result[3]
                focus_mention_sentence_num = result[4]
                focus_mention_index_in_sentence = result[5]
                is_subject = result[6]
                is_object = result[7]
                if not end_of_cluster:
                    end_of_cluster = True
                encounter_valid_invalid_cluster = False
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
                raw_sequence_data = RawSequenceData(in_annotated_mc, original_focus_mention, converted_focus_mention, focus_mention_cluster_id, focus_mention_sentence_num, focus_mention_index_in_sentence, is_subject, is_object, related_mentions, related_mention_cluster_ids, related_original_mention_text_list, related_converted_mention_text_list, raw_sequence, postag_sequence, related_sentence_num_to_sentence_text, start_token_index_in_sentence, end_token_index_in_sentence, original_pre_mention_sequence, converted_pre_mention_sequence, pre_mention_cluster_id_sequence, pre_mention_distance_sequence, pre_mention_info_list, original_post_mention_sequence, converted_post_mention_sequence, post_mention_cluster_id_sequence, post_mention_distance_sequence, post_mention_info_list)
                raw_sequence_data_list.append(raw_sequence_data)
                in_annotated_mc = -1
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
            else:
                print("there is an error.")
                print(line)
            previous_line = line
            
        if len(gold_cluster_id_to_cluster_data) > 0 and len(raw_sequence_data_list) > 0:
            raw_document_data = RawDocumentData(gold_cluster_id_to_cluster_data, auto_cluster_id_to_cluster_data, gold_invalid_cluster_id_to_cluster_data, auto_invalid_cluster_id_to_cluster_data, raw_sequence_data_list)
            self.raw_document_data_list.append(raw_document_data)
                
        f.close()
    
    def is_special_index(self, related_mentions, index):
        result = -1
        related_mention_index = -1
        
        for i in range(len(related_mentions)):
            related_mention = related_mentions[i]
            if index == related_mention[1]:
                result = 1
                related_mention_index = i
                break
            elif index > related_mention[0] and index < related_mention[1]:
                result = 0
                related_mention_index = i
                break
            elif index == related_mention[0]:
                result = 2
                related_mention_index = i
                break
            
        return [result, related_mention_index]   
    
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
    
    def create_padding(self, data_set):
        seql_1v = data_set[0]
        seql_2v = data_set[1]
        seqr_v, seqr_l = self.list_to_array(data_set[2], self.max_seqr_length, 2)
        ent_seql_1v = data_set[3]
        ent_seql_2v = data_set[4]
        ent_sep_pos_l = data_set[5]
        ent_seqr_v = data_set[6]
        ent_sep_pos_r = data_set[7]
        mention_bit_seql_1v, mention_bit_seql_1l = self.list_to_array(data_set[8], self.max_ent_seql_length, 1)
        mention_bit_seql_2v, mention_bit_seql_2l = self.list_to_array(data_set[9], self.max_ent_seql_length, 1)
        mention_bit_seqr_v, mention_bit_seqr_l = self.list_to_array(data_set[10], self.max_ent_seqr_length, 1)
        mention_distance_seql_1v, mention_distance_seql_1l = self.list_to_array(data_set[11], self.max_ent_seql_length, 1)
        mention_distance_seql_2v, mention_distance_seql_2l = self.list_to_array(data_set[12], self.max_ent_seql_length, 1)
        mention_distance_seqr_v, mention_distance_seqr_l = self.list_to_array(data_set[13], self.max_ent_seqr_length, 1)
        token_bit_seql_1v, _= self.list_to_array(data_set[25], self.max_seql_length, 1)
        token_bit_seql_2v, _ = self.list_to_array(data_set[26], self.max_seql_length, 1)
        token_bit_seqr_v, _ = self.list_to_array(data_set[27], self.max_seqr_length, 1)
        same_token_bit_seql_1v, _= self.list_to_array(data_set[28], self.max_seql_length, 1)
        same_token_bit_seql_2v, _ = self.list_to_array(data_set[29], self.max_seql_length, 1)
        same_token_bit_seqr_v, _ = self.list_to_array(data_set[30], self.max_seqr_length, 1)
        token_distance_seql_1v, _= self.list_to_array(data_set[31], self.max_seql_length, 1)
        token_distance_seql_2v, _ = self.list_to_array(data_set[32], self.max_seql_length, 1)
        token_distance_seqr_v, _ = self.list_to_array(data_set[33], self.max_seqr_length, 1)
        data = [seql_1v, seql_2v, seqr_v, seqr_l, ent_seql_1v, ent_seql_2v, ent_sep_pos_l, ent_seqr_v, ent_sep_pos_r, mention_bit_seql_1v, mention_bit_seql_1l, mention_bit_seql_2v, mention_bit_seql_2l, mention_bit_seqr_v, mention_bit_seqr_l, mention_distance_seql_1v, mention_distance_seql_1l, mention_distance_seql_2v, mention_distance_seql_2l, mention_distance_seqr_v, mention_distance_seqr_l, numpy.asarray(data_set[14], dtype=numpy.float32), numpy.asarray(data_set[15], dtype=numpy.float32), numpy.asarray(data_set[16], dtype=numpy.float32), numpy.asarray(data_set[17], dtype=numpy.float32), numpy.asarray(data_set[18], dtype=numpy.float32), numpy.asarray(data_set[19], dtype=numpy.float32), numpy.asarray(data_set[20], dtype=numpy.float32), numpy.asarray(data_set[21], dtype=numpy.float32), numpy.asarray(data_set[22], dtype=numpy.float32), numpy.asarray(data_set[23], dtype=numpy.float32), numpy.asarray(data_set[24],dtype=numpy.float32), token_bit_seql_1v, token_bit_seql_2v, token_bit_seqr_v, same_token_bit_seql_1v, same_token_bit_seql_2v, same_token_bit_seqr_v, token_distance_seql_1v, token_distance_seql_2v, token_distance_seqr_v, numpy.asarray(data_set[34],dtype=numpy.float32), self.max_seql_length, self.max_ent_seql_length, self.max_ent_seqr_length]
        return data

    def create_padding_set(self):
        train_set = self.create_padding(self.train_set)
        
        print('max seq left length: ')
        print(self.max_seql_length)
        print('max seq right length: ')
        print(self.max_seqr_length)
        print('max entity seq left length: ')
        print(self.max_ent_seql_length)
        print('max entity seq right length: ')
        print(self.max_ent_seqr_length)
        
        return train_set
     
def main():
    scenario = 2
    parser = argparse.ArgumentParser('Generate training data')

    parser.add_argument('--conll_data_dir', 
                    type=str, 
                    default='conll_data', 
                    help='Directory to put the conll data.')
    parser.add_argument('--training_data',
                    type=str,
                    default='training_data.txt',
                    help='Path of the conll training data.')
    parser.add_argument('--output_file',
                    type=str,
                    default="output_training_array/conll_padding", 
                    help='Output file.')
    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()
    conll_data_dir = FLAGS.conll_data_dir
    training_data = FLAGS.training_data
    output_file = FLAGS.output_file
    # create CoNLL data set
    conll = CONLL(conll_data_dir, training_data, output_file, scenario)
    special_count = conll._special_count
        
    if len(conll.train_set[0]) > 0:
        train_set = conll.create_padding_set()
        temp_file = output_file+ "_" + str(special_count+1) + ".pkl"
        pickle.dump(train_set, open(temp_file, 'wb'))
    
if __name__ == "__main__":  
    main()