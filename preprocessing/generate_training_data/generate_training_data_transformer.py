# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 15:04:09 2019

@author: chenm
"""

import argparse
import numpy
import pickle


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
    
    # sequences and labels
    def load_data(self, input_file, output_file, verbose = False):
        self.raw_document_data_list = []
        
        seq_list_1 = []
        seq_list_2 = []        
        Y_list_1 = []       
        Y_list_2 = []
        label_list = []
        mention_pos_list_1 = []
        mention_pos_list_2 = []
        
        self.read_file(input_file)
        
        removed_examples = 0
        total_examples = 0
        special_count = 0
        for raw_document_data in self.raw_document_data_list:
            cluster_id_to_cluster_data = raw_document_data._gold_cluster_id_to_cluster_data
            results = self.get_similar_clusters(cluster_id_to_cluster_data)
            cluster_id_to_he_clusters = results[0]
            cluster_id_to_she_clusters = results[1]
            cluster_id_to_it_clusters = results[2]
            cluster_id_to_they_clusters = results[3]
                    
            raw_sequence_data_list = raw_document_data._raw_sequence_data_list
            for raw_sequence_data in raw_sequence_data_list:    
                focus_mention_cluster_id = raw_sequence_data._focus_mention_cluster_id
                focus_mention = raw_sequence_data._original_focus_mention                
                focus_mention_parts = focus_mention.split()
                focus_mention_cluster_data = cluster_id_to_cluster_data[focus_mention_cluster_id]
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
                
                updated_sequence_focus, focus_coreference_info_matrix, related_offset = self.get_updated_sequence(raw_sequence, related_mentions, mention_parts=None, focus_mention_len=len(focus_mention_parts), offset=0, related_mention_cluster_ids=related_mention_cluster_ids, focus_mention_cluster_id=focus_mention_cluster_id)
                
                focus_sequence_text = ''
                for usf in updated_sequence_focus:
                    focus_sequence_text = focus_sequence_text + ' ' + usf
                focus_sequence_text = focus_sequence_text.strip()
                 
                total_examples = total_examples + len(mention_set)
                
                if len(updated_sequence_focus) > 300:
                    removed_examples = removed_examples + len(mention_set)
                    continue
                            
                for i in range(len(mention_set)):
                    mention = mention_set[i]
                    
                    if mention.lower() != focus_mention.lower():                       
                        mention_parts = mention.split()                    

                        updated_sequence_other, other_coreference_info_matrix, related_offset = self.get_updated_sequence(raw_sequence, related_mentions, mention_parts=mention_parts, focus_mention_len=len(focus_mention_parts), offset=0, related_mention_cluster_ids=related_mention_cluster_ids, focus_mention_cluster_id=focus_mention_cluster_id)
                        
                        other_sequence_text = ''
                        for uso in updated_sequence_other:
                            other_sequence_text = other_sequence_text + ' ' + uso
                        other_sequence_text = other_sequence_text.strip()
                         
                        if len(updated_sequence_other) > 300:
                            removed_examples = removed_examples + 1
                            continue
                        
                        seq_list_1.append(focus_sequence_text)
                        focus_coreference_info_matrix_copy = focus_coreference_info_matrix.copy()
                        Y_list_1.append(focus_coreference_info_matrix_copy)
                        mention_pos_list_1.append(99+related_offset+len(focus_mention_parts))
                        seq_list_2.append(other_sequence_text)
                        Y_list_2.append(other_coreference_info_matrix)
                        mention_pos_list_2.append(99+related_offset+len(mention_parts))
                        label_list.append(1)
                        
                        
                if len(seq_list_1) >= 16:
                    special_count = special_count + 1
                    temp_data_set = [seq_list_1[:16],Y_list_1[:16], mention_pos_list_1[:16], seq_list_2[:16], Y_list_2[:16], mention_pos_list_2[:16], label_list[:16]]                    
                    temp_file = output_file+ "_" + str(special_count) + ".pkl"
                    temp_f = open(temp_file, 'wb')
                    pickle.dump(temp_data_set, temp_f)
                    temp_data_set = []
                            
                    seq_list_1 = seq_list_1[16:]
                    Y_list_1 = Y_list_1[16:]
                    mention_pos_list_1 = mention_pos_list_1[16:]
                    seq_list_2 = seq_list_2[16:]
                    Y_list_2 = Y_list_2[16:]
                    mention_pos_list_2 = mention_pos_list_2[16:]
                    label_list = label_list[16:]
                    temp_f.close()
                    
        self._special_count = special_count
        print(total_examples)
        print(removed_examples)                
        return [seq_list_1, Y_list_1, mention_pos_list_1, seq_list_2, Y_list_2, mention_pos_list_2, label_list]

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
    
    def create_padding(self, data_set):
        data = [data_set[0], data_set[1], data_set[2]]
        return data

    def create_padding_set(self):
        train_set = self.create_padding(self.train_set)
        
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
                    default='training_data_transformer_100.txt',
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
        temp_file = output_file + "_" + str(special_count+1) + ".pkl"
        temp_f = open(temp_file, 'wb')
        pickle.dump(conll.train_set, temp_f)
        temp_f.close()
        
if __name__ == "__main__":  
    main()