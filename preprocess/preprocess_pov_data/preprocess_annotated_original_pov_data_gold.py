# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 13:45:35 2019

@author: jayzohio
"""

import zipfile
import xml.dom.minidom
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter
from allennlp.predictors.predictor import Predictor
import allennlp_models.ner.crf_tagger
import allennlp_models.syntax.biaffine_dependency_parser
import allennlp_models.coref

import argparse

class MentionConversion(object):
    def __init__(self, start_pos, original_text, converted_text, cluster_id):
        self._start_pos = start_pos
        self._original_text = original_text
        self._converted_text = converted_text
        self._cluster_id = cluster_id
        
class VerbConversion(object):
    def __init__(self, start_pos, original_text, converted_text):
        self._start_pos = start_pos
        self._original_text = original_text
        self._converted_text = converted_text
 
def combine_dic(dic, color_dic=None):
    updated_dic = {}
    updated_color_dic = {}
    sorted_keys = sorted(dic.keys())
    
    count = 0
    previous_value = -1
    previous_key = -1
    previous_color = ''
    while count < len(sorted_keys):
        sorted_key = sorted_keys[count]
        value = dic[sorted_key]
        color = ''
        if color_dic != None:
            color = color_dic[sorted_key]
        
        if previous_value == -1:
            if count+1 < len(sorted_keys):
                next_key = sorted_keys[count+1]
                next_color = ''
                if color_dic != None:
                    next_color = color_dic[next_key]
                if (next_key == value+1) and (color == next_color):
                    previous_value = dic[next_key]
                    previous_key = sorted_key
                    previous_color = color
                    count = count + 2
                else:                    
                    updated_dic[sorted_key]=value
                    if color_dic != None:
                        updated_color_dic[sorted_key]=color
                    count = count + 1
            else:
                updated_dic[sorted_key] = value
                if color_dic != None:
                    updated_color_dic[sorted_key] = color
                previous_value = -1
                previous_key = -1
                previous_color = ''
                count = count + 1
        else:
            if (sorted_key == previous_value+1) and (color == previous_color):
                previous_value = dic[sorted_key]
                count = count + 1
            else:
                updated_dic[previous_key] = previous_value
                if color_dic != None:
                    updated_color_dic[previous_key] = previous_color
                
                if count+1 < len(sorted_keys):
                    next_key = sorted_keys[count+1]
                    next_color = ''
                    if color_dic != None:
                        next_color = color_dic[next_key]
                    if (next_key == value+1) and (color == next_color):
                        previous_value = dic[next_key]
                        previous_key = sorted_key
                        previous_color = color
                        count = count + 2
                    else:
                        updated_dic[sorted_key] = value
                        if color_dic != None:
                            updated_color_dic[sorted_key]=color
                        count = count + 1
                        previous_value = -1
                        previous_key = -1
                        previous_color = ''
                else:
                    updated_dic[sorted_key] = value
                    if color_dic != None:
                        updated_color_dic[sorted_key] = color
                    previous_value = -1
                    previous_key = -1
                    previous_color = ''
                    count = count + 1
                    
    if previous_value != -1:
        updated_dic[previous_key] = previous_value
        if color_dic != None:
            updated_color_dic[previous_key] = previous_color
            
    return updated_dic, updated_color_dic

def remove_underline_pos_from_highlight_dic(underline_dic, highlight_dic, highlight_color, document_text):
    final_highlight_dic = {}             
    final_highlight_color = {}
    
    for pos in highlight_dic.keys():
        if pos not in underline_dic:
            final_highlight_dic[pos]=highlight_dic[pos]
            final_highlight_color[pos]=highlight_color[pos]
        else:
            end = highlight_dic[pos]
            if end != underline_dic[pos]:
                # For testing purpose
                print("overlap underline and highlight:")
                print(document_text[pos:end+1])
                
    return final_highlight_dic, final_highlight_color

def verify_pos(pos, dic, document_text):
    sorted_keys = sorted(dic.keys())
    for sorted_key in sorted_keys:
        value = dic[sorted_key]
        if pos > sorted_key and pos <= value:
            # For testing purpose
            print("overlap positions")
            print(document_text[sorted_key:value+1])
            print(document_text[pos])

def get_color_to_cluster_id(color_dict, start_cluster_id):
    color_to_cluster_id = {}
    for pos in color_dict.keys():
        color = color_dict[pos]
        if color not in color_to_cluster_id:
            color_to_cluster_id[color] = start_cluster_id + len(color_to_cluster_id)
            
    return color_to_cluster_id

def get_cluster_id_to_color(highlight_color_to_cluster_id, shading_color_to_cluster_id):
    cluster_id_to_color = {}
    for color in highlight_color_to_cluster_id.keys():
        cluster_id = highlight_color_to_cluster_id[color]
        if cluster_id not in cluster_id_to_color:
            cluster_id_to_color[cluster_id] = color
        else:
            print("There is error - overlapping cluster id")
            print(cluster_id)
      
    for color in shading_color_to_cluster_id.keys():
        cluster_id = shading_color_to_cluster_id[color]
        if cluster_id not in cluster_id_to_color:
            cluster_id_to_color[cluster_id] = color
        else:
            print("There is error - overlapping cluster id")
            print(cluster_id)
            
    return cluster_id_to_color
            
def iterate_decorate_texts(wb_dic, underline_dic, highlight_dic, shading_dic, highlight_color, shading_color, document_text):
    updated_text = ""
    sorted_start_pos_list = sorted(wb_dic.keys()) 
    start_pos_to_verb_conversion = {}
    start_pos_to_mention_conversion = {}
    
    for pos in underline_dic.keys():
        verify_pos(pos, highlight_dic, document_text)
        verify_pos(pos, shading_dic, document_text)

    for pos in highlight_dic.keys():
        verify_pos(pos, underline_dic, document_text)
        verify_pos(pos, shading_dic, document_text)
        
    for pos in shading_dic.keys():
        verify_pos(pos, underline_dic, document_text)
        verify_pos(pos, highlight_dic, document_text)
    
    highlight_color_to_cluster_id = get_color_to_cluster_id(highlight_color, 0)
    shading_color_to_cluster_id = get_color_to_cluster_id(shading_color, len(highlight_color_to_cluster_id))
    previous_start = -1
    previous_end = -1
    skip_start = 0
    offset = 0
    for pos in sorted_start_pos_list:
        current_end = wb_dic[pos]
        if pos in underline_dic:
            if previous_start == -1:
                print("There is an error - searching verb conversion")
                print(document_text[pos:underline_dic[pos]+1])
            else:
                original_text = document_text[previous_start:previous_end+1]
                converted_text = document_text[pos:current_end+1]
                verb_conversion = VerbConversion(previous_start-offset, original_text, converted_text)
                if (previous_start-offset) not in start_pos_to_verb_conversion:
                    start_pos_to_verb_conversion[previous_start-offset]=verb_conversion
                else:
                    print("There is an error - overlapping verb conversions")
                    print(document_text[previous_start:previous_end+1])
                previous_start = -1
                previous_end = -1
                updated_text = updated_text + document_text[skip_start:pos]
                skip_start = current_end+1
                offset = offset + len(converted_text)
        elif pos in highlight_dic:
            color = highlight_color[pos]
            cluster_id = highlight_color_to_cluster_id[color]
            if previous_start == -1:
                original_text = document_text[pos:current_end+1]
                mention_conversion = MentionConversion(pos-offset, original_text, original_text, cluster_id)
                updated_text = updated_text + document_text[skip_start:current_end+1]
                skip_start = current_end+1
                if (pos-offset) not in start_pos_to_mention_conversion:
                    start_pos_to_mention_conversion[pos-offset]=mention_conversion
                else:
                    print("There is an error - overlapping mention conversions")
                    print(document_text[pos:current_end+1])
            else:
                original_text = document_text[previous_start:previous_end+1]
                converted_text = document_text[pos:current_end+1]
                mention_conversion = MentionConversion(previous_start-offset, original_text, converted_text, cluster_id)
                if (previous_start-offset) not in start_pos_to_mention_conversion:
                    start_pos_to_mention_conversion[previous_start-offset]=mention_conversion
                else:
                    print("There is an error - overlapping mention conversions")
                    print(document_text[pos:current_end+1])
                previous_start = -1
                previous_end = -1
                updated_text = updated_text + document_text[skip_start:pos]
                skip_start = current_end+1                
                offset = offset + len(converted_text)
        elif pos in shading_dic:
            color = shading_color[pos]
            cluster_id = shading_color_to_cluster_id[color]
            if previous_start == -1:
                original_text = document_text[pos:current_end+1]
                mention_conversion = MentionConversion(pos-offset, original_text, original_text, cluster_id)
                updated_text = updated_text + document_text[skip_start:current_end+1]
                skip_start = current_end+1
                if (pos-offset) not in start_pos_to_mention_conversion:
                    start_pos_to_mention_conversion[pos-offset]=mention_conversion
                else:
                    print("There is an error - overlapping mention conversions")
                    print(document_text[pos:current_end+1])
            else:
                original_text = document_text[previous_start:previous_end+1]
                converted_text = document_text[pos:current_end+1]
                mention_conversion = MentionConversion(previous_start-offset, original_text, converted_text, cluster_id)
                if (previous_start-offset) not in start_pos_to_mention_conversion:
                    start_pos_to_mention_conversion[previous_start-offset]=mention_conversion
                else:
                    print("There is an error - overlapping mention conversions")
                    print(document_text[pos:current_end+1])
                previous_start = -1
                previous_end = -1
                updated_text = updated_text + document_text[skip_start:pos]
                skip_start = current_end+1                    
                offset = offset + len(converted_text)
        else:
            previous_start = pos
            previous_end = current_end
            updated_text = updated_text + document_text[skip_start:current_end+1]
            skip_start = current_end+1
        
    updated_text = updated_text + document_text[skip_start:]
    return updated_text, start_pos_to_verb_conversion, start_pos_to_mention_conversion, highlight_color_to_cluster_id, shading_color_to_cluster_id

def convert_word_doc_to_text(file_path):
    document = zipfile.ZipFile(file_path)
    xml_document = xml.dom.minidom.parseString(document.read('word/document.xml'))
    itemlist = xml_document.getElementsByTagName('w:r')
    
    document_text = ""
    wb_dic = {}
    underline_dic = {}
    highlight_dic = {}
    highlight_color = {}
    shading_dic = {}
    shading_color = {}
    
    highlight_color_to_cluster_id = {}
    shading_color_to_cluster_id = {}
    start_pos_to_mention_conversion = {}
    start_pos_to_verb_conversion = {}
    cluster_id_to_mention_list = {}
    
    
    for item in itemlist:
        start_pos = len(document_text)
        text_fields = item.getElementsByTagName('w:t')
        text_field = None
        if len(text_fields) > 0:
            text_field = text_fields[0]
        else:
            continue
        text = text_field.firstChild.data
        text = text.replace(u'\xa0',u' ')
        text = text.replace('’','\'')
        leading_space_count = len(text) - len(text.lstrip())
        trailing_space_count = len(text) - len(text.rstrip())
        updated_start_pos = start_pos + leading_space_count
        updated_end_pos = start_pos + len(text) - trailing_space_count - 1
        other = item.getElementsByTagName('w:rPr')
        if other != None and len(other) > 0:
            decorate = other[0]
            
            wb = decorate.getElementsByTagName('w:b')
            if wb != None and len(wb) > 0:
                if updated_start_pos not in wb_dic:
                    wb_dic[updated_start_pos] = updated_end_pos
                elif wb_dic[updated_start_pos] < updated_end_pos:
                    # For testing purpose
                    print("overlap bold texts:")
                    print(document_text)
                    print(text)
                    wb_dic[updated_start_pos] = updated_end_pos
                    
            underline = decorate.getElementsByTagName('w:u')
            if underline != None and len(underline) > 0:
                if updated_start_pos not in underline_dic:
                    underline_dic[updated_start_pos] = updated_end_pos
                elif underline_dic[updated_start_pos] < updated_end_pos:
                    # For testing purpose
                    print("overlap underline texts:")
                    print(document_text)
                    print(text)
                    underline_dic[updated_start_pos] = updated_end_pos
                    
            highlight = decorate.getElementsByTagName('w:highlight')
            if highlight != None and len(highlight) > 0:
                color = highlight[0].getAttribute("w:val")
                if updated_start_pos not in highlight_dic:
                    highlight_dic[updated_start_pos] = updated_end_pos
                    highlight_color[updated_start_pos] = color
                elif highlight_dic[updated_start_pos] < updated_end_pos:
                    # For testing purpose
                    print("overlap highlight texts:")
                    print(document_text)
                    print(text)
                    highlight_dic[updated_start_pos] = updated_end_pos
                    highlight_color[updated_start_pos] = color
                    
            shading = decorate.getElementsByTagName('w:shd')
            if shading != None and len(shading) > 0:
                color = shading[0].getAttribute("w:fill")
                if updated_start_pos not in shading_dic:
                    shading_dic[updated_start_pos] = updated_end_pos
                    shading_color[updated_start_pos] = color
                elif shading_dic[updated_start_pos] < updated_end_pos:
                    # For testing purpose
                    print("overlap shading texts:")
                    print(document_text)
                    print(text)
                    shading_dic[updated_start_pos] = updated_end_pos
                    shading_color[updated_start_pos] = color
                    
        document_text = document_text + text
 
    updated_underline_dic, _ = combine_dic(underline_dic)
    updated_highlight_dic, updated_highlight_color = combine_dic(highlight_dic, highlight_color)
    updated_shading_dic, updated_shading_color = combine_dic(shading_dic, shading_color)
    final_highlight_dic, final_highlight_color = remove_underline_pos_from_highlight_dic(updated_underline_dic, updated_highlight_dic, updated_highlight_color, document_text)
    
    updated_text, start_pos_to_verb_conversion, start_pos_to_mention_conversion, highlight_color_to_cluster_id, shading_color_to_cluster_id = iterate_decorate_texts(wb_dic, underline_dic, highlight_dic, shading_dic, highlight_color, shading_color, document_text)    
    
    xml_document = None
    return updated_text, start_pos_to_verb_conversion, start_pos_to_mention_conversion, highlight_color_to_cluster_id, shading_color_to_cluster_id    
  
def map_cfr_index_to_token_index(cfr_document):
    cfr_index_to_token_index = {}

    token_count = 0
    for i in range(len(cfr_document)):
        word = cfr_document[i]
        if word == ' ' or word == '\t':
            cfr_index_to_token_index[i] = -1
        else:
            cfr_index_to_token_index[i] = token_count
            token_count = token_count + 1
            
    return cfr_index_to_token_index

def change_verb_conjugation(document_text, token_index_to_start_pos):
    splitter = SpacySentenceSplitter()
    sentences = splitter.split_sentences(document_text)
    tokenizer = SpacyTokenizer(pos_tags=True,ner=True)
    
    predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz")    

    token_count = 0
    
    subject_pos_list = []
    object_pos_list = []
    
    for i in range(len(sentences)):
        sentence = sentences[i]
        result = predictor.predict(sentence=sentence)
        words = result['words']
        tokens = tokenizer.tokenize(sentence)
        
        # For testing purpose
        if len(tokens) != len(words):
            print("There is difference: tokenizer and parser length.")
        else:
            for j in range(len(tokens)):
                token = tokens[j]
                word = words[j]
                if token.text != word:
                    print("There is difference: ")
                    print(token.text)
                    print(word)
                    
        dependencies = result['predicted_dependencies']
        
        for j in range(len(tokens)):
            token = tokens[j]
            token_index = token_count
            token_count = token_count + 1
            start_pos = token_index_to_start_pos[token_index]
            dep = dependencies[j]
            
            if dep == 'nsubj' or dep == 'nsubjpass':
                subject_pos_list.append(start_pos)
             
            if dep == 'dobj' or dep == 'iobj':
                object_pos_list.append(start_pos)
                  
    splitter = None
    tokenizer = None
    predictor = None
    return subject_pos_list, object_pos_list
            
def preprocess(document_text):          
    start_pos_to_token = {}
    start_pos_to_sentence_num = {}
    sentence_num_to_sent_start_num = {}
    start_pos_to_token_index = {}
    token_index_to_start_pos = {}
    sentence_num_to_token_index_list = {}
    
    predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz")
    current_start = 0
    coreference_chains = predictor.predict(document=document_text) 
    predictor = None
    cfr_document = coreference_chains['document']
    cfr_clusters = coreference_chains['clusters']
    coreference_chains = None
    coreference_chains = None
    cfr_index_to_token_index = map_cfr_index_to_token_index(cfr_document)
    
    splitter = SpacySentenceSplitter()
    sentences = splitter.split_sentences(document_text)
    tokens_list = []
    tokenizer = SpacyTokenizer(pos_tags=True, ner=True)
    token_count = 0
    for i in range(len(sentences)):
        sentence = sentences[i]
        sentence_index = document_text.index(sentence,current_start)
        sentence_num_to_sent_start_num[i] = sentence_index
        tokens = tokenizer.tokenize(sentence)
        tokens_list.append(tokens)
        token_index_list = []
        for token in tokens:
            if token.text == ' ' or token.text == '\t':
                print("There is an error")
            elif ' 'in token.text or '\t' in token.text:
                print("There is space.")
                print(token.text)
                print(tokens)
                
            offset = token.idx                       
            start_pos = sentence_index + offset
            if start_pos not in start_pos_to_token:
                start_pos_to_token[start_pos] = token
                start_pos_to_sentence_num[start_pos] = i
                start_pos_to_token_index[start_pos] = token_count
                token_index_to_start_pos[token_count] = start_pos
            else:
                print("There is an error in the sentence, start_pos: " + str(start_pos))
            token_index_list.append(token_count)
            token_count = token_count + 1
            
        sentence_num_to_token_index_list[i]=token_index_list
        current_start = sentence_index + len(sentence)
    
    splitter = None
    tokenizer = None
    predictor = None        
    return sentences, start_pos_to_token, start_pos_to_sentence_num, sentence_num_to_sent_start_num, cfr_document, cfr_clusters, cfr_index_to_token_index, start_pos_to_token_index, token_index_to_start_pos,sentence_num_to_token_index_list, tokens_list

def identify_quotes(text):
    left_quote_list = []
    right_quote_list = []
    current_left_quote = ''
    current_left_quote_index = -1
    for i in range(len(text)):
        c = text[i]
        if c == '“':
            current_left_quote = c
            current_left_quote_index = i
        elif c == '”':
            if current_left_quote == '“':
                left_quote_list.append(current_left_quote_index)
                right_quote_list.append(i)
            current_left_quote = ''
            current_left_quote_index = -1
        elif c == '\"':
            if current_left_quote == '\"':
                left_quote_list.append(current_left_quote_index)
                right_quote_list.append(i)
                current_left_quote = ''
                current_left_quote_index = -1
            else:
                current_left_quote = c
                current_left_quote_index = i

    return left_quote_list, right_quote_list        

def check_mention_in_quotation_and_dialogue(start_pos, end_pos, left_quote_list, right_quote_list):
    is_valid = True
    
    for i in range(len(left_quote_list)):
        left_start = left_quote_list[i]
        right_end = right_quote_list[i]
        if (start_pos > left_start) and (end_pos < right_end):
            is_valid = False
            break
        
    return is_valid

def correct_string(s):
    previous_c = ''
    updated_s = ''
    for i in range(len(s)):
        c = s[i]
        if c == '\'' and previous_c != ' ' and previous_c != '':
            updated_s=updated_s+' ' + c
        elif c == ',' and previous_c != ' ' and previous_c != '':
            updated_s =updated_s+' ' + c
        elif c == '-' and previous_c != ' ' and previous_c != '':
            updated_s = updated_s+' '+c+' '
        else:
            updated_s = updated_s + c
        previous_c = c   
    return updated_s

def write_cluster_info(f, start_pos_to_mention_conversion, highlight_color_to_cluster_id, shading_color_to_cluster_id, focus_mention_string_set, start_pos_to_token):
    valid_cluster_ids = []
  
    first_person_pronouns = ['i','me','my','myself','mine']
    
    cluster_id_to_color = get_cluster_id_to_color(highlight_color_to_cluster_id, shading_color_to_cluster_id)
    
    gold_cluster_id_to_mentions = {}
    auto_cluster_id_to_mentions = {}
    cluster_id_to_mention_tags = {}
    
    valid_cluster_ids.append(-1)
    valid_cluster_ids.append(-2)
    sorted_start_pos_list = sorted(start_pos_to_token.keys())
    for start_pos in start_pos_to_mention_conversion:
        mention_conversion = start_pos_to_mention_conversion[start_pos]
        original_text = mention_conversion._original_text.lower()
        original_text = correct_string(original_text)
        converted_text = mention_conversion._converted_text.lower()
        converted_text = correct_string(converted_text)
        cluster_id = mention_conversion._cluster_id
        color = cluster_id_to_color[cluster_id]
        tag_list = []
        mention_length = len(original_text.split())
        if start_pos not in start_pos_to_token:
            print(start_pos)
            print(original_text)
            print(converted_text)
            print(color)
            continue
        insert_index = sorted_start_pos_list.index(start_pos)
        for iii in range(insert_index,insert_index+mention_length):
            sss = sorted_start_pos_list[iii]
            ttt = start_pos_to_token[sss]
            tag = ttt.tag_
            tag_list.append(tag)
            
        if color == 'yellow':
            if mention_length != 1:
                print('error - double gold yellow mention')
            if original_text.lower() in first_person_pronouns:
                if -1 not in gold_cluster_id_to_mentions:
                    ll = []
                    ll.append(converted_text.lower())
                    gold_cluster_id_to_mentions[-1]=ll
                    ttl = []
                    ttl.append(tag_list)
                    cluster_id_to_mention_tags[-1]=ttl
                else:
                    ll = gold_cluster_id_to_mentions[-1]
                    ll.append(converted_text.lower())
                    gold_cluster_id_to_mentions[-1]=ll
                    ttl = cluster_id_to_mention_tags[-1]
                    ttl.append(tag_list)
                    cluster_id_to_mention_tags[-1]=ttl
            else:
                if -2 not in gold_cluster_id_to_mentions:
                    ll = []
                    ll.append(converted_text.lower())
                    gold_cluster_id_to_mentions[-2]=ll
                    ttl = []
                    ttl.append(tag_list)
                    cluster_id_to_mention_tags[-2]=ttl
                else:
                    ll = gold_cluster_id_to_mentions[-2]
                    ll.append(converted_text.lower())
                    gold_cluster_id_to_mentions[-2] = ll
                    ttl = cluster_id_to_mention_tags[-2]
                    ttl.append(tag_list)
                    cluster_id_to_mention_tags[-2]=ttl
        else:
            if cluster_id not in valid_cluster_ids:
                valid_cluster_ids.append(cluster_id)
            if cluster_id not in gold_cluster_id_to_mentions:
                ll = []
                ll.append(converted_text.lower())
                gold_cluster_id_to_mentions[cluster_id] = ll
                ttl = []
                ttl.append(tag_list)
                cluster_id_to_mention_tags[cluster_id]=ttl
            else:
                ll = gold_cluster_id_to_mentions[cluster_id]
                ll.append(converted_text.lower())
                gold_cluster_id_to_mentions[cluster_id] = ll
                ttl = cluster_id_to_mention_tags[cluster_id]
                ttl.append(tag_list)
                cluster_id_to_mention_tags[cluster_id]=ttl
            if cluster_id not in auto_cluster_id_to_mentions:
                ll = []
                ll.append(original_text.lower())
                auto_cluster_id_to_mentions[cluster_id] = ll
            else:
                ll = auto_cluster_id_to_mentions[cluster_id]
                ll.append(original_text.lower())
                auto_cluster_id_to_mentions[cluster_id] = ll
         
    auto_cluster_id_to_mentions[-1] = focus_mention_string_set
    auto_cluster_id_to_mentions[-2] = ['they','them','their','themselves','theirs', 'they themselves']
            
    # Write cluster info
    for gold_cluster_id in gold_cluster_id_to_mentions:
        mention_text_list = gold_cluster_id_to_mentions[gold_cluster_id]
        mention_postag_list = cluster_id_to_mention_tags[gold_cluster_id]
        
        mention_to_count = {}
        mention_to_postags = {}

        for i in range(len(mention_text_list)):
            mention_text = mention_text_list[i]
            mention_postags = mention_postag_list[i]

            if mention_text.lower() not in mention_to_count:
                mention_to_count[mention_text.lower()] = 1
                mention_to_postags[mention_text.lower()] = mention_postags
            else:
                mention_to_count[mention_text.lower()] = mention_to_count[mention_text.lower()] + 1

        f.write("<gold_cluster_id>: ")
        f.write(str(gold_cluster_id))
        f.write("\t")
        max_count_mention = ""
        max_count = 0
        total_count = 0
        for mention, count in mention_to_count.items():
            total_count = total_count + count
            if count > max_count:
                max_count = count
                max_count_mention = mention
        f.write("<common_mention>: ")
        f.write(max_count_mention)
        f.write("\t")
        f.write("<max_count>: ")
        f.write(str(max_count))
        f.write("\t")
        f.write("<total_count>: ")
        f.write(str(total_count))
        f.write("\t")
                
        for mention, postags in mention_to_postags.items():
            f.write("<mention>: ")
            f.write(mention)
            f.write("\t")
            f.write("<postag>: ")
            for j in range(len(postags)-1):
                postag = postags[j]
                f.write(postag)
                f.write(' ')
            f.write(postags[len(postags)-1])
            f.write("\t")
        f.write("\n")
   
    for auto_cluster_id in auto_cluster_id_to_mentions:
        mention_text_list = auto_cluster_id_to_mentions[auto_cluster_id]
        
        mention_to_count = {}

        for i in range(len(mention_text_list)):
            mention_text = mention_text_list[i]

            if mention_text.lower() not in mention_to_count:
                mention_to_count[mention_text.lower()] = 1
            else:
                mention_to_count[mention_text.lower()] = mention_to_count[mention_text.lower()] + 1

        f.write("<auto_cluster_id>: ")
        f.write(str(auto_cluster_id))
        f.write("\t")
        max_count_mention = ""
        max_count = 0
        total_count = 0
        for mention, count in mention_to_count.items():
            total_count = total_count + count
            if count > max_count:
                max_count = count
                max_count_mention = mention
        f.write("<common_mention>: ")
        f.write(max_count_mention)
        f.write("\t")
        f.write("<max_count>: ")
        f.write(str(max_count))
        f.write("\t")
        f.write("<total_count>: ")
        f.write(str(total_count))
        f.write("\t")
                
        for mention in mention_to_count:
            f.write("<mention>: ")
            f.write(mention)
            f.write("\t")
        f.write("\n")
        
    return valid_cluster_ids,gold_cluster_id_to_mentions,auto_cluster_id_to_mentions,cluster_id_to_mention_tags

def get_from_info(starting_sent_num, starting_index, sentences, tokens_list):
    total_token_size = 50
    pre_padding = 0
    from_sent_num = starting_sent_num
    from_token_index = starting_index
    for i in range(starting_sent_num, -1, -1):
        if i == starting_sent_num:
            if starting_index >= 50:
                from_token_index = starting_index - 50
                total_token_size = 0
                break
            else:
                total_token_size = 50 - starting_index                            
        else:
            tokens = tokens_list[i]
            sentence_len = len(tokens)
            if sentence_len >= total_token_size:
                from_sent_num = i
                from_token_index = sentence_len - total_token_size
                total_token_size = 0
                break
            else:
                total_token_size = total_token_size - sentence_len
                
    if total_token_size != 0:
        from_sent_num = 0
        from_token_index = 0
        pre_padding = total_token_size  
                    
    return from_sent_num, from_token_index, pre_padding

def get_to_info(starting_sent_num, starting_index, sentences, tokens_list):
    total_token_size = 50
    post_padding = 0
    to_sent_num = starting_sent_num
    to_token_index = starting_index
    pre_sentence_len = 0
    for i in range(starting_sent_num, len(sentences)):
        tokens = tokens_list[i]
        sentence_len = len(tokens)
        pre_sentence_len = len(tokens)
        if i == starting_sent_num:
            if starting_index + 51 <= sentence_len:
                to_token_index = starting_index + 50
                total_token_size = 0
                break
            else:
                total_token_size = 51 - (sentence_len - starting_index)
        else:
            if sentence_len >= total_token_size:
                to_sent_num = i
                to_token_index = total_token_size - 1
                total_token_size = 0
                break
            else:
                total_token_size = total_token_size - sentence_len
    if total_token_size != 0:
        to_sent_num = len(sentences) - 1
        to_token_index = pre_sentence_len - 1
        post_padding = total_token_size
        
    return to_sent_num, to_token_index, post_padding

def fourth_round_process(pre_padding, post_padding, from_sent_num, to_sent_num, from_token_index, to_token_index, sentences, sent_num_to_mentions, key_to_cluster_id, key_to_original_mention_text, key_to_converted_mention_text, original_mention_len, converted_mention_len, tokens_list, sentence_num_to_token_index_list, token_index_to_start_pos, start_pos_to_mention_conversion, start_pos_to_verb_conversion,invalid_cluster_ids=[]):
    related_mentions = []
    related_mention_cluster_ids = []
    original_related_mention_text_list = []
    converted_related_mention_text_list = []
    raw_sequence = []
    postags = []
    
    total_len = 100 + original_mention_len
    
    if pre_padding != 0:
        for i in range(pre_padding):
            raw_sequence.append("<pad>")
            postags.append("<NONE>")
                
    for i in range(from_sent_num, to_sent_num+1):
        tokens = tokens_list[i]
        token_index_list = sentence_num_to_token_index_list[i]
        if i == from_sent_num and i != to_sent_num:
            for j in range(from_token_index, len(tokens)):
                token = tokens[j]
                
                to_add_text = token.text.lower()
                temp_index = token_index_list[j]
                temp_start_pos = token_index_to_start_pos[temp_index]
                if temp_start_pos in start_pos_to_verb_conversion:
                    verb_conversion = start_pos_to_verb_conversion[temp_start_pos]
                    temp_original_text = verb_conversion._original_text.lower()
                    if temp_original_text != token.text.lower():
                        print('error - double gold verb')
                    temp_converted_text = verb_conversion._converted_text
                    to_add_text = temp_converted_text.lower()
                    
                raw_sequence.append(to_add_text)
                postags.append(token.tag_)
                            
                if i in sent_num_to_mentions:
                    ms = sent_num_to_mentions[i]
                    m = ms[j]
                    if m != 0:
                        m_key = str(i) + ":" + str(j) + ":" + str(m)
                        m_cluster_id = key_to_cluster_id[m_key]
                        
                        if m_cluster_id in invalid_cluster_ids:
                            continue
                        
                        tt_index = token_index_list[j]
                        tt_start_pos = token_index_to_start_pos[tt_index]
                        tt_mc = start_pos_to_mention_conversion[tt_start_pos]
                        tt_ot = tt_mc._original_text
                        tt_ct = tt_mc._converted_text
                        
                        related_mention = []
                        if m == -1:
                            m = 0
                        if len(raw_sequence) - 1 != 50 and len(raw_sequence)-1+m-j <= total_len - 1:
                            related_mention.append(len(raw_sequence)-1)
                            related_mention.append(len(raw_sequence)-1+m-j)                            
                            related_mentions.append(related_mention)
                            related_mention_cluster_ids.append(m_cluster_id)
                            original_related_mention_text_list.append(tt_ot)
                            converted_related_mention_text_list.append(tt_ct)
        elif i != from_sent_num and i == to_sent_num:
            for j in range(0, to_token_index+1):
                token = tokens[j]
                to_add_text = token.text.lower()
                temp_index = token_index_list[j]
                temp_start_pos = token_index_to_start_pos[temp_index]
                if temp_start_pos in start_pos_to_verb_conversion:
                    verb_conversion = start_pos_to_verb_conversion[temp_start_pos]
                    temp_original_text = verb_conversion._original_text.lower()
                    if temp_original_text != token.text.lower():
                        print('error - double gold verb')
                    temp_converted_text = verb_conversion._converted_text
                    to_add_text = temp_converted_text.lower()
                    
                raw_sequence.append(to_add_text)
                postags.append(token.tag_)
                            
                if i in sent_num_to_mentions:
                    ms = sent_num_to_mentions[i]
                    m = ms[j]
                    if m != 0:
                        m_key = str(i) + ":" + str(j) + ":" + str(m)
                        m_cluster_id = key_to_cluster_id[m_key]
                        
                        if m_cluster_id in invalid_cluster_ids:
                            continue
                        
                        tt_index = token_index_list[j]
                        tt_start_pos = token_index_to_start_pos[tt_index]
                        tt_mc = start_pos_to_mention_conversion[tt_start_pos]
                        tt_ot = tt_mc._original_text
                        tt_ct = tt_mc._converted_text
                        
                        related_mention = []
                        if m == -1:
                            m = 0
                        if len(raw_sequence) - 1 != 50 and len(raw_sequence)-1+m-j <= total_len - 1:
                            related_mention.append(len(raw_sequence)-1)
                            related_mention.append(len(raw_sequence)-1+m-j)
                            related_mentions.append(related_mention)
                            related_mention_cluster_ids.append(m_cluster_id)
                            original_related_mention_text_list.append(tt_ot)
                            converted_related_mention_text_list.append(tt_ct)
        elif i != from_sent_num and i != to_sent_num:
            for j in range(0, len(tokens)):
                token = tokens[j]
                to_add_text = token.text.lower()
                temp_index = token_index_list[j]
                temp_start_pos = token_index_to_start_pos[temp_index]
                if temp_start_pos in start_pos_to_verb_conversion:
                    verb_conversion = start_pos_to_verb_conversion[temp_start_pos]
                    temp_original_text = verb_conversion._original_text.lower()
                    if temp_original_text != token.text.lower():
                        print('error - double gold verb')
                    temp_converted_text = verb_conversion._converted_text
                    to_add_text = temp_converted_text.lower()
                    
                raw_sequence.append(to_add_text)
                postags.append(token.tag_)
                            
                if i in sent_num_to_mentions:
                    ms = sent_num_to_mentions[i]
                    m = ms[j]
                    if m != 0:
                        m_key = str(i) + ":" + str(j) + ":" + str(m)
                        m_cluster_id = key_to_cluster_id[m_key]
                        
                        if m_cluster_id in invalid_cluster_ids:
                            continue
                        
                        tt_index = token_index_list[j]
                        tt_start_pos = token_index_to_start_pos[tt_index]
                        tt_mc = start_pos_to_mention_conversion[tt_start_pos]
                        tt_ot = tt_mc._original_text
                        tt_ct = tt_mc._converted_text
                        
                        related_mention = []
                        if m == -1:
                            m = 0
                        if len(raw_sequence) - 1 != 50 and len(raw_sequence)-1+m-j <= total_len - 1:
                            related_mention.append(len(raw_sequence)-1)
                            related_mention.append(len(raw_sequence)-1+m-j)
                            related_mentions.append(related_mention)
                            related_mention_cluster_ids.append(m_cluster_id)
                            original_related_mention_text_list.append(tt_ot)
                            converted_related_mention_text_list.append(tt_ct)
        elif i == from_sent_num and i == to_sent_num:
            for j in range(from_token_index, to_token_index+1):
                token = tokens[j]
                to_add_text = token.text.lower()
                temp_index = token_index_list[j]
                temp_start_pos = token_index_to_start_pos[temp_index]
                if temp_start_pos in start_pos_to_verb_conversion:
                    verb_conversion = start_pos_to_verb_conversion[temp_start_pos]
                    temp_original_text = verb_conversion._original_text.lower()
                    if temp_original_text != token.text.lower():
                        print('error - double gold verb')
                    temp_converted_text = verb_conversion._converted_text
                    to_add_text = temp_converted_text.lower()
                    
                raw_sequence.append(to_add_text)
                postags.append(token.tag_)
                            
                if i in sent_num_to_mentions:
                    ms = sent_num_to_mentions[i]
                    m = ms[j]
                    if m != 0:
                        m_key = str(i) + ":" + str(j) + ":" + str(m)
                        m_cluster_id = key_to_cluster_id[m_key]
                        
                        if m_cluster_id in invalid_cluster_ids:
                            continue
                        
                        tt_index = token_index_list[j]
                        tt_start_pos = token_index_to_start_pos[tt_index]
                        tt_mc = start_pos_to_mention_conversion[tt_start_pos]
                        tt_ot = tt_mc._original_text
                        tt_ct = tt_mc._converted_text
                        
                        related_mention = []
                        if m == -1:
                            m = 0
                        if len(raw_sequence) - 1 != 50 and len(raw_sequence)-1+m-j <= total_len - 1:
                            related_mention.append(len(raw_sequence)-1)
                            related_mention.append(len(raw_sequence)-1+m-j)
                            related_mentions.append(related_mention)
                            related_mention_cluster_ids.append(m_cluster_id)
                            original_related_mention_text_list.append(tt_ot)
                            converted_related_mention_text_list.append(tt_ct)
                            
    if post_padding != 0:
        for i in range(post_padding):
            raw_sequence.append("<pad>")
            postags.append("<NONE>")
    
    return related_mentions, original_related_mention_text_list, converted_related_mention_text_list, related_mention_cluster_ids, raw_sequence, postags
 
def write_sequence_info(f, related_mentions, original_related_mention_text_list, converted_related_mention_text_list, related_mention_cluster_ids, raw_sequence, postags):
    for i in range(len(related_mentions)):
        related_mention = related_mentions[i]
        related_mention_cluster_id = related_mention_cluster_ids[i]
        f.write("<related_mention>: ")
        f.write(str(related_mention[0]))
        f.write(" ")
        f.write(str(related_mention[1]))
        f.write("\t")
        f.write("<cluster_id>: ")
        f.write(str(related_mention_cluster_id))
        f.write('\t')
        f.write('<original_related_mention_text>: ')
        f.write(original_related_mention_text_list[i])
        f.write('\t')
        f.write('<converted_related_mention_text>: ')
        f.write(converted_related_mention_text_list[i])
        f.write("\n")
        
    if len(related_mentions) == 0:
        f.write("<no related mentions>\n")
                    
    f.write("<raw_sequence>: ")
    for rs in raw_sequence:
        if rs != "\n":
            f.write("%s " %rs)
        else:
            f.write("<\n> ")
    f.write("\n")
    
    f.write("<postags>: ")
    for rsp in postags:
        if rsp != "\n":
            f.write("%s " %rsp)
        else:
            f.write("<\n> ")
    f.write("\n") 

def get_from_mention_info(starting_sent_num, starting_index, sentences, tokens_list, sent_num_to_mentions, key_to_cluster_id, key_to_original_mention_text, key_to_converted_mention_text):
    original_pre_mention_sequence = []
    converted_pre_mention_sequence = []
    pre_mention_cluster_id_sequence = []
    pre_mention_distance_sequence = []
    pre_sent_index_list = []
    pre_start_pos_list = []
    pre_end_pos_list = []
                
    total_mention_size = 11
    
    for i in range(starting_sent_num, -1, -1):
        if total_mention_size == 0:
            break
        if i in sent_num_to_mentions:
            ms = sent_num_to_mentions[i]
            if i == starting_sent_num:
                for j in range(starting_index, -1, -1):
                    if total_mention_size == 0:
                        break
                    m = ms[j]
                    if m != 0:
                        m_key = str(i) + ":" + str(j) + ":" + str(m)
                        m_cluster_id = key_to_cluster_id[m_key]
                        m_original_text = key_to_original_mention_text[m_key]
                        m_converted_text = key_to_converted_mention_text[m_key]
                        if m == -1:
                            m = 0
                        original_pre_mention_sequence.insert(0, m_original_text)
                        converted_pre_mention_sequence.insert(0, m_converted_text)
                        pre_mention_cluster_id_sequence.insert(0, m_cluster_id)
                        pre_sent_index_list.insert(0, i)
                        pre_start_pos_list.insert(0, j)
                        pre_end_pos_list.insert(0, m)
                        total_mention_size = total_mention_size - 1                                               
            else:
                for j in range(len(ms)-1, -1, -1):
                    if total_mention_size == 0:
                        break
                    m = ms[j]
                    if m != 0:
                        m_key = str(i) + ":" + str(j) + ":" + str(m)
                        m_cluster_id = key_to_cluster_id[m_key]
                        m_original_text = key_to_original_mention_text[m_key]
                        m_converted_text = key_to_converted_mention_text[m_key]
                        if m == -1:
                            m = 0
                        original_pre_mention_sequence.insert(0, m_original_text)
                        converted_pre_mention_sequence.insert(0, m_converted_text)
                        pre_mention_cluster_id_sequence.insert(0, m_cluster_id)
                        pre_sent_index_list.insert(0, i)
                        pre_start_pos_list.insert(0, j)
                        pre_end_pos_list.insert(0, m)
                        total_mention_size = total_mention_size - 1 
                        
    if total_mention_size != 0:
        for i in range(total_mention_size):
            original_pre_mention_sequence.insert(0, "<pad>")
            converted_pre_mention_sequence.insert(0,"<pad>")
            pre_mention_cluster_id_sequence.insert(0, -3)

    if total_mention_size > 0:
        for i in range(total_mention_size):
            pre_mention_distance_sequence.append(0)
                
    for i in range(len(pre_sent_index_list)):
        if i == 0:
            distance = 0
            current_sent_index = pre_sent_index_list[i]
            current_start_index = pre_start_pos_list[i]
            if total_mention_size > 0:
                if current_sent_index > 0:
                    for j in range(current_sent_index):
                        ss_tokens = tokens_list[j]
                        distance = distance + len(ss_tokens)
                distance = distance + current_start_index
                pre_mention_distance_sequence.append(distance)
            else:
                found_mention = False
                if current_sent_index in sent_num_to_mentions:
                    temp_mentions = sent_num_to_mentions[current_sent_index]                                
                    for j in range(current_start_index-1, -1, -1):
                        temp_mention = temp_mentions[j]
                        if temp_mention != 0:
                            if temp_mention == -1:
                                temp_mention = 0
                            distance = current_start_index - temp_mention - 1
                            pre_mention_distance_sequence.append(distance)
                            found_mention = True
                            break
                if not found_mention:
                    distance = distance + current_start_index - 1
                    if current_sent_index-1 >= 0:
                        for j in range(current_sent_index-1, -1, -1):
                            if j in sent_num_to_mentions:
                                temp_mentions = sent_num_to_mentions[j]
                                for h in range(len(temp_mentions)-1, -1, -1):
                                    temp_mention = temp_mentions[h]
                                    if temp_mention != 0:
                                        if temp_mention == -1:
                                            temp_mention = 0
                                        distance = distance + len(temp_mentions) - temp_mention - 1
                                        pre_mention_distance_sequence.append(distance)
                                        found_mention = True
                                        break
                            if not found_mention:
                                ss_tokens = tokens_list[j]
                                distance = distance + len(ss_tokens)
                            else:
                                break
                if not found_mention:
                    pre_mention_distance_sequence.append(distance)
        else:
            previous_sent_index = pre_sent_index_list[i-1]
            previous_tokens = tokens_list[previous_sent_index]
            previous_sent_len = len(previous_tokens)
            current_sent_index = pre_sent_index_list[i]
            previous_end_index = pre_end_pos_list[i-1]
            current_start_index = pre_start_pos_list[i]
                    
            distance = 0
            if current_sent_index-1 >= previous_sent_index+1:
                for j in range(previous_sent_index+1, current_sent_index):
                    ss_tokens = tokens_list[j]
                    distance = distance + len(ss_tokens)
            if current_sent_index != previous_sent_index:
                distance = distance + current_start_index
                distance = distance + previous_sent_len - previous_end_index - 1
            else:
                distance = distance + current_start_index - previous_end_index - 1
                
            pre_mention_distance_sequence.append(distance)
            
    return original_pre_mention_sequence, converted_pre_mention_sequence, pre_mention_cluster_id_sequence, pre_mention_distance_sequence, pre_sent_index_list, pre_end_pos_list, pre_start_pos_list

def get_to_mention_info(starting_sent_num, starting_index, sentences, tokens_list, sent_num_to_mentions, key_to_cluster_id, key_to_original_mention_text, key_to_converted_mention_text, pre_sent_index_list, pre_end_pos_list):
    original_post_mention_sequence = []
    converted_post_mention_sequence = []
    post_mention_cluster_id_sequence = []
    post_mention_distance_sequence = []
    post_sent_index_list = []
    post_start_pos_list = []
    post_end_pos_list = []
        
    total_mention_size = 10

    for i in range(starting_sent_num, len(sentences)):
        if total_mention_size == 0:
            break
        if i in sent_num_to_mentions:
            ms = sent_num_to_mentions[i]
            
            if i == starting_sent_num:
                for j in range(starting_index+1, len(ms)):
                    if total_mention_size == 0:
                        break
                    m = ms[j]
                    if m != 0:
                        m_key = str(i) + ":" + str(j) + ":" + str(m)
                        m_cluster_id = key_to_cluster_id[m_key]
                        m_original_text = key_to_original_mention_text[m_key]
                        m_converted_text = key_to_converted_mention_text[m_key]
                        if m == -1:
                            m = 0
                        original_post_mention_sequence.append(m_original_text)
                        converted_post_mention_sequence.append(m_converted_text)
                        post_mention_cluster_id_sequence.append(m_cluster_id)
                        post_sent_index_list.append(i)
                        post_start_pos_list.append(j)
                        post_end_pos_list.append(m)
                        total_mention_size = total_mention_size - 1           
            else:
                for j in range(len(ms)):
                    if total_mention_size == 0:
                        break
                    m = ms[j]
                    if m != 0:
                        m_key = str(i) + ":" + str(j) + ":" + str(m)
                        m_cluster_id = key_to_cluster_id[m_key]
                        m_original_text = key_to_original_mention_text[m_key]
                        m_converted_text = key_to_converted_mention_text[m_key]
                        if m == -1:
                            m = 0
                        original_post_mention_sequence.append(m_original_text)
                        converted_post_mention_sequence.append(m_converted_text)
                        post_mention_cluster_id_sequence.append(m_cluster_id)
                        post_sent_index_list.append(i)
                        post_start_pos_list.append(j)
                        post_end_pos_list.append(m)
                        total_mention_size = total_mention_size - 1 
                                                  
    for i in range(len(post_sent_index_list)):
        previous_sent_index = -1
        previous_end_index = -1
        if i == 0:
            previous_sent_index = pre_sent_index_list[len(pre_sent_index_list)-1]      
            previous_end_index = pre_end_pos_list[len(pre_end_pos_list)-1]
        else:
            previous_sent_index = post_sent_index_list[i-1]
            previous_end_index = post_end_pos_list[i-1]
                        
        previous_tokens = tokens_list[previous_sent_index]
        previous_sent_len = len(previous_tokens)
        current_sent_index = post_sent_index_list[i]                  
        current_start_index = post_start_pos_list[i]
                    
        distance = 0
        if current_sent_index-1 >= previous_sent_index+1:
            for j in range(previous_sent_index+1, current_sent_index):
                ss_tokens = tokens_list[j]
                distance = distance + len(ss_tokens)
        if current_sent_index != previous_sent_index:
            distance = distance + current_start_index
            distance = distance + previous_sent_len - previous_end_index - 1
        else:
            distance = distance + current_start_index - previous_end_index - 1
        post_mention_distance_sequence.append(distance)
                                                          
    if total_mention_size != 0:
        for i in range(total_mention_size):
            original_post_mention_sequence.append("<pad>")
            converted_post_mention_sequence.append("<pad>")
            post_mention_cluster_id_sequence.append(-3)
            if i == 0:
                previous_sent_index = -1
                previous_end_index = -1
                if len(post_sent_index_list) == 0:
                    previous_sent_index = pre_sent_index_list[len(pre_sent_index_list)-1]      
                    previous_end_index = pre_end_pos_list[len(pre_end_pos_list)-1]
                else:
                    previous_sent_index = post_sent_index_list[len(post_sent_index_list)-1]
                    previous_end_index = post_end_pos_list[len(post_end_pos_list)-1]

                previous_tokens = tokens_list[previous_sent_index]
                distance = 0
                if len(sentences)-1 >= previous_sent_index+1:
                    for j in range(previous_sent_index+1,len(sentences)):
                        ss_tokens = tokens_list[j]
                        distance = distance + len(ss_tokens)
                distance = distance + len(previous_tokens) - previous_end_index - 1                    
                post_mention_distance_sequence.append(distance)
            else:
                post_mention_distance_sequence.append(0)

    return original_post_mention_sequence, converted_post_mention_sequence, post_mention_cluster_id_sequence, post_mention_distance_sequence, post_sent_index_list, post_end_pos_list, post_start_pos_list

def match_mention_to_position(original_text, start_token_index, token_index_list, token_index_to_start_pos, start_pos_to_token):
    original_text_len = 0
    for c in original_text:
        if c != ' ':
            original_text_len = original_text_len + 1
    total_len = 0
    found_index = -1
    ot = ''
    for i in range(start_token_index, len(token_index_list)):
        token_index = token_index_list[i]
        start_pos = token_index_to_start_pos[token_index]
        token = start_pos_to_token[start_pos]
        token_text = token.text
        total_len = total_len + len(token_text)        
        if total_len == original_text_len:
            ot = ot + ' ' + token_text.lower()
            found_index = i
            break
        else:
            ot = ot + ' ' + token_text.lower()
    ot = ot.strip()
    
    if found_index == -1:
        print("error")
        print(original_text)
        print(start_token_index)
        print(token_index_list)
        token_index = token_index_list[start_token_index]
        start_pos = token_index_to_start_pos[token_index]
        token = start_pos_to_token[start_pos]
        print(token.text)
    return ot

def get_sent_num_to_mentions(start_pos_to_sentence_num, start_pos_to_token, start_pos_to_mention_conversion, highlight_color_to_cluster_id, shading_color_to_cluster_id, sentence_num_to_token_index_list, token_index_to_start_pos):
    sent_num_to_mentions = {}
    key_to_cluster_id = {}
    key_to_original_mention_text = {}
    key_to_converted_mention_text = {}
    first_person_pronouns = {'i','me','my','myself','mine'}
    cluster_id_to_color = get_cluster_id_to_color(highlight_color_to_cluster_id, shading_color_to_cluster_id)
    
    for sent_num in sentence_num_to_token_index_list:
        token_index_list = sentence_num_to_token_index_list[sent_num]
        mentions=[0]*len(token_index_list)
        for i in range(len(token_index_list)):
            token_index = token_index_list[i]
            start_pos = token_index_to_start_pos[token_index]
            if start_pos in start_pos_to_mention_conversion:
                mc = start_pos_to_mention_conversion[start_pos]
                ot = mc._original_text
                ct = mc._converted_text
                ci = mc._cluster_id
                color = cluster_id_to_color[ci]
                ot = match_mention_to_position(ot, i, token_index_list, token_index_to_start_pos, start_pos_to_token)
                ot_parts = ot.split()
                ct = correct_string(ct)
                mention=0
                if i+len(ot_parts)-1 == 0:
                    mention=-1
                else:
                    mention=i+len(ot_parts)-1
                mentions[i]=mention
                key = str(sent_num)+":"+str(i)+":"+str(mention)
                if ot.lower() in first_person_pronouns:
                    key_to_cluster_id[key]=-1
                elif color=='yellow':
                    key_to_cluster_id[key]=-2
                else:
                    key_to_cluster_id[key]=ci
                    
                key_to_original_mention_text[key]=ot.lower()
                key_to_converted_mention_text[key]=ct.lower()
        sent_num_to_mentions[sent_num]=mentions
        
    return sent_num_to_mentions,key_to_cluster_id,key_to_original_mention_text,key_to_converted_mention_text

def process_data_gold(output_file, document_text, start_pos_to_mention_conversion, highlight_color_to_cluster_id, shading_color_to_cluster_id, sentences, start_pos_to_token, start_pos_to_sentence_num, sentence_num_to_sent_start_num, start_pos_to_token_index, token_index_to_start_pos, focus_mention_string_set, sentence_num_to_token_index_list, tokens_list, start_pos_to_verb_conversion, subject_pos_list, object_pos_list):    
    f = open(output_file, "w+", encoding="utf-8")

    valid_cluster_ids,gold_cluster_id_to_mentions,auto_cluster_id_to_mentions,cluster_id_to_mention_tags=write_cluster_info(f, start_pos_to_mention_conversion, highlight_color_to_cluster_id, shading_color_to_cluster_id, focus_mention_string_set, start_pos_to_token)
     
    sent_num_to_mentions,key_to_cluster_id,key_to_original_mention_text,key_to_converted_mention_text = get_sent_num_to_mentions(start_pos_to_sentence_num, start_pos_to_token, start_pos_to_mention_conversion, highlight_color_to_cluster_id, shading_color_to_cluster_id, sentence_num_to_token_index_list, token_index_to_start_pos)
   
    f.write('<document_text>')
    f.write('\n')
    f.write(document_text)
    f.write('\n')
    f.write('<document_text>')
    f.write('\n')
    ann_mention = 0
    for sent_num in sorted(sent_num_to_mentions.keys()):
        mentions = sent_num_to_mentions[sent_num]
        tt_index_list = sentence_num_to_token_index_list[sent_num]
        for i in range(len(mentions)):
            token_index = tt_index_list[i]
            sp = token_index_to_start_pos[token_index]
            mention = mentions[i]
            if mention != 0:
                key = str(sent_num) + ":" + str(i) + ":" + str(mention)
                cluster_id = key_to_cluster_id[key]
                original_mention_text = key_to_original_mention_text[key]
                original_mention_text_parts = original_mention_text.strip().split()
                converted_mention_text = key_to_converted_mention_text[key]
                converted_mention_text_parts = converted_mention_text.strip().split()
                
                if cluster_id not in valid_cluster_ids:
                    continue
                
                is_subject = False
                is_object = False
                eep = mention
                if mention == -1:
                    eep = 0
                for j in range(i, eep+1):
                    tti = tt_index_list[j]
                    spp = token_index_to_start_pos[tti]
                    if spp in subject_pos_list:
                        is_subject = True
                    elif spp in object_pos_list:
                        is_object = True
                        
                f.write('<identified_mention>: 1')
                f.write('\t')                
                f.write("<annotated_mention>: 1")
                f.write("\t")
                f.write('<start_pos>: ')
                f.write(str(sp))
                f.write('\t')        
                f.write("<identified_original_focus_mention>: ")
                f.write(original_mention_text)
                f.write("\t")
                f.write("<annotated_original_focus_mention>: ")
                f.write(original_mention_text)
                f.write('\t')
                f.write("<converted_focus_mention>: ")
                f.write(converted_mention_text)
                f.write("\t")
                f.write("<cluster_id>: ")
                f.write(str(cluster_id))
                f.write("\t")
                f.write("<sentence_num>: ")
                f.write(str(sent_num))
                f.write("\t")
                f.write("<index_in_sentence>: ")
                f.write(str(i))
                f.write('\t')
                f.write("<is_subject>: ")
                if is_subject:
                    f.write("1")
                else:
                    f.write("0")
                f.write("\t")
                f.write("<is_object>: ")
                if is_object:
                    f.write("1")
                else:
                    f.write("0")
                f.write("\n")
                
                ann_mention = ann_mention + 1
                if mention == -1:
                    mention = 0
                from_sent_num, from_token_index, pre_padding = get_from_info(sent_num, i, sentences, tokens_list)
                to_sent_num, to_token_index, post_padding = get_to_info(sent_num, mention, sentences, tokens_list)
                
                related_mentions, original_related_mention_text_list, converted_related_mention_text_list, related_mention_cluster_ids, raw_sequence, postags = fourth_round_process(pre_padding, post_padding, from_sent_num, to_sent_num, from_token_index, to_token_index, sentences, sent_num_to_mentions, key_to_cluster_id, key_to_original_mention_text, key_to_converted_mention_text, len(original_mention_text_parts), len(converted_mention_text_parts), tokens_list, sentence_num_to_token_index_list, token_index_to_start_pos, start_pos_to_mention_conversion, start_pos_to_verb_conversion)  
                
                write_sequence_info(f, related_mentions, original_related_mention_text_list, converted_related_mention_text_list, related_mention_cluster_ids, raw_sequence, postags)
                 
                for j in range(from_sent_num, to_sent_num+1):
                    tokens = tokens_list[j]
                    token_index_list = sentence_num_to_token_index_list[j]            
                    sentence_text = ""
                    for h in range(len(tokens)):
                        token = tokens[h]
                        to_add_text = token.text.lower()
                        temp_index = token_index_list[h]
                        temp_start_pos = token_index_to_start_pos[temp_index]
                        if temp_start_pos in start_pos_to_verb_conversion:
                            verb_conversion = start_pos_to_verb_conversion[temp_start_pos]
                            temp_original_text = verb_conversion._original_text.lower()
                            if temp_original_text != token.text.lower():
                                print('error - double gold verb')
                            temp_converted_text = verb_conversion._converted_text
                            to_add_text = temp_converted_text.lower()
                        
                        sentence_text = sentence_text + " " + to_add_text
                    sentence_text = sentence_text.strip()
                    
                    f.write("<sentence_num>: ")
                    f.write(str(j))
                    f.write("\t")
                    f.write("<sentence_text>: ")
                    f.write(sentence_text)
                    f.write("\n")
                f.write("<start_token_index_in_sentence>: ")
                f.write(str(from_token_index))
                f.write("\t")
                f.write("<end_token_index_in_sentence>: ")
                f.write(str(to_token_index))
                f.write("\n")
                
                original_pre_mention_sequence, converted_pre_mention_sequence, pre_mention_cluster_id_sequence, pre_mention_distance_sequence, pre_sent_index_list, pre_end_pos_list, pre_start_pos_list = get_from_mention_info(sent_num, i, sentences, tokens_list, sent_num_to_mentions, key_to_cluster_id, key_to_original_mention_text, key_to_converted_mention_text)
                
                original_post_mention_sequence, converted_post_mention_sequence, post_mention_cluster_id_sequence, post_mention_distance_sequence, post_sent_index_list, post_end_pos_list, post_start_pos_list = get_to_mention_info(sent_num, i, sentences, tokens_list, sent_num_to_mentions, key_to_cluster_id, key_to_original_mention_text, key_to_converted_mention_text, pre_sent_index_list, pre_end_pos_list)   
                                          
                f.write("<original_pre_mention_sequence>: ")
                for X_mention in original_pre_mention_sequence:
                    f.write(X_mention)
                    f.write("\t")
                f.write("\n")
                
                f.write("<converted_pre_mention_sequence>: ")
                for X_mention in converted_pre_mention_sequence:
                    f.write(X_mention)
                    f.write("\t")
                f.write("\n")
                
                pre_padding_count = 0
                for j in range(len(converted_pre_mention_sequence)):
                    mention = converted_pre_mention_sequence[j]
                    if mention == '<pad>':
                        pre_padding_count = pre_padding_count + 1
                        continue
                    sent_index = pre_sent_index_list[j-pre_padding_count]
                    this_mentions = sent_num_to_mentions[sent_index]
                    start_pos = pre_start_pos_list[j-pre_padding_count]
                    tokens = tokens_list[sent_index]
                    token_index_list = sentence_num_to_token_index_list[sent_index]
                    offset = 0
                    sentence_text = ""
                    h = 0
                    while h < len(tokens):
                        token = tokens[h]
                        to_add_text = token.text.lower()
                        temp_index = token_index_list[h]
                        temp_start_pos = token_index_to_start_pos[temp_index]
                        if temp_start_pos in start_pos_to_verb_conversion:
                            verb_conversion = start_pos_to_verb_conversion[temp_start_pos]
                            temp_original_text = verb_conversion._original_text.lower()
                            if temp_original_text != token.text.lower():
                                print('error - double gold verb')
                            temp_converted_text = verb_conversion._converted_text
                            to_add_text = temp_converted_text.lower()
                        this_mention = this_mentions[h]
                        if this_mention != 0:
                            this_key = str(sent_index) + ":" + str(h) + ":" + str(this_mention)                        
                            tt_oo_mm = key_to_original_mention_text[this_key].lower()
                            tt_oo_mm_parts = tt_oo_mm.split()
                            tt_cc_mm = key_to_converted_mention_text[this_key].lower()
                            tt_cc_mm_parts = tt_cc_mm.split()   
                            tt_oo_mm_len = len(tt_oo_mm_parts)
                            for tt_cc_mm_part in tt_cc_mm_parts:
                                sentence_text = sentence_text + " " + tt_cc_mm_part                         
                            if h < start_pos:
                                offset = offset + len(tt_cc_mm_parts) - len(tt_oo_mm_parts)
                            h = h + tt_oo_mm_len
                        else:
                            h = h + 1
                            sentence_text = sentence_text + " " + to_add_text
                
                    sentence_text = sentence_text.strip()
                    start_pos = start_pos + offset
                    
                    f.write("<pre_mention_text>: ")
                    f.write(mention)
                    f.write("\t")
                    f.write("<pre_mention_index_in_sentence>: ")
                    f.write(str(start_pos))
                    f.write("\t")
                    f.write("<pre_mention_in_sentence>: ")                   
                    f.write(sentence_text)      
                    f.write("\t")
                    f.write("<pre_mention_sentence_num>: ")
                    f.write(str(sent_index))
                    f.write("\n")
                    
                f.write("<pre_mention_cluster_id_sequence>: ")
                for X_cluster_id in pre_mention_cluster_id_sequence:
                    f.write(str(X_cluster_id))
                    f.write("\t")                  
                f.write("\n")
                
                f.write("<pre_mention_distance_sequence>: ")
                for X_mention_distance in pre_mention_distance_sequence:
                    f.write(str(X_mention_distance))
                    f.write("\t")                  
                f.write("\n")
                
                f.write("<original_post_mention_sequence>: ")
                for X_mention in original_post_mention_sequence:
                    f.write(X_mention)
                    f.write("\t")
                f.write("\n")
                
                f.write("<converted_post_mention_sequence>: ")
                for X_mention in converted_post_mention_sequence:
                    f.write(X_mention)
                    f.write("\t")
                f.write("\n")
                
                for j in range(len(converted_post_mention_sequence)):
                    mention = converted_post_mention_sequence[j]
                    if mention == '<pad>':
                        continue
                    sent_index = post_sent_index_list[j]
                    this_mentions = sent_num_to_mentions[sent_index]
                    start_pos = post_start_pos_list[j]
                    tokens = tokens_list[sent_index]
                    token_index_list = sentence_num_to_token_index_list[sent_index]
                    offset = 0
                    sentence_text = ""
                    h = 0
                    while h < len(tokens):
                        token = tokens[h]
                        to_add_text = token.text.lower()
                        temp_index = token_index_list[h]
                        temp_start_pos = token_index_to_start_pos[temp_index]
                        if temp_start_pos in start_pos_to_verb_conversion:
                            verb_conversion = start_pos_to_verb_conversion[temp_start_pos]
                            temp_original_text = verb_conversion._original_text.lower()
                            if temp_original_text != token.text.lower():
                                print('error - double gold verb')
                            temp_converted_text = verb_conversion._converted_text
                            to_add_text = temp_converted_text.lower()
                        this_mention = this_mentions[h]
                         
                        if this_mention != 0:
                            this_key = str(sent_index) + ":" + str(h) + ":" + str(this_mention)                        
                            tt_oo_mm = key_to_original_mention_text[this_key].lower()
                            tt_oo_mm_parts = tt_oo_mm.split()
                            tt_cc_mm = key_to_converted_mention_text[this_key].lower()
                            tt_cc_mm_parts = tt_cc_mm.split()   
                            tt_oo_mm_len = len(tt_oo_mm_parts)
                            for tt_cc_mm_part in tt_cc_mm_parts:
                                sentence_text = sentence_text + " " + tt_cc_mm_part                         
                            if h < start_pos:
                                offset = offset + len(tt_cc_mm_parts) - len(tt_oo_mm_parts)
                            h = h + tt_oo_mm_len
                        else:
                            h = h + 1
                            sentence_text = sentence_text + " " + to_add_text
                        
                    sentence_text = sentence_text.strip()
                    start_pos = start_pos + offset
                    f.write("<post_mention_text>: ")
                    f.write(mention)
                    f.write("\t")
                    f.write("<post_mention_index_in_sentence>: ")
                    f.write(str(start_pos))
                    f.write("\t")
                    f.write("<post_mention_in_sentence>: ")                   
                    f.write(sentence_text)      
                    f.write("\t")
                    f.write("<post_mention_sentence_num>: ")
                    f.write(str(sent_index))
                    f.write("\n")
                    
                f.write("<post_mention_cluster_id_sequence>: ")
                for X_cluster_id in post_mention_cluster_id_sequence:
                    f.write(str(X_cluster_id))
                    f.write("\t")                  
                f.write("\n")
                
                f.write("<post_mention_distance_sequence>: ")
                for X_mention_distance in post_mention_distance_sequence:
                    f.write(str(X_mention_distance))
                    f.write("\t")                  
                f.write("\n")  
    print(ann_mention)
    
def identify_gender(start_pos, end_pos, cfr_document, cfr_clusters, cfr_index_to_token_index, token_index_to_start_pos, start_pos_to_token):
    male_pronouns = {'he','him','his','himself'}
    female_pronouns = {'she','her','herself','hers'}
    is_male = False
    is_female = False
    name_list = []
    for cluster in cfr_clusters:
        proper_name_list = []
        found_mention = False
        found_male = False
        found_female = False
        for span in cluster:
            span_start = span[0]
            span_end = span[1]
            
            real_start = -1
            temp_mention = ''
            for i in range(span_start, span_end+1):
                temp = cfr_document[i]
                if temp != ' ' and temp != '\t':
                    temp_mention = temp_mention + temp + ' '
                    if real_start == -1:
                        real_start = i
            temp_mention = temp_mention.strip()
            
            real_end = -1
            for i in range(span_end, span_start-1, -1):
                temp = cfr_document[i]
                if temp != ' ' and temp != '\t':
                    if real_end == -1:
                        real_end = i
                        break
                    
            if real_start == -1 or real_end == -1:
                print("There is an error - cofounding coreference chain 2")
                continue
            
            start_token_index = cfr_index_to_token_index[real_start]
            i_start_pos = token_index_to_start_pos[start_token_index]
            end_token_index = cfr_index_to_token_index[real_end]
            end_start_pos = token_index_to_start_pos[end_token_index]
            end_token = start_pos_to_token[end_start_pos]
            i_end_pos = end_start_pos + len(end_token.text)-1
            if start_pos == i_start_pos and end_pos == i_end_pos:  
                found_mention = True
                
            if temp_mention.lower() in female_pronouns:
                found_female = True
            elif temp_mention.lower() in male_pronouns:
                found_male = True
            
            is_proper_name = True
            for i in range(start_token_index,end_token_index+1):
                temp_start = token_index_to_start_pos[i]
                temp_token_pos_tag = start_pos_to_token[temp_start].tag_
                if temp_token_pos_tag != 'NNP':
                    is_proper_name = False
                    break
                
            if is_proper_name:
                proper_name_list.append(temp_mention)
    
        if found_mention:
            if found_female:
                is_female = True
            elif found_male:
                is_male = True
            name_list = proper_name_list.copy()
            break
        
    if is_female:
        return [1,name_list]
    elif is_male:
        return [2, name_list]
    else:
        return [0, name_list]

def generate_mention_string_set_from_relational_nouns_gold(highlight_color_to_cluster_id, shading_color_to_cluster_id, start_pos_to_mention_conversion, start_pos_to_token, cfr_document, cfr_clusters, cfr_index_to_token_index, token_index_to_start_pos, focus_mention_gender):
    mention_string_set = []
    relation_noun_dic = {}
    f = open('seed.tsv')
    lines = f.readlines()
    for line in lines:
        fields = line.split('\t')
        relation = ''
        converse_relations = []
        for i in range(len(fields)):
            if i == 0:
                relation = fields[i]
            else:
                ttt = fields[i]
                if ttt.endswith('\n'):
                    ttt = ttt[0:len(ttt)-1]
                converse_relations.append(ttt)
        relation_noun_dic[relation] = converse_relations
     
    cluster_id_to_color = get_cluster_id_to_color(highlight_color_to_cluster_id, shading_color_to_cluster_id)
    sorted_start_pos_list = sorted(start_pos_to_token.keys())
    for i in range(len(sorted_start_pos_list)):
        sorted_start_pos = sorted_start_pos_list[i]
        if sorted_start_pos in start_pos_to_mention_conversion:
            mc = start_pos_to_mention_conversion[sorted_start_pos]
            ot = mc._original_text
            ci = mc._cluster_id
            m_color = cluster_id_to_color[ci]
            entity_mention = ot.lower()
            if entity_mention == 'my':
                next_long_mention = ''
                next_long_mention_start_pos = -1
                next_long_mention_end_pos = -1
                if i < len(sorted_start_pos_list)-5:
                    one_mention_after_sp = sorted_start_pos_list[i+1]
                    next_long_mention_start_pos = one_mention_after_sp
                    one_mention_after = start_pos_to_token[one_mention_after_sp].text.lower()
                    two_mention_after_sp = sorted_start_pos_list[i+2]
                    two_mention_after = start_pos_to_token[two_mention_after_sp].text.lower()
                    three_mention_after_sp = sorted_start_pos_list[i+3]
                    three_mention_after = start_pos_to_token[three_mention_after_sp].text.lower()
                    four_mention_after_sp = sorted_start_pos_list[i+4]
                    four_mention_after = start_pos_to_token[four_mention_after_sp].text.lower()
                    five_mention_after_sp = sorted_start_pos_list[i+5]
                    five_mention_after = start_pos_to_token[five_mention_after_sp].text.lower()
                    next_long_mention = one_mention_after+two_mention_after+three_mention_after+four_mention_after+five_mention_after
                    next_long_mention_end_pos = five_mention_after_sp + len(five_mention_after)-1
                if i < len(sorted_start_pos_list)-1:
                    next_mention_start_pos = -1
                    sorted_end_pos = -1
                    next_mention = ''
                    if next_long_mention.find('-') != -1 and next_long_mention in relation_noun_dic:
                        next_mention = next_long_mention
                        next_mention_start_pos = next_long_mention_start_pos
                        sorted_end_pos = next_long_mention_end_pos
                    else:
                        next_mention_start_pos = sorted_start_pos_list[i+1]
                        next_mention = start_pos_to_token[next_mention_start_pos].text.lower()
                        sorted_end_pos = next_mention_start_pos + len(next_mention) - 1
                    if next_mention in relation_noun_dic:
                        [i_gender, name_list] = identify_gender(sorted_start_pos, sorted_end_pos, cfr_document, cfr_clusters, cfr_index_to_token_index, token_index_to_start_pos, start_pos_to_token)
                        mention_string_list = []
                        for name in name_list:
                            if name.endswith('s'):
                                ms = name+' \' '
                            else:
                                ms = name+' \'s '
                            mention_string_list.append(ms)
                        mention_string = ''
                        relation_list = relation_noun_dic[next_mention]
                        if relation_list[0] == 'm':
                            mention_string = 'his '                            
                        elif relation_list[0] == 'f':
                            mention_string = 'her '
                        elif relation_list[0] == 'a':
                            if focus_mention_gender == 'm':
                                mention_string = 'her '
                            else:
                                mention_string = 'his '
                        elif relation_list[0] == 'n':                          
                            if i_gender == 1:
                                mention_string = 'her '
                            elif i_gender == 2:
                                mention_string = 'his '
                            elif len(name_list) == 0:
                                mention_string = 'his '
                        else:
                            print('Wrong annotation - gender')
                    
                        mention_string_list.append(mention_string)
                        
                        if len(relation_list) == 3:
                            if focus_mention_gender == 'm':
                                for mx in mention_string_list:
                                    fmx = mx + relation_list[1]
                                    fmx = correct_string(fmx)
                                    if fmx not in mention_string_set:
                                        mention_string_set.append(fmx)
                            else:
                                for mx in mention_string_list:
                                    fmx = mx + relation_list[2]
                                    fmx = correct_string(fmx)
                                    if fmx not in mention_string_set:
                                        mention_string_set.append(fmx)
                        else:
                            for mx in mention_string_list:
                                fmx = mx + relation_list[1]
                                fmx = correct_string(fmx)
                                if fmx not in mention_string_set:
                                    mention_string_set.append(fmx)
    f.close()   
    return mention_string_set

def read_mention_string(focus_mention_string_path):
    sex = ''
    focus_mention_string_set = []
    f = open(focus_mention_string_path)
    lines = f.readlines()
    for i in range(len(lines)):
        line = lines[i]
        mentions = line.split('\t')
        for j in range(len(mentions)):
            if i == 0 and j == 0:
                sex = mentions[j]
            elif i == 0:
                focus_mention_string_set.append(mentions[j])
    
    f.close()    
    return focus_mention_string_set,sex
    
def main():
    parser = argparse.ArgumentParser('Preprocess')

    parser.add_argument('--input_dir', 
                    type=str, 
                    default='pov_data/', 
                    help='input file.')
    parser.add_argument('--output_dir',
                    type=str,
                    default='output_data_gold/',
                    help='output file.')
    parser.add_argument('--focus_mention_string_dir',
                    type=str,
                    default='focus_mention_string/',
                    help='Path of focus mention string set.')

    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()
    input_dir = FLAGS.input_dir
    output_dir = FLAGS.output_dir
    focus_mention_string_dir = FLAGS.focus_mention_string_dir
#    input_files=["Additional_conll_dev.docx", "Additional_conll_test.docx"]
#    focus_mention_string_files=["Additional_conll_dev.txt", "Additional_conll_test.txt"]
#    output_files=["additional_1.txt", "additional_2.txt"]
    
    input_files=["A_Walk_in_the_Woods-with_annotations.docx","Another_Bullshit_Night_in_Suck_City-with_annotations.docx","Dispatches-with_annotations.docx","How_to_change_your_mind-with_annotations.docx","I'm_a_doctor_with_annotations.docx","Mother_Night-with_annotations.docx","My_Dad_Tried_to_Kill_Me_with_an_Alligator-with_annotations.docx","Nobody_Here_But-with_annotations.docx","Notes_from_No_Man's_Land-with_annotations.docx","Selkie_Stories_Are_for_Losers_with_annotations.docx","Sweetness-with_annotations_V2.docx","The_37_with_annotations_V2.docx","The_Faraway_Nearby-with_annotations.docx","The_Handmaid's_Tale-with_annotations.docx","The_Nausea_with_annotations.docx","The_Water_That_Falls_on_You_from_Nowhere_with_annotations.docx","There's_no_recipe_for_growing_up-with_annotations.docx","Understand_with_annotations.docx", "Additional_conll_dev.docx", "Additional_conll_test.docx"]
    focus_mention_string_files=["A_Walk_in_the_Woods-with_annotations.txt","Another_Bullshit_Night_in_Suck_City-with_annotations.txt","Dispatches-with_annotations.txt","How_to_change_your_mind-with_annotations.txt","I'm_a_doctor_with_annotations.txt","Mother_Night-with_annotations.txt","My_Dad_Tried_to_Kill_Me_with_an_Alligator-with_annotations.txt","Nobody_Here_But-with_annotations.txt","Notes_from_No_Man's_Land-with_annotations.txt","Selkie_Stories_Are_for_Losers_with_annotations.txt","Sweetness-with_annotations_V2.txt","The_37_with_annotations_V2.txt","The_Faraway_Nearby-with_annotations.txt","The_Handmaid's_Tale-with_annotations.txt","The_Nausea_with_annotations.txt","The_Water_That_Falls_on_You_from_Nowhere_with_annotations.txt","There's_no_recipe_for_growing_up-with_annotations.txt","Understand_with_annotations.txt", "Additional_conll_dev.txt", "Additional_conll_test.txt"]
    output_files=["walk.txt", "night.txt", "dispatches.txt", "mind.txt", "doctor.txt", "mother.txt", "dad.txt", "nobody.txt", "noman.txt", "selkie.txt", "sweetness.txt", "37.txt", "faraway.txt", "handmaid.txt", "nausea.txt", "water.txt", "recipe.txt", "understand.txt", "additional_1.txt", "additional_2.txt"]
    
    for i in range(20):
        part_input_file = input_files[i]
        part_output_file = output_files[i]
        input_file = input_dir + part_input_file
        output_file = output_dir + part_output_file
        focus_mention_string_path = focus_mention_string_dir + focus_mention_string_files[i]
        updated_text, start_pos_to_verb_conversion, start_pos_to_mention_conversion, highlight_color_to_cluster_id, shading_color_to_cluster_id = convert_word_doc_to_text(input_file)
        sentences, start_pos_to_token, start_pos_to_sentence_num, sentence_num_to_sent_start_num, cfr_document, cfr_clusters, cfr_index_to_token_index, start_pos_to_token_index, token_index_to_start_pos, sentence_num_to_token_index_list, tokens_list = preprocess(updated_text)
    
        subject_pos_list, object_pos_list = change_verb_conjugation(updated_text, token_index_to_start_pos)
    
        temp_set,sex = read_mention_string(focus_mention_string_path)
        focus_mention_string_set = generate_mention_string_set_from_relational_nouns_gold(highlight_color_to_cluster_id, shading_color_to_cluster_id, start_pos_to_mention_conversion, start_pos_to_token, cfr_document, cfr_clusters, cfr_index_to_token_index, token_index_to_start_pos, sex)
        if sex == 'm':
            focus_mention_string_set.extend(['him', 'his', 'he', 'himself','he himself'])
        else:
            focus_mention_string_set.extend(['she','her','herself','she herself'])
   
        focus_mention_string_set.extend(temp_set)
    
        print("processing file: ")
        print(input_file)
        process_data_gold(output_file, updated_text, start_pos_to_mention_conversion, highlight_color_to_cluster_id, shading_color_to_cluster_id, sentences, start_pos_to_token, start_pos_to_sentence_num, sentence_num_to_sent_start_num, start_pos_to_token_index, token_index_to_start_pos, focus_mention_string_set, sentence_num_to_token_index_list, tokens_list, start_pos_to_verb_conversion, subject_pos_list, object_pos_list)
    
if __name__ == "__main__":
    main()