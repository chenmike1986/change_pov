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

def extract_second_person_entity(document_text, start_pos_to_verb_conversion, start_pos_to_mention_conversion, highlight_color_to_cluster_id, shading_color_to_cluster_id, sentences, start_pos_to_token, start_pos_to_sentence_num):
    second_person_mentions = ['you', 'your', 'yourself', 'yours']
    plural_first_person_mentions = ['we','us','our','ourselves','ours']
    cluster_id_to_color = get_cluster_id_to_color(highlight_color_to_cluster_id, shading_color_to_cluster_id)
    left_quote_list, right_quote_list = identify_quotes(document_text)
       
    start_pos_to_focus_entity_mention = {}
    sorted_start_pos_list = sorted(start_pos_to_token.keys())
    for sorted_start_pos in sorted_start_pos_list:
        token = start_pos_to_token[sorted_start_pos]
        original_text = token.text
        original_text_lower = original_text.lower()
        sorted_end_pos = sorted_start_pos + len(original_text) - 1
        if original_text_lower in second_person_mentions:
            is_valid = check_mention_in_quotation_and_dialogue(sorted_start_pos, sorted_end_pos, left_quote_list, right_quote_list)
            if is_valid:
                start_pos_to_focus_entity_mention[sorted_start_pos] = token
                   
    # For testing purpose
    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    count5 = 0
    count6 = 0
    
#    cluster_id_for_focus_entity = -1
#    cluster_id_for_plural_first_person = -2
#    for start_pos in start_pos_to_mention_conversion:
#        mention_conversion = start_pos_to_mention_conversion[start_pos]
#        original_text = mention_conversion._original_text.lower()
#        cluster_id = mention_conversion._cluster_id
#        color = cluster_id_to_color[cluster_id]
#        if (color == 'yellow') and (original_text in second_person_mentions):
#            cluster_id_for_focus_entity = cluster_id
#            break
#        
#    for start_pos in start_pos_to_mention_conversion:
#        mention_conversion = start_pos_to_mention_conversion[start_pos]
#        cluster_id = mention_conversion._cluster_id
#        color = cluster_id_to_color[cluster_id]
#        if (color == 'yellow') and (cluster_id != cluster_id_for_focus_entity):
#            cluster_id_for_plural_first_person = cluster_id
#            break
        
    for start_pos in start_pos_to_mention_conversion:
        mention_conversion = start_pos_to_mention_conversion[start_pos]
        original_text = mention_conversion._original_text.lower()
        converted_text = mention_conversion._converted_text.lower()
#        cluster_id = mention_conversion._cluster_id
        if original_text in second_person_mentions:
            if start_pos not in start_pos_to_focus_entity_mention:
                count1 = count1 + 1
                sentence_num = start_pos_to_sentence_num[start_pos]
                sentence = sentences[sentence_num]
                print("Annotated focus entity mention was not identified: ")
                print("Annotated original text: ")
                print(original_text)
                print("Annotated converted text: ")
                print(converted_text)
                print("Sentence: ")
                print(sentence)
            else:
                count2 = count2 + 1
                i_token = start_pos_to_focus_entity_mention[start_pos]
                if i_token.text.lower() != original_text:
                    count3 = count3 + 1
                    sentence_num = start_pos_to_sentence_num[start_pos]
                    sentence = sentences[sentence_num]
                    print("Not matched focus entity mention: ")
                    print("Annotated original mention: ")
                    print(original_text)
                    print("Identified original mention: ")
                    print(i_token.text)
                    print("Sentence: ")
                    print(sentence)
    
    for start_pos in start_pos_to_focus_entity_mention:
        mention = start_pos_to_focus_entity_mention[start_pos].text.lower()
        if start_pos not in start_pos_to_mention_conversion:
            count4 = count4 + 1
            sentence_num = start_pos_to_sentence_num[start_pos]
            sentence = sentences[sentence_num]
            print("Identified focus entity mention was not annotated: ")
            print("Identified original mention: ")
            print(mention)
            print("Sentence: ")
            print(sentence)
        else:
            count5 = count5 + 1
            mention_conversion = start_pos_to_mention_conversion[start_pos]
            original_text = mention_conversion._original_text.lower()
            if mention != original_text:
                count6 = count6 + 1
                
    print("Annotated focus entity mention conversions that are not identified: ")            
    print(count1)
    print("Annotated focus entity mention conversions that are identified: ")
    print(count2)
    print("Annotated focus entity mention conversions that are identified but incorrectly identified: ")
    print(count3)
    print("Identified focus entity mention conversions that are not annotated: ")
    print(count4)
    print("Identified focus entity mention conversions that are annotated: ")
    print(count5)
    print("Identified focus entity mention conversions that are annotated but incorrectly identified: ")
    print(count6)
    
    return start_pos_to_focus_entity_mention

def extract_confounding_entities(document_text, start_pos_to_verb_conversion, start_pos_to_mention_conversion, highlight_color_to_cluster_id, shading_color_to_cluster_id, sentences, start_pos_to_token, start_pos_to_sentence_num, sentence_num_to_sent_start_num, cfr_document, cfr_clusters, cfr_index_to_token_index, token_index_to_start_pos, start_pos_to_focus_entity_mention,sex):
    third_person_pronouns = ['he', 'him', 'his', 'himself', 'she', 'her', 'herself', 'hers']
    first_person_pronouns= ['i', 'me', 'my', 'myself', 'mine', 'we', 'us', 'our', 'ourselves', 'ours']
    second_person_pronouns = ['you', 'your', 'yourself', 'yours']
    third_entity_pronouns = ['it', 'its', 'itself']
    cluster_id_to_color = get_cluster_id_to_color(highlight_color_to_cluster_id, shading_color_to_cluster_id)
    left_quote_list, right_quote_list = identify_quotes(document_text)
    
    cluster_id_to_mention_set = {}
    for cluster in cfr_clusters:
        start_pos_to_mention = {}
        is_person_entity = False
        contains_third_person_mention = False
        third_person_mention_count = 0
        contains_special_mention = False
        special_mention_count = 0
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
             
            real_end = -1
            for i in range(span_end, span_start-1, -1):
                temp = cfr_document[i]
                if temp != ' ' and temp != '\t':
                    if real_end == -1:
                        real_end = i
                        break
                    
            if real_start == -1 or real_end == -1:
                print("There is an error - cofounding coreference chain 1")
                continue
            
            temp_mention = temp_mention.strip()
            if temp_mention != ' ' and temp_mention != '\t' and temp_mention != '':
                if temp_mention.lower() in third_person_pronouns:
                    contains_third_person_mention = True
                    third_person_mention_count = third_person_mention_count + 1
                if temp_mention.lower().find('mike') != -1:
                    contains_special_mention = True
                    special_mention_count = special_mention_count + 1
                    
                start_token_index = cfr_index_to_token_index[real_start]
                start_pos = token_index_to_start_pos[start_token_index]
                
                end_token_index = cfr_index_to_token_index[real_end]
                end_start_pos = token_index_to_start_pos[end_token_index]
                end_token = start_pos_to_token[end_start_pos]
                end_pos = end_start_pos + len(end_token.text) - 1
                token_list = []
                all_valid_person = True
                for j in range(start_token_index, end_token_index+1):
                    temp_pos = token_index_to_start_pos[j]
                    temp_token = start_pos_to_token[temp_pos]                   
                    temp_ner = temp_token.ent_type_
                    if temp_ner != 'PERSON':
                        all_valid_person = False
                    token_list.append([temp_pos,temp_token])

                if all_valid_person:
                    is_person_entity = True                    
                is_valid = check_mention_in_quotation_and_dialogue(start_pos, end_pos, left_quote_list, right_quote_list)
                if is_valid:
                    start_pos_to_mention[start_pos] = token_list
        
        not_valid = False
        for sp in start_pos_to_mention:
            tl = start_pos_to_mention[sp]
            for tt in tl:
                if tt[1].text == ' ' or tt[1].text == '\t' or tt[1].text == '':
                    continue
                elif (tt[1].text.lower() in second_person_pronouns) or (tt[1].text.lower() in third_entity_pronouns):
                    not_valid = True
                    break
            if not_valid:
                break

        total_mention_count = len(start_pos_to_mention)
        percent = 0
        percent1 = 0
        if total_mention_count > 0:
            percent = 1.0 * third_person_mention_count / total_mention_count 
            percent1 = 1.0 * special_mention_count / total_mention_count           
        if not_valid and percent < 0.5 and percent1 < 0.5:
            continue
        
        if contains_third_person_mention or is_person_entity or contains_special_mention:
            cluster_id_to_mention_set[len(cluster_id_to_mention_set)] = start_pos_to_mention
        
    start_pos_to_range = {}
    start_pos_to_cluster_id = {}
    cluster_id_to_string_set = {}
    narcissist_cluster_id_list = []
    for cluster_id in cluster_id_to_mention_set:
        mention_set = cluster_id_to_mention_set[cluster_id]
        mms = []
        for ss in mention_set:
            pos_list_and_tokens = mention_set[ss]
            last_pos_and_token = pos_list_and_tokens[len(pos_list_and_tokens)-1]
            last_pos = last_pos_and_token[0]
            last_token = last_pos_and_token[1]
            ee = last_pos + len(last_token.text) - 1
            mm = ''
            for i in range(len(pos_list_and_tokens)):
                pos_and_token = pos_list_and_tokens[i]
                pp = pos_and_token[0]
                tt = pos_and_token[1]
                mm = mm + tt.text.lower() + ' '
                if pp not in start_pos_to_range:
                    start_pos_to_range[pp] = [ss, ee]
                    start_pos_to_cluster_id[pp] = cluster_id
                else:
                    original_ss = start_pos_to_range[pp][0]
                    original_ee = start_pos_to_range[pp][1]
                    
                    if ss >= original_ss and ee <= original_ee:
                        start_pos_to_range[pp]=[ss,ee]
                        start_pos_to_cluster_id[pp]=cluster_id
     
            mm = mm.strip()
            if mm.startswith('your'):
                if sex == 'f':
                    mm = mm.replace('your ', 'her ')
                else:
                    mm = mm.replace('your ', 'his ')
            if mm not in mms:
                mms.append(mm)
                
        if "mike" in mms:
            mms = ['you', 'your', 'yourself', 'yours']
            if cluster_id not in narcissist_cluster_id_list:
                narcissist_cluster_id_list.append(cluster_id)
        cluster_id_to_string_set[cluster_id] = mms
     
    used_pos_list = []
    updated_cluster_id_to_mention_set = {}
    for cluster_id in cluster_id_to_mention_set:
        mention_set = cluster_id_to_mention_set[cluster_id]  
        updated_mention_set = {}         
        for ss in mention_set:
            if ss in start_pos_to_focus_entity_mention:
                continue
            pos_list_and_tokens = mention_set[ss]
            last_pos_and_token = pos_list_and_tokens[len(pos_list_and_tokens)-1]
            last_pos = last_pos_and_token[0]
            last_token = last_pos_and_token[1]
            ee = last_pos + len(last_token.text) - 1
            
            keep_mention = True
            for pos_and_token in pos_list_and_tokens:
                pp = pos_and_token[0]
                tt = pos_and_token[1]
                current_range = start_pos_to_range[pp]
                current_ss = current_range[0]
                current_ee = current_range[1]

                if pp in start_pos_to_focus_entity_mention:
                    keep_mention = False
                    break
                if current_ss != ss or current_ee != ee:
                    keep_mention = False
                    break
            if keep_mention and (ss not in used_pos_list):
                used_pos_list.append(ss)
                updated_mention_set[ss] = pos_list_and_tokens
        if len(updated_mention_set) > 0:
            updated_cluster_id_to_mention_set[cluster_id] = updated_mention_set
             
    # For testing purpose
    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    count5 = 0
    count6 = 0
    sorted_pos_list = sorted(start_pos_to_mention_conversion.keys())
    for sorted_pos in sorted_pos_list:
        mention_conversion = start_pos_to_mention_conversion[sorted_pos]
        original_text = mention_conversion._original_text
        converted_text = mention_conversion._converted_text
        cluster_id = mention_conversion._cluster_id
        color = cluster_id_to_color[cluster_id]
    
        if color == 'yellow':
            continue
        else:
            found_match = False
            found_cluster_id = -1
            for uci in updated_cluster_id_to_mention_set:
                ms = updated_cluster_id_to_mention_set[uci]
        
                if sorted_pos in ms:
                    found_match = True
                    found_cluster_id = uci
                    break
                              
            if not found_match:                
                if original_text.lower() != converted_text.lower():
                    count1 = count1 + 1
                    sentence_num = start_pos_to_sentence_num[sorted_pos]
                    sentence = sentences[sentence_num]
                    print("Annotated confounding entity mention was not identified: ")
                    print("Annotated original text: ")
                    print(original_text)
                    print("Annotated converted text: ")
                    print(converted_text)
                    print("Sentence: ")
                    print(sentence)
            else:
                count2 = count2 + 1
                string_set = cluster_id_to_string_set[found_cluster_id]
                ms = updated_cluster_id_to_mention_set[found_cluster_id]
                pos_list_and_tokens = ms[sorted_pos]
                mm = ''
                for pos_and_token in pos_list_and_tokens:
                    pp = pos_and_token[0]
                    tt = pos_and_token[1]
                    mm = mm + tt.text.lower() + ' '                
                mm = mm.strip()

                if original_text.lower() != mm.lower():
                    count3 = count3 + 1
                    sentence_num = start_pos_to_sentence_num[sorted_pos]
                    sentence = sentences[sentence_num]
                    print("Not matched confounding entity mention: ")
                    print("Annotated original mention: ")
                    print(original_text)
                    print("Identified original mention: ")
                    print(mm)
                    print("Sentence: ")
                    print(sentence)
                    print("mention string set: ")
                    print(string_set)
                     
    for uci in updated_cluster_id_to_mention_set:
        ms = updated_cluster_id_to_mention_set[uci]
        for ss in ms:
            pos_list_and_tokens = ms[ss]
            mm = ''
            for pos_and_token in pos_list_and_tokens:
                pp = pos_and_token[0]
                tt = pos_and_token[1]
                mm = mm + tt.text.lower() + ' '                
            mm = mm.strip()
                
            if ss not in start_pos_to_mention_conversion:
                count4 = count4 + 1
                sentence_num = start_pos_to_sentence_num[ss]
                sentence = sentences[sentence_num]
                print("Identified confounding entity mention was not annotated: ")
                print("Identified original text: ")
                print(mm)
                print("Sentence: ")
                print(sentence)
            else:
                count5 = count5 + 1
                mc = start_pos_to_mention_conversion[ss]
                string_set = cluster_id_to_string_set[uci]
                oo = mc._original_text
                cc = mc._converted_text
                                   
                if oo.lower() != mm.lower():
                    count6 = count6 + 1
                            
    print("Annotated confounding entity mention conversions that are not identified: ")            
    print(count1)
    print("Annotated confounding entity mention conversions that are identified: ")
    print(count2)
    print("Annotated confounding entity mention conversions that are identified but incorrectly identified: ")
    print(count3)
    print("Identified confounding entity mention conversions that are not annotated: ")
    print(count4)
    print("Identified confounding entity mention conversions that are annotated: ")
    print(count5)
    print("Identified confounding entity mention conversions that are annotated but incorrectly identified: ")
    print(count6)
    
    return updated_cluster_id_to_mention_set,narcissist_cluster_id_list

def read_verb_dictionary():
    verb_dic = {}
    f = open('conjugations_english.tab')
    lines = f.readlines()
    for line in lines:
        fields = line.split('\t')
        if fields[3] == '':
            continue
        else:
            verb_dic[fields[0].lower()] = fields[3].lower()
        
    return verb_dic
    f.close()
    
def get_singular_verb(word):
    singular_verb = ''
    if word == 'have':
        singular_verb = 'has'
    elif word == 'do':
        singular_verb = 'does'
    elif word == 'are' or word == 'am':
        singular_verb = 'is'
    elif word.endswith("s") or word.endswith("sh") or word.endswith("ch") or word.endswith("x") or word.endswith("z") or word.endswith("o"):
        singular_verb = word+"es"
    elif word.endswith("ay") or word.endswith("ey") or word.endswith("iy") or word.endswith("oy") or word.endswith("uy"):
        singular_verb = word+"s"
    elif word.endswith("y"):
        singular_verb = word[0:len(word)-1]+"ies"
    else:
        singular_verb = word+"s"
        
    return singular_verb

def change_verb_conjugation(document_text, start_pos_to_focus_entity_mention, start_pos_to_verb_conversion, start_pos_to_token, start_pos_to_sentence_num, token_index_to_start_pos, start_pos_to_token_index, verb_dic, is_second_person):
    splitter = SpacySentenceSplitter()
    sentences = splitter.split_sentences(document_text)
    tokenizer = SpacyTokenizer(pos_tags=True,ner=True)
    
    predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz")    
    non_change_modal_verbs = ['not','never','must', 'should', 'would', 'could', 'might', 'shall', 'will', 'can', 'may', 'maybe', '’d', '’ll', '\'d', '\'ll']
    verb_list_1 = ['’ve', '\'ve']
    verb_list_2 = ['am','\'m','’m']
    verb_list_3 = ['are', '’re', '\'re']

    start_pos_to_identified_verb_conversion = {}
    token_count = 0
    
    # For binary features: if is object/subject
    subject_pos_list = []
    object_pos_list = []
    
    # For extracting occupation words/phrases
    cop_words = []
    appos_words = []
    #other_words = []
    
    for i in range(len(sentences)):
        sentence = sentences[i]
        result = predictor.predict(sentence=sentence)
        pos_tags = result['pos']
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
        predicted_heads = result['predicted_heads']
        
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
                
            if start_pos not in start_pos_to_focus_entity_mention:
                continue
                       
            head = predicted_heads[j]
            
            # For extracting occupation words/phrases
            if dep == 'appos' and head != 0:
                head_tag = pos_tags[head-1]
                if head_tag == 'NOUN':
                    appos_words.append(words[head-1])
            
            # For extracting the head verb of the subject
            if dep == 'nsubj' or dep == 'nsubjpass' or dep == 'dep':
                if head != 0:
                    is_conj = False
                    for h in range(len(predicted_heads)):
                        predicted_head = predicted_heads[h]
                        if predicted_head == j+1:
                            temp_dep = dependencies[h]
                            if temp_dep == 'conj':
                                is_conj = True
                                break
                    if is_conj:
                        continue
                                    
                    head_pos = pos_tags[head-1]
                    if head_pos == 'VERB' or 'AUX':
                        aux_list = []
                        for h in range(len(predicted_heads)):
                            predicted_head = predicted_heads[h]
                            if predicted_head == head:
                                temp_dep = dependencies[h]
                                if temp_dep == 'aux' or temp_dep == 'auxpass' or temp_dep == 'cop':
                                    aux_list.append(h)
                                
                        contains_non_change_modal_verbs = False
                        aux_verb_index = -1
                        for h in aux_list:
                            v = words[h]
                            if v.lower() in non_change_modal_verbs:
                                contains_non_change_modal_verbs = True
                                break
                            else:
                                aux_verb_index = h
                                break
                            
                        if contains_non_change_modal_verbs:
                            continue                            
                        elif aux_verb_index != -1:
                            aux_token = tokens[aux_verb_index]
                            aux_lemma = aux_token.lemma_
                            aux_pos = pos_tags[aux_verb_index]
                            aux_tag = aux_token.tag_
                            if aux_pos != 'VERB' and aux_pos != 'AUX':
                                continue
                            converted_verb = ""
                            
                            if aux_token.text.lower() in verb_list_1:
                                converted_verb = 'has'
                            elif (aux_token.text.lower() in verb_list_2) or (aux_token.text.lower() in verb_list_3):
                                converted_verb = 'is'
                            elif is_second_person and (aux_token.text.lower() == 'were'):
                                converted_verb = 'was'
                            elif aux_token.text.lower() in non_change_modal_verbs:
                                continue
                            elif aux_lemma.lower() == aux_token.text.lower():
                                if aux_tag == 'VBD' or aux_tag == 'VBN':
                                    continue
                                if aux_lemma.lower() in verb_dic:
                                    converted_verb = verb_dic[aux_lemma.lower()]                             
                                else:
                                    print("not in dic: ")
                                    print(aux_lemma)
                                    continue
                            else:
                                continue
                            
                            aux_token_index = token_index + aux_verb_index - j
                            aux_token_start_pos = token_index_to_start_pos[aux_token_index]                           
                            verb_conversion = VerbConversion(aux_token_start_pos,aux_token.text,converted_verb)
                            start_pos_to_identified_verb_conversion[aux_token_start_pos] = verb_conversion
                        else:
                            verb_token = tokens[head-1]
                            verb_lemma = verb_token.lemma_
                            verb_pos = pos_tags[head-1]
                            verb_tag = verb_token.tag_
                            if verb_pos != 'VERB' and verb_pos != 'AUX':
                                continue
                            converted_verb = ''
                            if verb_token.text.lower() in verb_list_1:
                                converted_verb = 'has'
                            elif (verb_token.text.lower() in verb_list_2) or (verb_token.text.lower() in verb_list_3):
                                converted_verb = 'is'
                            elif is_second_person and (verb_token.text.lower() == 'were'):
                                converted_verb = 'was'
                            elif verb_token.text.lower() in non_change_modal_verbs:
                                continue
                            elif verb_lemma.lower() == verb_token.text.lower():
                                if verb_tag == 'VBD' or verb_tag == 'VBN':
                                    continue
                                if verb_lemma.lower() in verb_dic:
                                    converted_verb = verb_dic[verb_lemma.lower()]
                                else:
                                    print("not in dic: ")
                                    print(verb_lemma)
                                    continue
                            else:
                                continue
                                                       
                            verb_token_index = token_index + head - 1 - j
                            verb_token_start_pos = token_index_to_start_pos[verb_token_index]
                            verb_conversion = VerbConversion(verb_token_start_pos, verb_token.text, converted_verb)
                            start_pos_to_identified_verb_conversion[verb_token_start_pos] = verb_conversion
                    else:
                        cop_list = []
                        aux_list = []
                        for h in range(len(predicted_heads)):
                            predicted_head = predicted_heads[h]
                            if predicted_head == head:
                                temp_dep = dependencies[h]
                                if temp_dep == 'cop':
                                    cop_list.append(h)
                                elif temp_dep == 'aux' or temp_dep == 'auxpass':
                                    aux_list.append(h)
                                    
                        contains_non_change_modal_verbs = False
                        aux_verb_index = -1
                        if len(cop_list) == 0:
                            continue
                        
                        # For extracting occupation words/phrases
                        if head_pos == 'NOUN':
                            cop_words.append(words[head-1])
                            
                        for h in aux_list:
                            v = words[h]
                            if v.lower() in non_change_modal_verbs:
                                contains_non_change_modal_verbs = True
                                break
                            else:
                                aux_verb_index = h
                                break
                            
                        cop_index = -1
                        for h in cop_list:
                            v = words[h]
                            cop_index = h
                            break
                        
                        if contains_non_change_modal_verbs:
                            continue                            
                        elif aux_verb_index != -1:
                            aux_token = tokens[aux_verb_index]
                            aux_pos = pos_tags[aux_verb_index]
                            aux_tag = aux_token.tag_
                            if aux_pos != 'VERB' and aux_pos != 'AUX':
                                continue
                            aux_lemma = aux_token.lemma_
                            converted_verb = ""
                            
                            if aux_token.text.lower() in verb_list_1:
                                converted_verb = 'has'
                            elif (aux_token.text.lower() in verb_list_2) or (aux_token.text.lower() in verb_list_3):
                                converted_verb = 'is'
                            elif is_second_person and (aux_token.text.lower() == 'were'):
                                converted_verb = 'was'
                            elif aux_token.text.lower() in non_change_modal_verbs:
                                continue
                            elif aux_lemma.lower() == aux_token.text.lower():
                                if aux_tag == 'VBD' or aux_tag == 'VBN':
                                    continue
                                if aux_lemma.lower() in verb_dic:
                                    converted_verb = verb_dic[aux_lemma.lower()]
                                else:   
                                    print("not in dic:")
                                    print(aux_lemma)
                                    continue
                            else:
                                continue
                                                                                                           
                            aux_token_index = token_index + aux_verb_index - j
                            aux_token_start_pos = token_index_to_start_pos[aux_token_index]
                            verb_conversion = VerbConversion(aux_token_start_pos,aux_token.text,converted_verb)
                            start_pos_to_identified_verb_conversion[aux_token_start_pos] = verb_conversion
                        else:
                            verb_token = tokens[cop_index]
                            verb_pos = pos_tags[cop_index]
                            verb_tag = verb_token.tag_
                            if verb_pos != 'VERB' and verb_pos != 'AUX':
                                continue
                            verb_lemma = verb_token.lemma_
                            converted_verb = ''
                            if verb_token.text.lower() in verb_list_1:
                                converted_verb = 'has'
                            elif (verb_token.text.lower in verb_list_2) or (verb_token.text.lower() in verb_list_3):
                                converted_verb = 'is'
                            elif is_second_person and (verb_token.text.lower() == 'were'):
                                converted_verb = 'was'
                            elif verb_token.text.lower() in non_change_modal_verbs:
                                continue
                            elif verb_lemma.lower() == verb_token.text.lower():
                                if verb_tag == 'VBD' or verb_tag == 'VBN':
                                    continue
                                if verb_lemma.lower() in verb_dic:
                                    converted_verb = verb_dic[verb_lemma.lower()]
                                else:
                                    print("not in dic:")
                                    print(verb_lemma)
                                    continue
                            else:
                                continue
                                                        
                            verb_token_index = token_index + head - 1 - j
                            verb_token_start_pos = token_index_to_start_pos[verb_token_index]
                            verb_conversion = VerbConversion(verb_token_start_pos, verb_token.text, converted_verb)
                            start_pos_to_identified_verb_conversion[verb_token_start_pos] = verb_conversion 
    
    #For testing purpose
    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    count5 = 0
    count6 = 0
    
    for start_pos in sorted(start_pos_to_verb_conversion.keys()):
        verb_conversion = start_pos_to_verb_conversion[start_pos]
        original_verb = verb_conversion._original_text
        converted_verb = verb_conversion._converted_text
        
        if start_pos not in start_pos_to_identified_verb_conversion:
            print("Cannot map the annotated verb conversion:")
            print("Annotated original verb: ")
            print(original_verb)
            print("Annotated converted verb: ")
            print(converted_verb)
            count1 = count1 + 1
        else:
            i_verb_conversion = start_pos_to_identified_verb_conversion[start_pos]
            i_original_verb = i_verb_conversion._original_text
            i_converted_verb = i_verb_conversion._converted_text
            count2 = count2 + 1
            if original_verb.lower() != i_original_verb.lower() or converted_verb.lower() != i_converted_verb.lower():
                print("Not matched verb conversions: ")
                print("Annotated original verb: ")
                print(original_verb)
                print("Identified original verb: ")
                print(i_original_verb)
                print("Annotated converted verb: ")
                print(converted_verb)
                print("Identified converted verb: ")
                print(i_converted_verb)
                count3 = count3 + 1
                
    for start_pos in sorted(start_pos_to_identified_verb_conversion.keys()):
         i_verb_conversion = start_pos_to_identified_verb_conversion[start_pos]
         i_original_verb = i_verb_conversion._original_text
         i_converted_verb = i_verb_conversion._converted_text
         if i_converted_verb == '':
            print('wrong')
         if start_pos not in start_pos_to_verb_conversion:
             print("Cannot map the identified verb conversion:")
             print("Identified original verb: ")
             print(i_original_verb)
             print("Identified converted verb: ")
             print(i_converted_verb)
             count4 = count4 + 1
         else:
             verb_conversion = start_pos_to_verb_conversion[start_pos]
             original_verb = verb_conversion._original_text
             converted_verb = verb_conversion._converted_text
             count5 = count5 + 1
             if original_verb.lower() != i_original_verb.lower() or converted_verb.lower() != i_converted_verb.lower():
                 count6 = count6 + 1
    
    print("Annotated verb conversions that are not identified: ")            
    print(count1)
    print("Annotated verb conversions that are identified: ")
    print(count2)
    print("Annotated verb conversions that are identified but incorrectly converted: ")
    print(count3)
    print("Identified verb conversions that are not annotated: ")
    print(count4)
    print("Identified verb conversions that are annotated: ")
    print(count5)
    print("Identified verb conversions that are annotated but incorrectly converted: ")
    print(count6)
    
    print("cop words: ")
    print(cop_words)
    print("appos words: ")
    print(appos_words)
    
    splitter = None
    tokenizer = None
    predictor = None
    return start_pos_to_identified_verb_conversion, subject_pos_list, object_pos_list

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

def write_cluster_info(f, start_pos_to_mention_conversion, start_pos_to_focus_entity_mention, cluster_id_to_mention_set, focus_mention_string_set, start_pos_to_token,narcissist_cluster_id_list, sex):
    valid_cluster_ids = []
    
    second_person_pronouns = ['you', 'your', 'yourself', 'yours']    
    
    gold_cluster_id_to_mentions = {}
    auto_cluster_id_to_mentions = {}
    cluster_id_to_mention_tags = {}
    
    valid_cluster_ids.append(-1)

    sorted_start_pos_list = sorted(start_pos_to_token.keys())
    for start_pos in start_pos_to_focus_entity_mention:
        original_text = start_pos_to_focus_entity_mention[start_pos].text.lower()
        converted_text = original_text
        tag_list = []
        if start_pos in start_pos_to_mention_conversion:
            mention_conversion = start_pos_to_mention_conversion[start_pos]
            original_text = mention_conversion._original_text.lower()
            converted_text = mention_conversion._converted_text.lower()
            converted_text = correct_string(converted_text)
        mention_length = len(original_text.split())     
        if mention_length != 1:
            print("error - double auto second person")
        insert_index = sorted_start_pos_list.index(start_pos)
        for iii in range(insert_index,insert_index+mention_length):
            sss = sorted_start_pos_list[iii]
            ttt = start_pos_to_token[sss]
            tag = ttt.tag_
            tag_list.append(tag)

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
    
    start_pos_to_cluster_id = {}
    start_pos_to_mm = {}            
    for cluster_id in cluster_id_to_mention_set:            
        start_pos_to_mention = cluster_id_to_mention_set[cluster_id]
        for start_pos in start_pos_to_mention:
            tt_list = start_pos_to_mention[start_pos]
            start_pos_to_mm[start_pos]=tt_list
            start_pos_to_cluster_id[start_pos]=cluster_id
            original_text = ''
            for tt in tt_list:
                original_text = original_text + ' ' + tt[1].text.lower()
            original_text = original_text.strip()
            if original_text.startswith('your'):
                if sex == 'f':
                    original_text = original_text.replace('your ', 'her ')
                else:
                    original_text = original_text.replace('your ', 'his ')
            converted_text = original_text
            tag_list = []
            if start_pos in start_pos_to_mention_conversion:
                mention_conversion = start_pos_to_mention_conversion[start_pos]
                ot = mention_conversion._original_text.lower()
                ot = correct_string(ot)
                if ot == original_text:                    
                    converted_text = mention_conversion._converted_text.lower()
                    converted_text = correct_string(converted_text)
                else:
                    if cluster_id in narcissist_cluster_id_list:
                        print('error')
                        print(ot)
                        print(original_text)
            mention_length = len(original_text.split())       
            insert_index = sorted_start_pos_list.index(start_pos)
            for iii in range(insert_index,insert_index+mention_length):
                sss = sorted_start_pos_list[iii]
                ttt = start_pos_to_token[sss]
                tag = ttt.tag_
                tag_list.append(tag)
            
  
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
            if cluster_id in narcissist_cluster_id_list:
                continue
            if cluster_id not in auto_cluster_id_to_mentions:
                ll = []
                ll.append(original_text.lower())
                auto_cluster_id_to_mentions[cluster_id] = ll
            else:
                ll = auto_cluster_id_to_mentions[cluster_id]
                ll.append(original_text.lower())
                auto_cluster_id_to_mentions[cluster_id] = ll
         
    auto_cluster_id_to_mentions[-1] = focus_mention_string_set
    for narcissist_cluster_id in narcissist_cluster_id_list:
        auto_cluster_id_to_mentions[narcissist_cluster_id] = second_person_pronouns
            
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
            
    return valid_cluster_ids,gold_cluster_id_to_mentions,auto_cluster_id_to_mentions,cluster_id_to_mention_tags, start_pos_to_cluster_id, start_pos_to_mm

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

def fourth_round_process(pre_padding, post_padding, from_sent_num, to_sent_num, from_token_index, to_token_index, sentences, sent_num_to_mentions, key_to_cluster_id, key_to_original_mention_text, key_to_converted_mention_text, original_mention_len, converted_mention_len, tokens_list, sentence_num_to_token_index_list, token_index_to_start_pos, start_pos_to_mention_conversion, start_pos_to_identified_verb_conversion, invalid_cluster_ids=[]):
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
                if temp_start_pos in start_pos_to_identified_verb_conversion:
                    verb_conversion = start_pos_to_identified_verb_conversion[temp_start_pos]
                    temp_original_text = verb_conversion._original_text.lower()
                    if temp_original_text != token.text.lower():
                        print('error - double auto verb')
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
                        
                        # tt_index = token_index_list[j]
                        # tt_start_pos = token_index_to_start_pos[tt_index]
                        # tt_mc = start_pos_to_mention_conversion[tt_start_pos]
                        tt_ot = key_to_original_mention_text[m_key]
                        tt_ct = key_to_converted_mention_text[m_key]
                        
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
                if temp_start_pos in start_pos_to_identified_verb_conversion:
                    verb_conversion = start_pos_to_identified_verb_conversion[temp_start_pos]
                    temp_original_text = verb_conversion._original_text.lower()
                    if temp_original_text != token.text.lower():
                        print('error - double auto verb')
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
                        
                        # tt_index = token_index_list[j]
                        # tt_start_pos = token_index_to_start_pos[tt_index]
                        # tt_mc = start_pos_to_mention_conversion[tt_start_pos]
                        tt_ot = key_to_original_mention_text[m_key]
                        tt_ct = key_to_converted_mention_text[m_key]
                        
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
                if temp_start_pos in start_pos_to_identified_verb_conversion:
                    verb_conversion = start_pos_to_identified_verb_conversion[temp_start_pos]
                    temp_original_text = verb_conversion._original_text.lower()
                    if temp_original_text != token.text.lower():
                        print('error - double auto verb')
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
                        
                        # tt_index = token_index_list[j]
                        # tt_start_pos = token_index_to_start_pos[tt_index]
                        # tt_mc = start_pos_to_mention_conversion[tt_start_pos]
                        tt_ot = key_to_original_mention_text[m_key]
                        tt_ct = key_to_converted_mention_text[m_key]
                        
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
                if temp_start_pos in start_pos_to_identified_verb_conversion:
                    verb_conversion = start_pos_to_identified_verb_conversion[temp_start_pos]
                    temp_original_text = verb_conversion._original_text.lower()
                    if temp_original_text != token.text.lower():
                        print('error - double auto verb')
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
                        
                        # tt_index = token_index_list[j]
                        # tt_start_pos = token_index_to_start_pos[tt_index]
                        # tt_mc = start_pos_to_mention_conversion[tt_start_pos]
                        tt_ot = key_to_original_mention_text[m_key]
                        tt_ct = key_to_converted_mention_text[m_key]
                        
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
 
def get_sent_num_to_mentions(start_pos_to_sentence_num, start_pos_to_token, start_pos_to_mention_conversion, start_pos_to_verb_conversion, start_pos_to_focus_entity_mention, cluster_id_to_mention_set, start_pos_to_identified_verb_conversion, sentence_num_to_token_index_list, token_index_to_start_pos, start_pos_to_cluster_id, start_pos_to_mm, sex):
    sent_num_to_identified_mentions = {}
    identified_key_to_cluster_id = {}
    identified_key_to_original_mention_text = {}
    identified_key_to_converted_mention_text = {}
    
    sent_num_to_annotated_mentions = {}
    annotated_key_to_mention_conversion = {}
    
    sent_num_to_identified_verbs = {}
    identified_key_to_verb_conversion = {}
    
    sent_num_to_annotated_verbs = {}
    annotated_key_to_verb_conversion = {}
    
    second_person_pronouns = {'you','your','yourself','yours'}
    first_person_plural_pronouns = {'we','us','our','ourselves','ours'}
    
    for sent_num in sentence_num_to_token_index_list:
        token_index_list = sentence_num_to_token_index_list[sent_num]
        identified_mentions=[0]*len(token_index_list)
        annotated_mentions = [0]*len(token_index_list)
        identified_verbs = [0]*len(token_index_list)
        annotated_verbs = [0]*len(token_index_list)
        for i in range(len(token_index_list)):
            token_index = token_index_list[i]
            start_pos = token_index_to_start_pos[token_index]
            ot = ''
            ct = ''
            ci = -4
            
            if start_pos in start_pos_to_identified_verb_conversion:
                verb = i
                if i == 0:
                    verb = -1
                identified_verbs[i]=verb
                key = str(sent_num)+":"+str(verb)
                identified_key_to_verb_conversion[key]=start_pos_to_identified_verb_conversion[start_pos]
            if start_pos in start_pos_to_verb_conversion:
                verb = i
                if i == 0:
                    verb = -1
                annotated_verbs[i]=verb
                key = str(sent_num)+":"+str(verb)
                annotated_key_to_verb_conversion[key]=start_pos_to_verb_conversion[start_pos]
                
            if start_pos in start_pos_to_focus_entity_mention:
                ot = start_pos_to_focus_entity_mention[start_pos].text.lower()
                ct = ot
                ci = -1
                if start_pos in start_pos_to_mention_conversion:
                    mc = start_pos_to_mention_conversion[start_pos]
                    ct = mc._converted_text.lower()
                    ct = correct_string(ct)
            elif start_pos in start_pos_to_mm:
                mm = start_pos_to_mm[start_pos]
                for m in mm:
                    ot = ot + m[1].text.lower() + ' '
                ot = ot.strip()
                ct = ot
                if ct.startswith('your'):
                    if sex == 'f':
                        ct = ct.replace('your ', 'her ')
                    else:
                        ct = ct.replace('your ', 'his ')
                ci = start_pos_to_cluster_id[start_pos]
                if start_pos in start_pos_to_mention_conversion:
                    mc = start_pos_to_mention_conversion[start_pos]
                    oot = mc._original_text.lower()
                    oot = correct_string(oot)
                    if oot == ot:
                        ct = mc._converted_text.lower()
                        ct = correct_string(ct)
 
            if ci == -4:
                if start_pos in start_pos_to_mention_conversion:
                    mc = start_pos_to_mention_conversion[start_pos]
                    ot = mc._original_text.lower()
                    ot = correct_string(ot)
                    ct = mc._converted_text.lower()
                    ct = correct_string(ct)
                    ot_parts = ot.split()
                    mention=0
                    if i+len(ot_parts)-1 == 0:
                        mention=-1
                    else:
                        mention=i+len(ot_parts)-1
                    annotated_mentions[i]=mention
                    key = str(sent_num)+":"+str(i)+":"+str(mention)
                    annotated_key_to_mention_conversion[key]= start_pos_to_mention_conversion[start_pos]
                continue
            ot_parts = ot.split()
            mention=0
            if i+len(ot_parts)-1 == 0:
                mention=-1
            else:
                mention=i+len(ot_parts)-1
            identified_mentions[i]=mention
            key = str(sent_num)+":"+str(i)+":"+str(mention)
            if ot.lower() in second_person_pronouns:
                identified_key_to_cluster_id[key]=-1
            else:
                identified_key_to_cluster_id[key]=ci
                    
            identified_key_to_original_mention_text[key]=ot.lower()
            identified_key_to_converted_mention_text[key]=ct.lower()
        sent_num_to_identified_mentions[sent_num]=identified_mentions
        sent_num_to_annotated_mentions[sent_num]=annotated_mentions
        sent_num_to_identified_verbs[sent_num]=identified_verbs
        sent_num_to_annotated_verbs[sent_num]=annotated_verbs
        
    return sent_num_to_identified_mentions,identified_key_to_cluster_id,identified_key_to_original_mention_text,identified_key_to_converted_mention_text,sent_num_to_annotated_mentions,annotated_key_to_mention_conversion,sent_num_to_identified_verbs,identified_key_to_verb_conversion,sent_num_to_annotated_verbs,annotated_key_to_verb_conversion
          
def process_data_auto(output_file, document_text, start_pos_to_mention_conversion, start_pos_to_verb_conversion,start_pos_to_focus_entity_mention, cluster_id_to_mention_set, start_pos_to_identified_verb_conversion, highlight_color_to_cluster_id, shading_color_to_cluster_id, sentences, start_pos_to_token, start_pos_to_sentence_num, sentence_num_to_sent_start_num, start_pos_to_token_index, token_index_to_start_pos, focus_mention_string_set, sentence_num_to_token_index_list, tokens_list,narcissist_cluster_id_list, subject_pos_list, object_pos_list,sex):    
    f = open(output_file, "w+", encoding="utf-8")

    valid_cluster_ids,gold_cluster_id_to_mentions,auto_cluster_id_to_mentions,cluster_id_to_mention_tags,start_pos_to_cluster_id, start_pos_to_mm=write_cluster_info(f, start_pos_to_mention_conversion, start_pos_to_focus_entity_mention, cluster_id_to_mention_set, focus_mention_string_set, start_pos_to_token,narcissist_cluster_id_list,sex)
     
    sent_num_to_identified_mentions,identified_key_to_cluster_id,identified_key_to_original_mention_text,identified_key_to_converted_mention_text,sent_num_to_annotated_mentions,annotated_key_to_mention_conversion,sent_num_to_identified_verbs,identified_key_to_verb_conversion,sent_num_to_annotated_verbs,annotated_key_to_verb_conversion = get_sent_num_to_mentions(start_pos_to_sentence_num, start_pos_to_token, start_pos_to_mention_conversion, start_pos_to_verb_conversion,start_pos_to_focus_entity_mention, cluster_id_to_mention_set, start_pos_to_identified_verb_conversion, sentence_num_to_token_index_list, token_index_to_start_pos, start_pos_to_cluster_id, start_pos_to_mm,sex)
    
    f.write('<document_text>')
    f.write('\n')
    f.write(document_text)
    f.write('\n')
    f.write('<document_text>')
    f.write('\n')
    for sent_num in sorted(sent_num_to_identified_mentions.keys()):       
        identified_mentions = sent_num_to_identified_mentions[sent_num]
        annotated_mentions = sent_num_to_annotated_mentions[sent_num]
        identified_verbs = sent_num_to_identified_verbs[sent_num]
        annotated_verbs = sent_num_to_annotated_verbs[sent_num]
        
        tt_index_list = sentence_num_to_token_index_list[sent_num]

        for i in range(len(identified_mentions)):
            identified_mention = identified_mentions[i]
            annotated_mention = annotated_mentions[i]
            identified_verb = identified_verbs[i]
            annotated_verb = annotated_verbs[i]
            token_index = tt_index_list[i]
            sp = token_index_to_start_pos[token_index]
            if identified_mention != 0:
                key = str(sent_num) + ":" + str(i) + ":" + str(identified_mention)
                cluster_id = identified_key_to_cluster_id[key]
                original_mention_text = identified_key_to_original_mention_text[key]
                original_mention_text_parts = original_mention_text.strip().split()
                converted_mention_text = identified_key_to_converted_mention_text[key]
                converted_mention_text_parts = converted_mention_text.strip().split()               
                annotated_original_mention_text = ''              

                if cluster_id not in valid_cluster_ids:
                    continue
                
                is_subject = False
                is_object = False
                eep = identified_mention
                if identified_mention == -1:
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
                f.write("<annotated_mention>: ")
                if sp in start_pos_to_mention_conversion:            
                    f.write("1")
                    annotated_original_mention_text = start_pos_to_mention_conversion[sp]._original_text.lower()
                else:
                    f.write("0")
                f.write("\t")
                f.write('<start_pos>: ')
                f.write(str(sp))
                f.write('\t')
                f.write("<identified_original_focus_mention>: ")
                f.write(original_mention_text)
                f.write("\t")
                f.write("<annotated_original_focus_mention>: ")
                f.write(annotated_original_mention_text)
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
                
                if identified_mention == -1:
                    identified_mention = 0
                from_sent_num, from_token_index, pre_padding = get_from_info(sent_num, i, sentences, tokens_list)
                to_sent_num, to_token_index, post_padding = get_to_info(sent_num, identified_mention, sentences, tokens_list)
                
                related_mentions, original_related_mention_text_list, converted_related_mention_text_list, related_mention_cluster_ids, raw_sequence, postags = fourth_round_process(pre_padding, post_padding, from_sent_num, to_sent_num, from_token_index, to_token_index, sentences, sent_num_to_identified_mentions, identified_key_to_cluster_id, identified_key_to_original_mention_text, identified_key_to_converted_mention_text, len(original_mention_text_parts), len(converted_mention_text_parts), tokens_list, sentence_num_to_token_index_list, token_index_to_start_pos, start_pos_to_mention_conversion, start_pos_to_identified_verb_conversion)  
                
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
                        if temp_start_pos in start_pos_to_identified_verb_conversion:
                            verb_conversion = start_pos_to_identified_verb_conversion[temp_start_pos]
                            temp_original_text = verb_conversion._original_text.lower()
                            if temp_original_text != token.text.lower():
                                print('error - double auto verb')
                            temp_converted_text = verb_conversion._converted_text
                            to_add_text = temp_converted_text.lower()
#                            if to_add_text == '':
#                                print("empty verb: ")
#                                print(temp_original_text)
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
                
                original_pre_mention_sequence, converted_pre_mention_sequence, pre_mention_cluster_id_sequence, pre_mention_distance_sequence, pre_sent_index_list, pre_end_pos_list, pre_start_pos_list = get_from_mention_info(sent_num, i, sentences, tokens_list, sent_num_to_identified_mentions, identified_key_to_cluster_id, identified_key_to_original_mention_text, identified_key_to_converted_mention_text)
                
                original_post_mention_sequence, converted_post_mention_sequence, post_mention_cluster_id_sequence, post_mention_distance_sequence, post_sent_index_list, post_end_pos_list, post_start_pos_list = get_to_mention_info(sent_num, i, sentences, tokens_list, sent_num_to_identified_mentions, identified_key_to_cluster_id, identified_key_to_original_mention_text, identified_key_to_converted_mention_text, pre_sent_index_list, pre_end_pos_list)   
                                          
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
                    this_mentions = sent_num_to_identified_mentions[sent_index]
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
                        if temp_start_pos in start_pos_to_identified_verb_conversion:
                            verb_conversion = start_pos_to_identified_verb_conversion[temp_start_pos]
                            temp_original_text = verb_conversion._original_text.lower()
                            if temp_original_text != token.text.lower():
                                print('error - double auto verb')
                            temp_converted_text = verb_conversion._converted_text
                            to_add_text = temp_converted_text.lower()
                        
                        this_mention = this_mentions[h]
                        if this_mention != 0:
                            this_key = str(sent_index) + ":" + str(h) + ":" + str(this_mention)                        
                            tt_oo_mm = identified_key_to_original_mention_text[this_key].lower()
                            tt_oo_mm_parts = tt_oo_mm.split()
                            tt_cc_mm = identified_key_to_converted_mention_text[this_key].lower()
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
                    this_mentions = sent_num_to_identified_mentions[sent_index]
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
                        if temp_start_pos in start_pos_to_identified_verb_conversion:
                            verb_conversion = start_pos_to_identified_verb_conversion[temp_start_pos]
                            temp_original_text = verb_conversion._original_text.lower()
                            if temp_original_text != token.text.lower():
                                print('error - double auto verb')
                            temp_converted_text = verb_conversion._converted_text
                            to_add_text = temp_converted_text.lower()
                        this_mention = this_mentions[h]
                        
                        if this_mention != 0:
                            this_key = str(sent_index) + ":" + str(h) + ":" + str(this_mention)                        
                            tt_oo_mm = identified_key_to_original_mention_text[this_key].lower()
                            tt_oo_mm_parts = tt_oo_mm.split()
                            tt_cc_mm = identified_key_to_converted_mention_text[this_key].lower()
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
            elif annotated_mention != 0:
                f.write('<identified_mention>: 0')
                f.write('\t')
                f.write('<annotated_mention>: 1')
                f.write('\t')
                f.write('<start_pos>: ')
                f.write(str(sp))
                f.write('\t')
                key = str(sent_num) + ":" + str(i) + ":" + str(annotated_mention)
                annotated_mention_conversion = annotated_key_to_mention_conversion[key]
                f.write("<annotated_original_focus_mention>: ")
                f.write(annotated_mention_conversion._original_text.lower())
                f.write("\t")
                f.write("<annotated_converted_focus_mention>: ")
                f.write(annotated_mention_conversion._converted_text.lower())
                f.write('\n')
            elif annotated_verb != 0 and identified_verb != 0:
                key = str(sent_num) + ":" + str(identified_verb)
                f.write('<identified_verb>: 1')
                f.write('\t')
                f.write('<annotated_verb>: 1')
                f.write('\t')
                annotated_verb_conversion = annotated_key_to_verb_conversion[key]
                identified_verb_conversion = identified_key_to_verb_conversion[key]
                f.write('<start_pos>: ')
                f.write(str(sp))
                f.write('\t')
                f.write('<annotated_converted_verb>: ')
                f.write(annotated_verb_conversion._converted_text.lower())
                f.write('\t')
                f.write('<identified_converted_verb>: ')
                f.write(identified_verb_conversion._converted_text.lower())
                f.write('\n')
            elif annotated_verb != 0:
                key = str(sent_num) + ":" + str(annotated_verb)
                f.write('<identified_verb>: 0')
                f.write('\t')
                f.write('<annotated_verb>: 1')
                f.write('\t')
                annotated_verb_conversion = annotated_key_to_verb_conversion[key]
                f.write('<start_pos>: ')
                f.write(str(sp))
                f.write('\t')
                f.write('<annotated_original_verb>: ')
                f.write(annotated_verb_conversion._original_text.lower())
                f.write('\t')
                f.write('<annotated_converted_verb>: ')
                f.write(annotated_verb_conversion._converted_text.lower())
                f.write('\n')
            elif identified_verb != 0:
                key = str(sent_num) + ":" + str(identified_verb)
                f.write('<identified_verb>: 1')
                f.write('\t')
                f.write('<annotated_verb>: 0')
                f.write('\t')
                identified_verb_conversion = identified_key_to_verb_conversion[key]
                f.write('<start_pos>: ')
                f.write(str(sp))
                f.write('\t')
                f.write('<identified_original_verb>: ')
                f.write(identified_verb_conversion._original_text.lower())
                f.write('\t')
                f.write('<identified_converted_verb>: ')
                f.write(identified_verb_conversion._converted_text.lower())
                f.write('\n')

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
  
def generate_mention_string_set_from_relational_nouns_auto(start_pos_to_focus_entity_mention, start_pos_to_token, cfr_document, cfr_clusters, cfr_index_to_token_index, token_index_to_start_pos, focus_mention_gender):
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
        
    sorted_start_pos_list = sorted(start_pos_to_token.keys())
    for i in range(len(sorted_start_pos_list)):
        sorted_start_pos = sorted_start_pos_list[i]
        if sorted_start_pos in start_pos_to_focus_entity_mention:
            entity_mention = start_pos_to_focus_entity_mention[sorted_start_pos].text.lower()
            if entity_mention == 'your':
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
                sex = mentions[i]
            elif i == 0:
                focus_mention_string_set.append(mentions[i])
    
    f.close()    
    return focus_mention_string_set,sex
    
def main():
    parser = argparse.ArgumentParser('Preprocess')

    parser.add_argument('--input_file', 
                    type=str, 
                    default='pov_data/Narcissistic_Abuse_with_annotations_both_replace.docx', 
                    help='input file.')
    parser.add_argument('--output_file',
                    type=str,
                    default='output_data_auto/narcissist.txt',
                    help='output file.')
    parser.add_argument('--focus_mention_string_path',
                    type=str,
                    default='focus_mention_string/Narcissistic_Abuse_with_annotations_both_replace.txt',
                    help='Path of focus mention string set.')
    parser.add_argument('--is_second_person',
                    type=bool,
                    default=True,
                    help='If the original PoV is second person.')
    
    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()
    input_file = FLAGS.input_file
    output_file = FLAGS.output_file
    focus_mention_string_path = FLAGS.focus_mention_string_path
    is_second_person=FLAGS.is_second_person
    
    updated_text, start_pos_to_verb_conversion, start_pos_to_mention_conversion, highlight_color_to_cluster_id, shading_color_to_cluster_id = convert_word_doc_to_text(input_file)
    sentences, start_pos_to_token, start_pos_to_sentence_num, sentence_num_to_sent_start_num, cfr_document, cfr_clusters, cfr_index_to_token_index, start_pos_to_token_index, token_index_to_start_pos,sentence_num_to_token_index_list, tokens_list = preprocess(updated_text)
    
    start_pos_to_focus_entity_mention = extract_second_person_entity(updated_text, start_pos_to_verb_conversion, start_pos_to_mention_conversion, highlight_color_to_cluster_id, shading_color_to_cluster_id, sentences, start_pos_to_token, start_pos_to_sentence_num)   
    
    verb_dic = read_verb_dictionary()
    start_pos_to_identified_verb_conversion, subject_pos_list, object_pos_list = change_verb_conjugation(updated_text, start_pos_to_focus_entity_mention, start_pos_to_verb_conversion, start_pos_to_token, start_pos_to_sentence_num, token_index_to_start_pos, start_pos_to_token_index, verb_dic,is_second_person)
    
    temp_set,sex = read_mention_string(focus_mention_string_path)
    cluster_id_to_mention_set,narcissist_cluster_id_list = extract_confounding_entities(updated_text, start_pos_to_verb_conversion, start_pos_to_mention_conversion, highlight_color_to_cluster_id, shading_color_to_cluster_id, sentences, start_pos_to_token, start_pos_to_sentence_num, sentence_num_to_sent_start_num, cfr_document, cfr_clusters, cfr_index_to_token_index, token_index_to_start_pos, start_pos_to_focus_entity_mention, sex)
    
    focus_mention_string_set = generate_mention_string_set_from_relational_nouns_auto(start_pos_to_focus_entity_mention, start_pos_to_token, cfr_document, cfr_clusters, cfr_index_to_token_index, token_index_to_start_pos, sex)
        
    if sex == 'm':
        focus_mention_string_set.extend(['him', 'his', 'he', 'himself', 'he himself'])
    else:
        focus_mention_string_set.extend(['she','her','herself', 'she herself'])
    
    focus_mention_string_set.extend(temp_set)   
    process_data_auto(output_file, updated_text, start_pos_to_mention_conversion, start_pos_to_verb_conversion,start_pos_to_focus_entity_mention, cluster_id_to_mention_set, start_pos_to_identified_verb_conversion, highlight_color_to_cluster_id, shading_color_to_cluster_id, sentences, start_pos_to_token, start_pos_to_sentence_num, sentence_num_to_sent_start_num, start_pos_to_token_index, token_index_to_start_pos, focus_mention_string_set, sentence_num_to_token_index_list, tokens_list,narcissist_cluster_id_list, subject_pos_list, object_pos_list,sex)

if __name__ == "__main__":
    main()