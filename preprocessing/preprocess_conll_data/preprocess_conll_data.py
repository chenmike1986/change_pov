# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 15:20:04 2019

@author: chenm
"""

import logging
import collections
from typing import Any, Dict, List, Optional, Tuple, DefaultDict, Set

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.dataset_utils.ontonotes import OntonotesSentence
from allennlp.data.fields import Field, ListField, TextField, SpanField, MetadataField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.dataset_readers.dataset_utils import Ontonotes, enumerate_spans

from nltk.tree import Tree

import sys
import argparse

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def canonicalize_clusters(clusters: DefaultDict[int, List[Tuple[int, int]]]) -> List[List[Tuple[int, int]]]:
    """
    The CONLL 2012 data includes 2 annotated spans which are identical,
    but have different ids. This checks all clusters for spans which are
    identical, and if it finds any, merges the clusters containing the
    identical spans.
    """
    merged_clusters: List[Set[Tuple[int, int]]] = []
    for cluster in clusters.values():
        cluster_with_overlapping_mention = None
        for mention in cluster:
            # Look at clusters we have already processed to
            # see if they contain a mention in the current
            # cluster for comparison.
            for cluster2 in merged_clusters:
                if mention in cluster2:
                    # first cluster in merged clusters
                    # which contains this mention.
                    cluster_with_overlapping_mention = cluster2
                    break
            # Already encountered overlap - no need to keep looking.
            if cluster_with_overlapping_mention is not None:
                break
        if cluster_with_overlapping_mention is not None:
            # Merge cluster we are currently processing into
            # the cluster in the processed list.
            cluster_with_overlapping_mention.update(cluster)
        else:
            merged_clusters.append(set(cluster))
    return [list(c) for c in merged_clusters]

def normalize_word(word: str):
    if word == "/." or word == "/?":
        return word[1:]
    else:
        return word

def check_candidate_mentions_are_well_defined(span_starts, span_ends, text):
    candidate_mentions = []
    for start, end in zip(span_starts, span_ends):
        # Spans are inclusive.
        text_span = text[start: end + 1]
        candidate_mentions.append(text_span)

    # Check we aren't considering zero length spans and all
    # candidate spans are less than what we specified
    assert all([50 >= len(x) > 0 for x in candidate_mentions])  # pylint: disable=len-as-condition
    return candidate_mentions

class ConllCorefNERReader(DatasetReader):
    """
    Reads a single CoNLL-formatted file. This is the same file format as used in the
    :class:`~allennlp.data.dataset_readers.semantic_role_labelling.SrlReader`, but is preprocessed
    to dump all documents into a single file per train, dev and test split. See
    scripts/compile_coref_data.sh for more details of how to pre-process the Ontonotes 5.0 data
    into the correct format.
    Returns a ``Dataset`` where the ``Instances`` have four fields: ``text``, a ``TextField``
    containing the full document text, ``spans``, a ``ListField[SpanField]`` of inclusive start and
    end indices for span candidates, and ``metadata``, a ``MetadataField`` that stores the instance's
    original text. For data with gold cluster labels, we also include the original ``clusters``
    (a list of list of index pairs) and a ``SequenceLabelField`` of cluster ids for every span
    candidate.
    Parameters
    ----------
    max_span_width: ``int``, required.
        The maximum width of candidate spans to consider.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        This is used to index the words in the document.  See :class:`TokenIndexer`.
        Default is ``{"tokens": SingleIdTokenIndexer()}``.
    """
    def __init__(self,
                 max_span_width: int,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._max_span_width = max_span_width
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._ontonotes_sentence_set = []

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        ontonotes_reader = Ontonotes()
        for sentences in ontonotes_reader.dataset_document_iterator(file_path):
            self._ontonotes_sentence_set.append(sentences)
            
            clusters: DefaultDict[int, List[Tuple[int, int]]] = collections.defaultdict(list)

            total_tokens = 0
            for sentence in sentences:
                for typed_span in sentence.coref_spans:
                    # Coref annotations are on a _per sentence_
                    # basis, so we need to adjust them to be relative
                    # to the length of the document.
                    span_id, (start, end) = typed_span
                    clusters[span_id].append((start + total_tokens,
                                              end + total_tokens))
                total_tokens += len(sentence.words)

            canonical_clusters = canonicalize_clusters(clusters)
            yield self.text_to_instance([s.words for s in sentences], sentences, canonical_clusters)

    @overrides
    def text_to_instance(self,  # type: ignore
                         sentences: List[List[str]],
                         ontonotes_sentences: List[OntonotesSentence],
                         gold_clusters: Optional[List[List[Tuple[int, int]]]] = None) -> Instance:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        sentences : ``List[List[str]]``, required.
            A list of lists representing the tokenised words and sentences in the document.
        gold_clusters : ``Optional[List[List[Tuple[int, int]]]]``, optional (default = None)
            A list of all clusters in the document, represented as word spans. Each cluster
            contains some number of spans, which can be nested and overlap, but will never
            exactly match between clusters.
        Returns
        -------
        An ``Instance`` containing the following ``Fields``:
            text : ``TextField``
                The text of the full document.
            spans : ``ListField[SpanField]``
                A ListField containing the spans represented as ``SpanFields``
                with respect to the document text.
            span_labels : ``SequenceLabelField``, optional
                The id of the cluster which each possible span belongs to, or -1 if it does
                 not belong to a cluster. As these labels have variable length (it depends on
                 how many spans we are considering), we represent this a as a ``SequenceLabelField``
                 with respect to the ``spans ``ListField``.
        """
        flattened_sentences = [normalize_word(word)
                               for sentence in sentences
                               for word in sentence]

        metadata: Dict[str, Any] = {"original_text": flattened_sentences}
        if gold_clusters is not None:
            metadata["clusters"] = gold_clusters

        text_field = TextField([Token(word) for word in flattened_sentences], self._token_indexers)

        cluster_dict = {}
        if gold_clusters is not None:
            for cluster_id, cluster in enumerate(gold_clusters):
                for mention in cluster:
                    cluster_dict[tuple(mention)] = cluster_id

        spans: List[Field] = []
        span_labels: Optional[List[int]] = [] if gold_clusters is not None else None

        sentence_offset = 0
        for sentence in sentences:
            for start, end in enumerate_spans(sentence,
                                              offset=sentence_offset,
                                              max_span_width=self._max_span_width):
                if span_labels is not None:
                    if (start, end) in cluster_dict:
                        span_labels.append(cluster_dict[(start, end)])
                    else:
                        span_labels.append(-1)

                spans.append(SpanField(start, end, text_field))
            sentence_offset += len(sentence)            
            
        span_field = ListField(spans)

        sequences: List[Field] = []
        sequences2: List[Field] = []
        parse_trees: List[Tree] = []
        for ontonotes_sentence in ontonotes_sentences:
            tokens = [Token(normalize_word(t)) for t in ontonotes_sentence.words]
            ner_tags = ontonotes_sentence.named_entities
            pos_tags = ontonotes_sentence.pos_tags
            sequence = TextField(tokens, self._token_indexers)
            sequence2 = TextField(tokens, self._token_indexers)
            sequences.append(SequenceLabelField(ner_tags, sequence))
            sequences2.append(SequenceLabelField(pos_tags, sequence2))
            parse_tree = ontonotes_sentence.parse_tree
            parse_trees.append(parse_tree)
            
        sequence_field = ListField(sequences)
        sequence_field2 = ListField(sequences2)
        
        metadata["parsetrees"] = parse_trees       
        metadata_field = MetadataField(metadata)
        
        fields: Dict[str, Field] = {"text": text_field,
                                    "spans": span_field,
                                    "metadata": metadata_field,
                                    "tags": sequence_field,
                                    "postags": sequence_field2}
        if span_labels is not None:
            fields["span_labels"] = SequenceLabelField(span_labels, span_field)

        return Instance(fields)
 
def identify_subject_and_object(parse_tree):
    
    
    found_o = -1
    found_i = -1
    found_j = -1
    found_h = -1
    found_vp = False
    found_x = -1
    found_y = -1
    found_z = -1
    subject_pos_list = []
    object_pos_list = []
    if parse_tree is None:
        print('none tree')
        return subject_pos_list, object_pos_list
    
    leaves = parse_tree.leaves()
    for o,first_child in enumerate(parse_tree):
        if isinstance(first_child, Tree):
            label_0 = first_child.label()
            if label_0 == 'S':                
                for i, child in enumerate(first_child):        
                    if isinstance(child, Tree):
                        label_1 = child.label()
                        if label_1 == 'NP':
                            for j, grandchild in enumerate(child):
                                if isinstance(grandchild, Tree):
                                    label_2 = grandchild.label()
                                    if label_2 == 'NN' or label_2 == 'NNS' or label_2 == 'PRP' or label_2 == 'NNP' or label_2 == 'NNPS':
                                        for h, ggc in enumerate(grandchild):
                                            if not isinstance(ggc, Tree):
                                                found_o = o
                                                found_i = i
                                                found_j = j
                                                found_h = h 
                        if label_1 == 'VP':
                            found_vp = True
                            for x, grandchild in enumerate(child):
                                if isinstance(grandchild,Tree):
                                    label_2 = grandchild.label()
                                    if label_2 == 'NP':
                                        for y, ggc in enumerate(grandchild):
                                            if isinstance(ggc, Tree):
                                                label_3 = ggc.label()
                                                if label_3 == 'NN' or label_3 == 'NNS' or label_3 == 'PRP' or label_3 == 'NNP' or label_3 == 'NNPS':
                                                    for z, gggc in enumerate(ggc):
                                                        if not isinstance(gggc, Tree):
                                                            found_x = x
                                                            found_y = y
                                                            found_z = z

    for i in range(len(leaves)):
        tree_location = parse_tree.leaf_treeposition(i)
        if len(tree_location) == 4:
            if tree_location[0] == found_o and tree_location[1] == found_i and tree_location[2] == found_j and tree_location[3] == found_h and found_vp:
                subject_pos_list.append(i)
        if len(tree_location) == 5:
            if tree_location[0] == found_o and tree_location[1] == found_i and tree_location[2] == found_x and tree_location[3] == found_y and tree_location[4] == found_z:
                object_pos_list.append(i)
                
    return subject_pos_list, object_pos_list

def iterate_cluster(cluster, text, tags, parse_trees, token_id_to_sentence_num, token_id_to_index_in_sentence):
    first_person_pronouns = ["i","me","myself","me"]
    second_person_pronouns = ["you","your","yourself"]
    third_male_person_pronouns = ["he", "him", "his", "himself"]
    third_female_person_pronouns = ["she", "her", "herself"]
    plural_third_person_pronouns = ["they", "them", "their", "themselves"]
    
    is_person = False
    is_valid_first_person = False
    is_valid_second_person = False
    is_valid_male_third_person = False
    is_valid_female_third_person = False
    is_valid_plural_third_person = False
    
    subject_start_pos_list = []
    object_start_pos_list = []
    for mention in cluster:
        mention_tuple = tuple(mention)
        mention_start = mention_tuple[0]
        mention_end = mention_tuple[1]
        mention_text_span = text[mention_start:mention_end+1]
        tag_subset = tags[mention_start:mention_end+1]
        mention_text = ""
        for m in mention_text_span:
            mention_text = mention_text + " " + m
        mention_text = mention_text.strip()
        mention_text = mention_text.lower()
            
        mention_sentence_num = token_id_to_sentence_num[mention_start]
        mention_index_in_sentence = token_id_to_index_in_sentence[mention_start]
        parse_tree = parse_trees[mention_sentence_num]
        subject_pos_list, object_pos_list = identify_subject_and_object(parse_tree)
        
        is_subject = False
        is_object = False
        for i in range(mention_index_in_sentence, mention_index_in_sentence+mention_end-mention_start+1):
            if i in subject_pos_list:
               is_subject = True
               break
            elif i in object_pos_list:
                is_object = True
                break
            
        if is_subject:
            subject_start_pos_list.append(mention_start)
        elif is_object:
            object_start_pos_list.append(mention_start)
            
        found_none_person = False
        if not is_person:
            for tag in tag_subset:
                if tag.find("PERSON") == -1:
                    found_none_person = True
                    break
            if not found_none_person:
                is_person = True
                            
        if mention_text in first_person_pronouns:
            is_valid_first_person = True
                            
        if mention_text in second_person_pronouns:
            is_valid_second_person = True
                        
        if mention_text in third_male_person_pronouns:
            is_valid_male_third_person = True
                        
        if mention_text in third_female_person_pronouns:
            is_valid_female_third_person = True
                
        if mention_text in plural_third_person_pronouns:
            is_valid_plural_third_person = True
            
    return is_person, is_valid_first_person, is_valid_second_person, is_valid_male_third_person, is_valid_female_third_person, is_valid_plural_third_person, subject_start_pos_list, object_start_pos_list
  
def main():
    parser = argparse.ArgumentParser('Preprocess CoNLL')

    parser.add_argument('--input_file', 
                    type=str, 
                    default='original_conll_data/all_training.gold_conll', 
                    help='Input file.')
    parser.add_argument('--output_file',
                    type=str,
                    default='output_conll_data/training_data.txt',
                    help='Output file.')

    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()
    input_file = FLAGS.input_file
    output_file = FLAGS.output_file
    f = open(output_file, "w+", encoding="utf-8")
    conll_coref_ner_reader = ConllCorefNERReader(max_span_width=50)
    instances = conll_coref_ner_reader._read(input_file)

    """
    iterate each document
    """    
    COUNT = 0
    total_mention = 0
    total_entity = 0
    total_token = 0
    for instance in instances:
        ontonotes_sentences = conll_coref_ner_reader._ontonotes_sentence_set[COUNT]
        COUNT = COUNT + 1
        
        sentence_num_to_sentence_text = {}
        token_id_to_sentence_num = {}
        token_id_to_index_in_sentence = {}
        token_count = 0
        sentence_num = 0
        for ontonotes_sentence in ontonotes_sentences:
            tokens = [normalize_word(t) for t in ontonotes_sentence.words]
            sentence_text = ""
            for token in tokens:
                sentence_text = sentence_text + " " + token
            sentence_text = sentence_text.strip()
        
            token_index_in_sentence = 0
            for token in tokens:
                token_id_to_sentence_num[token_count] = sentence_num
                token_id_to_index_in_sentence[token_count] = token_index_in_sentence
                token_index_in_sentence = token_index_in_sentence + 1
                token_count = token_count + 1
             
            sentence_num_to_sentence_text[sentence_num] = sentence_text
            sentence_num = sentence_num + 1
        fields = instance.fields
        
        # Get fields
        text = [x.text for x in fields["text"].tokens]
        token_index_to_start_pos = {}
        start_pos_to_token_index = {}
        temp_pos = 0
        flat_text = ""
        for i in range(len(text)):
            token = text[i]
            flat_text = flat_text + token.lower() + " "
            token_index_to_start_pos[i] = temp_pos
            start_pos_to_token_index[temp_pos] = i
            temp_pos = len(flat_text)           
        flat_text = flat_text.strip()
        
        spans = fields["spans"].field_list
        span_starts, span_ends = zip(*[(field.span_start, field.span_end) for field in spans])
        
        tags = []
        sequences = fields["tags"].field_list
        for sequence in sequences:
            tags.extend(sequence.labels)
           
        postags = []
        sequences2 = fields["postags"].field_list
        for sequence2 in sequences2:
            postags.extend(sequence2.labels)
        
        metadata = fields["metadata"]
        gold_clusters = None
        if "clusters" in metadata:
            gold_clusters = metadata["clusters"]
        parse_trees = metadata['parsetrees'] 
        if gold_clusters is not None:
            start_pos_to_mention = {}
            start_pos_to_cluster_id = {}
            cluster_id_to_cluster = {}
            
            invalid_start_pos_to_mention = {}
            invalid_start_pos_to_cluster_id = {}
            invalid_cluster_id_to_cluster = {}
            
            sorted_start_pos_to_mention = {}
            sorted_start_pos_to_cluster_id = {}
            sorted_cluster_id_to_cluster = {}
            
            updated_invalid_start_pos_to_mention = {}
            updated_invalid_start_pos_to_cluster_id= {}
            updated_invalid_cluster_id_to_cluster = {}
            
            sorted_invalid_start_pos_to_mention = {}
            sorted_invalid_start_pos_to_cluster_id= {}
            sorted_invalid_cluster_id_to_cluster = {}
            
            all_subject_start_pos_list = []
            all_object_start_pos_list = []
            
            # Iterate clusters
            for cluster_id, cluster in enumerate(gold_clusters):
                is_person, is_valid_first_person, is_valid_second_person, is_valid_male_third_person, is_valid_female_third_person, is_valid_plural_third_person, subject_start_pos_list, object_start_pos_list =  iterate_cluster(cluster, text, tags, parse_trees, token_id_to_sentence_num, token_id_to_index_in_sentence)
                 
                all_subject_start_pos_list.extend(subject_start_pos_list)
                all_object_start_pos_list.extend(object_start_pos_list)
                
                first_person_cluster_ids = []
                second_person_cluster_ids = []
                male_third_person_cluster_ids = []
                female_third_person_cluster_ids = []
                plural_third_person_cluster_ids = []
                if is_valid_first_person and (cluster_id not in first_person_cluster_ids) and is_person:
                    first_person_cluster_ids.append(cluster_id)
                    
                if is_valid_second_person and (cluster_id not in second_person_cluster_ids) and is_person:
                    second_person_cluster_ids.append(cluster_id)
                    
                if is_valid_male_third_person and (cluster_id not in male_third_person_cluster_ids) and is_person:
                    male_third_person_cluster_ids.append(cluster_id)
                    
                if is_valid_female_third_person and (cluster_id not in female_third_person_cluster_ids) and is_person:
                    female_third_person_cluster_ids.append(cluster_id)
                   
                if is_valid_plural_third_person and (cluster_id not in plural_third_person_cluster_ids) and is_person:
                    plural_third_person_cluster_ids.append(cluster_id)
                  
                if (len(first_person_cluster_ids) > 0) or (len(second_person_cluster_ids) > 0):
                    continue
                
                if is_person:
                    if (not is_valid_first_person) and (not is_valid_second_person):
                        for mention in cluster:
                            mention_tuple = tuple(mention)
                            mention_start = mention_tuple[0]
                            mention_end = mention_tuple[1]
                            if mention_start in start_pos_to_mention:
                                exist_mention = start_pos_to_mention[mention_start]
                                exist_mention_tuple = tuple(exist_mention)
                                exist_mention_end = exist_mention_tuple[1]
                                if mention_end > exist_mention_end:
                                    start_pos_to_mention[mention_start] = mention
                                    start_pos_to_cluster_id[mention_start] = cluster_id
                                    cluster_id_to_cluster[cluster_id] = cluster
                            else:
                                start_pos_to_mention[mention_start] = mention
                                start_pos_to_cluster_id[mention_start] = cluster_id
                                cluster_id_to_cluster[cluster_id] = cluster
                    else:
                        for mention in cluster:
                            mention_tuple = tuple(mention)
                            mention_start = mention_tuple[0]
                            mention_end = mention_tuple[1]
                            if mention_start in invalid_start_pos_to_mention:
                                exist_mention = invalid_start_pos_to_mention[mention_start]
                                exist_mention_tuple = tuple(exist_mention)
                                exist_mention_end = exist_mention_tuple[1]
                                if mention_end > exist_mention_end:
                                    invalid_start_pos_to_mention[mention_start] = mention
                                    invalid_start_pos_to_cluster_id[mention_start] = cluster_id
                                    invalid_cluster_id_to_cluster[cluster_id] = cluster
                            else:
                                invalid_start_pos_to_mention[mention_start] = mention
                                invalid_start_pos_to_cluster_id[mention_start] = cluster_id
                                invalid_cluster_id_to_cluster[cluster_id] = cluster
            
            # Prune clusters
            for start_pos, mention in start_pos_to_mention.items():
                mention_tuple = tuple(mention)
                mention_start = mention_tuple[0]
                mention_end = mention_tuple[1]
                is_valid_mention = True
                for csp, cm in start_pos_to_mention.items():
                    cm_tuple = tuple(cm)
                    cm_start = cm_tuple[0]
                    cm_end = cm_tuple[1]
                    if mention_start == cm_start and mention_end == cm_end:
                        continue
                    elif mention_start >= cm_start and mention_end <= cm_end:
                        is_valid_mention = False
                        break
                if is_valid_mention:  
                    valid_cluster_id = start_pos_to_cluster_id[mention_start]
                    sorted_start_pos_to_mention[mention_start] = mention
                    sorted_start_pos_to_cluster_id[mention_start] = valid_cluster_id
                    sorted_cluster_id_to_cluster[valid_cluster_id] = cluster_id_to_cluster[valid_cluster_id]
            start_pos_to_mention = {}
            start_pos_to_cluster_id = {}
            cluster_id_to_cluster = {}
            
            for start_pos, mention in invalid_start_pos_to_mention.items():
                mention_tuple = tuple(mention)
                mention_start = mention_tuple[0]
                mention_end = mention_tuple[1]
                is_valid_mention = True
                for csp, cm in invalid_start_pos_to_mention.items():
                    cm_tuple = tuple(cm)
                    cm_start = cm_tuple[0]
                    cm_end = cm_tuple[1]
                    if mention_start == cm_start and mention_end == cm_end:
                        continue
                    elif mention_start >= cm_start and mention_end <= cm_end:
                        is_valid_mention = False
                        break
                if is_valid_mention:  
                    valid_cluster_id = invalid_start_pos_to_cluster_id[mention_start]
                    updated_invalid_start_pos_to_mention[mention_start] = mention
                    updated_invalid_start_pos_to_cluster_id[mention_start] = valid_cluster_id
                    updated_invalid_cluster_id_to_cluster[valid_cluster_id] = invalid_cluster_id_to_cluster[valid_cluster_id]
            invalid_start_pos_to_mention = {}
            invalid_start_pos_to_cluster_id = {}
            invalid_cluster_id_to_cluster = {}
            
            for start_pos, mention in updated_invalid_start_pos_to_mention.items():
                mention_tuple = tuple(mention)
                mention_start = mention_tuple[0]
                mention_end = mention_tuple[1]
                is_valid_mention = True
                for csp, cm in sorted_start_pos_to_mention.items():
                    cm_tuple = tuple(cm)
                    cm_start = cm_tuple[0]
                    cm_end = cm_tuple[1]
                    if mention_start >= cm_start and mention_start <= cm_end:
                        is_valid_mention = False
                        break
                    elif mention_start <= cm_start and mention_end >= cm_start:
                        is_valid_mention = False
                        break
                    
                if is_valid_mention:  
                    valid_cluster_id = updated_invalid_start_pos_to_cluster_id[mention_start]
                    sorted_invalid_start_pos_to_mention[mention_start] = mention
                    sorted_invalid_start_pos_to_cluster_id[mention_start] = valid_cluster_id
                    sorted_invalid_cluster_id_to_cluster[valid_cluster_id] = updated_invalid_cluster_id_to_cluster[valid_cluster_id]
            updated_invalid_start_pos_to_mention = {}
            updated_invalid_start_pos_to_cluster_id= {}
            updated_invalid_cluster_id_to_cluster = {}
            
            # Generate preprocessing results
            cluster_id_to_mentions = {}
            for cluster_id, cluster in sorted_cluster_id_to_cluster.items():
                mention_to_count = {}
                mention_to_postags = {}
                for mention in cluster:
                    mention_tuple = tuple(mention)
                    mention_start = mention_tuple[0]
                    mention_end = mention_tuple[1]
                    if mention_start not in sorted_start_pos_to_mention:
                        continue
                    else:
                        exist_mention = sorted_start_pos_to_mention[mention_start]
                        exist_mention_tuple = tuple(exist_mention)
                        exist_mention_end = exist_mention_tuple[1]
                        if mention_end < exist_mention_end:
                            continue
                    mention_text_span = text[mention_start:mention_end+1]
                    postag_subset = postags[mention_start: mention_end + 1]
                    
                    mention_text = ""
                    for m in mention_text_span:
                        mention_text = mention_text + " " + m
                    mention_text = mention_text.strip()
                    mention_text = mention_text.lower()
                    
                    if mention_text not in mention_to_count:
                        mention_to_count[mention_text] = 1
                        mention_to_postags[mention_text] = postag_subset
                    else:
                        mention_to_count[mention_text] = mention_to_count[mention_text] + 1
                  
                cluster_id_to_mentions[cluster_id] = mention_to_count.keys()
                f.write("<cluster_id>: ")
                f.write(str(cluster_id))
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
                
                for mention, postag_subset in mention_to_postags.items():
                    f.write("<mention>: ")
                    f.write(mention)
                    f.write("\t")
                    f.write("<postag>: ")
                    for postag in postag_subset:
                        f.write("%s " %postag)
                    f.write("\t")
                f.write("\n")
            
            for cluster_id, cluster in sorted_invalid_cluster_id_to_cluster.items():
                mention_to_count = {}
                mention_to_postags = {}
                for mention in cluster:
                    mention_tuple = tuple(mention)
                    mention_start = mention_tuple[0]
                    mention_end = mention_tuple[1]
                    if mention_start not in sorted_invalid_start_pos_to_mention:
                        continue
                    else:
                        exist_mention = sorted_invalid_start_pos_to_mention[mention_start]
                        exist_mention_tuple = tuple(exist_mention)
                        exist_mention_end = exist_mention_tuple[1]
                        if mention_end < exist_mention_end:
                            continue
                    mention_text_span = text[mention_start:mention_end+1]
                    postag_subset = postags[mention_start: mention_end + 1]
                    mention_text = ""
                    for m in mention_text_span:
                        mention_text = mention_text + " " + m
                    mention_text = mention_text.strip()
                    mention_text = mention_text.lower()
                    
                    if mention_text not in mention_to_count:
                        mention_to_count[mention_text] = 1
                        mention_to_postags[mention_text] = postag_subset
                    else:
                        mention_to_count[mention_text] = mention_to_count[mention_text] + 1

                f.write("<invalid_cluster_id>: ")
                f.write(str(cluster_id))
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

                for mention, postag_subset in mention_to_postags.items():
                    f.write("<mention>: ")
                    f.write(mention)
                    f.write("\t")
                    f.write("<postag>: ")
                    for postag in postag_subset:
                        f.write("%s " %postag)
                    f.write("\t")
                f.write("\n")
            
            sorted_all_pos_list = []
            sorted_all_pos_list.extend(sorted_start_pos_to_mention.keys())
            sorted_all_pos_list.extend(sorted_invalid_start_pos_to_mention.keys())
            
            sorted_start_pos_list = sorted(sorted_all_pos_list)
            
            if len(sorted_start_pos_to_mention) > 0:
                total_entity = total_entity + len(sorted_cluster_id_to_cluster)
                total_token = total_token + len(token_id_to_index_in_sentence)
                
            total_mention = total_mention + len(sorted_start_pos_to_mention)
            for sorted_start_pos in sorted(sorted_start_pos_to_mention.keys()):
                COUNT_INDEX = sorted_start_pos_list.index(sorted_start_pos)
                if COUNT_INDEX == -1:
                    print("There is an error in the dataset.")
                focus_mention = sorted_start_pos_to_mention[sorted_start_pos]
                focus_mention_cluster_id = sorted_start_pos_to_cluster_id[sorted_start_pos]
                
                focus_mention_tuple = tuple(focus_mention)
                focus_mention_start = focus_mention_tuple[0]
                focus_mention_end = focus_mention_tuple[1]                 
                focus_mention_text_span = text[focus_mention_start:focus_mention_end+1]
                focus_mention_sentence = token_id_to_sentence_num[focus_mention_start]
                focus_mention_index_in_sentence = token_id_to_index_in_sentence[focus_mention_start]
                    
                f.write("<focus_mention>: ")
                for word in focus_mention_text_span:
                    f.write("%s " %word)
                f.write("\t")
                f.write("<cluster_id>: ")
                f.write(str(focus_mention_cluster_id))
                f.write("\t")
                f.write("<sentence_num>: ")
                f.write(str(focus_mention_sentence))
                f.write("\t")
                f.write("<index_in_sentence>: ")
                f.write(str(focus_mention_index_in_sentence))
                f.write('\t')
                f.write("<is_subject>: ")
                if focus_mention_start in all_subject_start_pos_list:
                    f.write('1')
                else:
                    f.write('0')
                f.write('\t')
                f.write('<is_object>: ')
                if focus_mention_start in all_object_start_pos_list:
                    f.write('1')
                else:
                    f.write('0')
                f.write("\n")
                
                start_index = -1
                end_index = -1
                pre_padding = 0
                post_padding = 0
                if focus_mention_start >= 50 and focus_mention_end + 50 <= len(text) - 1:
                    start_index = focus_mention_start - 50
                    end_index = focus_mention_end + 50
                elif focus_mention_start < 50 and focus_mention_end + 50 <= len(text) - 1:
                    start_index= 0
                    end_index = focus_mention_end + 50
                    pre_padding = 50 - focus_mention_start
                elif focus_mention_start >= 50 and focus_mention_end + 50 > len(text) - 1:
                    start_index = focus_mention_start - 50
                    end_index = len(text) - 1
                    post_padding = 51 - len(text) + focus_mention_end
                elif focus_mention_start < 50 and focus_mention_end + 50 > len(text) - 1:
                    start_index = 0
                    end_index = len(text) - 1
                    pre_padding = 50 - focus_mention_start
                    post_padding = 51 - len(text) + focus_mention_end
                    
                related_mentions = []
                for i in range(start_index, end_index+1):
                    if i in sorted_start_pos_to_mention:
                        related_mention = sorted_start_pos_to_mention[i]
                        related_mention_cluster_id = sorted_start_pos_to_cluster_id[i]
                        if i != focus_mention_start:                                                      
                            related_mention_tuple = tuple(related_mention)
                            related_mention_start = related_mention_tuple[0]
                            related_mention_end = related_mention_tuple[1]
                            updated_related_mention_start = related_mention_start - start_index + pre_padding
                            updated_related_mention_end = related_mention_end - start_index + pre_padding
                            if related_mention_end <= end_index and related_mention_end >= start_index:
                                related_mentions.append(i)
                                f.write("<related_mention>: ")
                                f.write(str(updated_related_mention_start))
                                f.write(" ")
                                f.write(str(updated_related_mention_end))
                                f.write("\t")
                                f.write("<cluster_id>: ")
                                f.write(str(related_mention_cluster_id))
                                f.write("\n")
                if len(related_mentions) == 0:
                    f.write("<no related mentions>\n")
                                        
                result_sequence = text[start_index:end_index+1]
                result_sequence_postag = postags[start_index:end_index+1] 
                if pre_padding != 0:
                    for i in range(pre_padding):
                        result_sequence.insert(0, "<pad>")
                        result_sequence_postag.insert(0, "<NONE>")
                if post_padding != 0:
                    for i in range(post_padding):
                        result_sequence.append("<pad>")
                        result_sequence_postag.append("<NONE>")
                
                f.write("<raw_sequence>: ")
                for rs in result_sequence:
                    if rs != "\n":
                        f.write("%s " %rs)
                    else:
                        f.write("<\n> ")
                f.write("\n")
                f.write("<postags>: ")
                for rsp in result_sequence_postag:
                    if rsp != "\n":
                        f.write("%s " %rsp)
                    else:
                        f.write("<\n> ")
                f.write("\n")
                
                start_sentence_num = token_id_to_sentence_num[start_index]
                end_sentence_num = token_id_to_sentence_num[end_index]
                start_token_index_in_sentence = token_id_to_index_in_sentence[start_index]
                end_token_index_in_sentence = token_id_to_index_in_sentence[end_index]
                
                for i in range(start_sentence_num, end_sentence_num+1):
                    sentence_text = sentence_num_to_sentence_text[i]
                    f.write("<sentence_num>: ")
                    f.write(str(i))
                    f.write("\t")
                    f.write("<sentence_text>: ")
                    f.write(sentence_text)
                    f.write("\n")
                f.write("<start_token_index_in_sentence>: ")
                f.write(str(start_token_index_in_sentence))
                f.write("\t")
                f.write("<end_token_index_in_sentence>: ")
                f.write(str(end_token_index_in_sentence))
                f.write("\n")
                
                mention_sequence_start_index= -1
                mention_sequence_end_index = -1
                mention_sequence_pre_padding = 0
                mention_sequence_post_padding = 0
                if COUNT_INDEX >= 10 and COUNT_INDEX + 10 <= len(sorted_start_pos_list) - 1:
                    mention_sequence_start_index = COUNT_INDEX - 10
                    mention_sequence_end_index = COUNT_INDEX + 10
                elif COUNT_INDEX < 10 and COUNT_INDEX + 10 <= len(sorted_start_pos_list) - 1:
                    mention_sequence_start_index = 0
                    mention_sequence_end_index = COUNT_INDEX + 10
                    mention_sequence_pre_padding = 10 - COUNT_INDEX
                elif COUNT_INDEX >= 10 and COUNT_INDEX + 10 > len(sorted_start_pos_list) - 1:
                    mention_sequence_start_index = COUNT_INDEX - 10
                    mention_sequence_end_index = len(sorted_start_pos_list) - 1
                    mention_sequence_post_padding = 11 - len(sorted_start_pos_list) + COUNT_INDEX
                elif COUNT_INDEX < 10 and COUNT_INDEX + 10 > len(sorted_start_pos_list) - 1:
                    mention_sequence_start_index = 0
                    mention_sequence_end_index = len(sorted_start_pos_list) - 1
                    mention_sequence_pre_padding = 10 - COUNT_INDEX
                    mention_sequence_post_padding = 11 - len(sorted_start_pos_list) + COUNT_INDEX
                    
                pre_mention_sequence = []
                post_mention_sequence = []
                pre_mention_cluster_id_sequence = []
                post_mention_cluster_id_sequence = []
                pre_mention_distance_sequence = []
                post_mention_distance_sequence = []
                pre_mention_text_list = []
                pre_mention_token_id_list = []
                post_mention_text_list = []
                post_mention_token_id_list = []
                if mention_sequence_pre_padding != 0:
                    for i in range(mention_sequence_pre_padding):
                        pre_mention_sequence.append("<pad>")
                        pre_mention_cluster_id_sequence.append(-1)
                        pre_mention_distance_sequence.append(0)
                        
                if mention_sequence_start_index != -1 and mention_sequence_end_index != -1 and mention_sequence_start_index <= mention_sequence_end_index:
                    for i in range(mention_sequence_start_index, COUNT_INDEX+1):
                        X_start_pos = sorted_start_pos_list[i]
                        
                        X_mention = None
                        X_cluster_id = -1
                        
                        if X_start_pos in sorted_start_pos_to_mention:
                            X_mention = sorted_start_pos_to_mention[X_start_pos]
                            X_cluster_id = sorted_start_pos_to_cluster_id[X_start_pos]
                        else:
                            X_mention = sorted_invalid_start_pos_to_mention[X_start_pos]
                            X_cluster_id = sorted_invalid_start_pos_to_cluster_id[X_start_pos]
                            
                        X_mention_tuple = tuple(X_mention)
                        X_mention_start = X_mention_tuple[0]
                        X_mention_end = X_mention_tuple[1]                 
                        X_mention_text_span = text[X_mention_start:X_mention_end+1]
                                               
                        X_mention_text = ""
                        for X_m in X_mention_text_span:
                            X_mention_text = X_mention_text + " " + X_m
                        X_mention_text = X_mention_text.strip()
                        X_mention_text = X_mention_text.lower()
                        
                        pre_mention_text_list.append(X_mention_text)
                        pre_mention_token_id_list.append(X_mention_start)
                        
                        if i > 0:
                            X_start_pos_prev = sorted_start_pos_list[i-1]
                            X_mention_prev = None
                            if X_start_pos_prev in sorted_start_pos_to_mention:
                                X_mention_prev = sorted_start_pos_to_mention[X_start_pos_prev]
                            else:
                                X_mention_prev = sorted_invalid_start_pos_to_mention[X_start_pos_prev]
                            X_mention_prev_tuple = tuple(X_mention_prev)
                            X_mention_prev_end = X_mention_prev_tuple[1]
                            distance = X_mention_start - X_mention_prev_end - 1
                            pre_mention_distance_sequence.append(distance)
                        else:
                            pre_mention_distance_sequence.append(X_mention_start)
                            
                        pre_mention_sequence.append(X_mention_text)
                        pre_mention_cluster_id_sequence.append(X_cluster_id)
                    
                    last_mention_end = -1
                    for i in range(COUNT_INDEX+1, mention_sequence_end_index + 1):
                        X_start_pos = sorted_start_pos_list[i]
                        
                        X_mention = None
                        X_cluster_id = -1
                        
                        if X_start_pos in sorted_start_pos_to_mention:
                            X_mention = sorted_start_pos_to_mention[X_start_pos]
                            X_cluster_id = sorted_start_pos_to_cluster_id[X_start_pos]
                        else:
                            X_mention = sorted_invalid_start_pos_to_mention[X_start_pos]
                            X_cluster_id = sorted_invalid_start_pos_to_cluster_id[X_start_pos]
                            
                        X_mention_tuple = tuple(X_mention)
                        X_mention_start = X_mention_tuple[0]
                        X_mention_end = X_mention_tuple[1]                 
                        X_mention_text_span = text[X_mention_start:X_mention_end+1]                        
                        
                        X_mention_text = ""
                        for X_m in X_mention_text_span:
                            X_mention_text = X_mention_text + " " + X_m
                        X_mention_text = X_mention_text.strip()
                        X_mention_text = X_mention_text.lower()
                        
                        post_mention_text_list.append(X_mention_text)
                        post_mention_token_id_list.append(X_mention_start)
                        
                        if i == mention_sequence_end_index:
                            last_mention_end = X_mention_end
                            
                        if i > 0:
                            X_start_pos_prev = sorted_start_pos_list[i-1]
                            X_mention_prev = None
                            if X_start_pos_prev in sorted_start_pos_to_mention:
                                X_mention_prev = sorted_start_pos_to_mention[X_start_pos_prev]
                            else:
                                X_mention_prev = sorted_invalid_start_pos_to_mention[X_start_pos_prev]
                            X_mention_prev_tuple = tuple(X_mention_prev)
                            X_mention_prev_end = X_mention_prev_tuple[1]
                            distance = X_mention_start - X_mention_prev_end - 1
                            post_mention_distance_sequence.append(distance)
                        else:
                            post_mention_distance_sequence.append(0)
                            
                        post_mention_sequence.append(X_mention_text)
                        post_mention_cluster_id_sequence.append(X_cluster_id)
                        
                if last_mention_end == -1:
                    last_mention_end = focus_mention_end
                    
                if mention_sequence_post_padding != 0:
                    for i in range(mention_sequence_post_padding):
                        post_mention_sequence.append("<pad>")
                        post_mention_cluster_id_sequence.append(-1)
                        if i == 0:
                            post_mention_distance_sequence.append(len(text)-last_mention_end)
                        else:
                            post_mention_distance_sequence.append(0)
                        
                f.write("<pre_mention_sequence>: ")
                for X_mention in pre_mention_sequence:
                    f.write(X_mention)
                    f.write("\t")
                f.write("\n")
                
                for i in range(len(pre_mention_text_list)):
                    pre_mention_token_id = pre_mention_token_id_list[i]
                    pre_mention_index_in_sentence = token_id_to_index_in_sentence[pre_mention_token_id]
                    pre_mention_sentence_num = token_id_to_sentence_num[pre_mention_token_id]
                    pre_mention_sentence_text = sentence_num_to_sentence_text[pre_mention_sentence_num]
                    
                    f.write("<pre_mention_text>: ")
                    f.write(pre_mention_text_list[i])
                    f.write("\t")
                    f.write("<pre_mention_index_in_sentence>: ")
                    f.write(str(pre_mention_index_in_sentence))
                    f.write("\t")
                    f.write("<pre_mention_in_sentence>: ")                   
                    f.write(pre_mention_sentence_text)      
                    f.write("\t")
                    f.write("<pre_mention_sentence_num>: ")
                    f.write(str(pre_mention_sentence_num))
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
                
                f.write("<post_mention_sequence>: ")
                for X_mention in post_mention_sequence:
                    f.write(X_mention)
                    f.write("\t")
                f.write("\n")
                
                for i in range(len(post_mention_text_list)):
                    post_mention_token_id = post_mention_token_id_list[i]
                    post_mention_index_in_sentence = token_id_to_index_in_sentence[post_mention_token_id]
                    post_mention_sentence_num = token_id_to_sentence_num[post_mention_token_id]
                    post_mention_sentence_text = sentence_num_to_sentence_text[post_mention_sentence_num]
                    
                    f.write("<post_mention_text>: ")
                    f.write(post_mention_text_list[i])
                    f.write("\t")
                    f.write("<post_mention_index_in_sentence>: ")
                    f.write(str(post_mention_index_in_sentence))
                    f.write("\t")
                    f.write("<post_mention_in_sentence>: ")                 
                    f.write(post_mention_sentence_text)
                    f.write("\t")
                    f.write("<post_mention_sentence_num>: ")
                    f.write(str(post_mention_sentence_num))
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
    
    print('total entity: ')
    print(total_entity)
    print('total token: ')
    print(total_token)            
    print("total mention: ")
    print(total_mention)           
    f.close()
          
if __name__ == "__main__":   
    main()