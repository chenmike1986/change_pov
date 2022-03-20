# PoV Change
This is a repository of the code and the data used for the experiments reported in the papers:

<a href="https://arxiv.org/abs/2103.04176">Changing the Narrative Perspective: From Deictic to Anaphoric Point of View</a>
Mike Chen and Razvan Bunescu, 
<a href="https://www.journals.elsevier.com/information-processing-and-management">Information Processing & Management</a>, Special Issue on Creative Language Processing, 2021.

Changing the Narrative Perspective: From Ranking to Prompt-Based Generation of Entity Mentions, Mike Chen and Razvan Bunescu, <a href="https://text2story22.inesctec.pt/">Proceedings of the Fifth International Workshop on Narrative Extraction from Texts (Text2Story)</a>, Stavanger, Norway, April 2022.

## Description

This project implements the tool for changing the narrative perspective, which consists of a text processing pipeline as described in <a href="https://arxiv.org/abs/2103.04176">Changing the Narrative Perspective: From Deictic to Anaphoric Point of View</a>: (1) mention identification and coreference resolution; (2) change of verb conjugation; (3) generation of candidate mention strings; (4) mention selection. Step (1) to (3) are implemented as preprocessing steps, which are in the folder "preprocess" and step (4) is accomplished by different models as described in the two papers mentioned above, which are in the folder "mention_selection".

## Getting Started

### Dependencies

cudatoolkit, pytorch, torchvision, transformers, tensorboard, allennlp, allennlp-models.

### Executing program

#### Preprocess

In the "preprocess" folder, it contains the pipeline components: identification of entity mentions and coreference resolution, change of verb conjugations and generation of candidate mention strings as described in the paper <a href="https://arxiv.org/abs/2103.04176">Changing the Narrative Perspective: From Deictic to Anaphoric Point of View</a>. 

In order to preprocess the PoV dataset, run the following command:

python preprocess_annotated_original_pov_data_auto.py

--input_dir pov_data/

--output_dir output_data_auto/

--focus_mention_string_dir focus_mention_string/

"input_dir" specifies the directory of the annotated PoV articles, "output_dir" specifies the directory of the output files and "focus_mention_string_dir" specifies the directory of mention string files, which contain names of the focus entities of the annotated articles.

#### Mention selection

In the "mention_selection" folder, it contains the pipeline component: mention selection.

There are 6 models in total, described in 
The results in Table 5 of the paper are updated. We fixed a bug related to the way dropout was used, which leads to slightly different results (overall better), as shown in the table below.

<p align="center">
<img src="https://github.com/chenmike1986/change_pov/blob/main/mention_selection/update_results.png" width="450" height="120">
</p>

### Dataset

The annotated pov data is in the folder "pov_data".
