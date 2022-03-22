# PoV Change
This is a repository of the code and the data used for the experiments reported in the papers:

<a href="https://arxiv.org/abs/2103.04176">Changing the Narrative Perspective: From Deictic to Anaphoric Point of View</a>,
Mike Chen and Razvan Bunescu, 
<a href="https://www.journals.elsevier.com/information-processing-and-management">Information Processing & Management</a>, Special Issue on Creative Language Processing, 2021.

Changing the Narrative Perspective: From Ranking to Prompt-Based Generation of Entity Mentions, Mike Chen and Razvan Bunescu, <a href="https://text2story22.inesctec.pt/">Proceedings of the Fifth International Workshop on Narrative Extraction from Texts (Text2Story)</a>, Stavanger, Norway, April 2022.

## Description

This project implements the tool for changing the narrative perspective, which consists of a text processing pipeline as described in <a href="https://arxiv.org/abs/2103.04176">Changing the Narrative Perspective: From Deictic to Anaphoric Point of View</a>: (1) mention identification and coreference resolution; (2) change of verb conjugation; (3) generation of candidate mention strings; (4) mention selection. Step (1) to (3) are implemented as preprocessing steps, which are in the folder "preprocess" and step (4) is accomplished by different models as described in the two papers mentioned above, which are in the folder "mention_selection".

## Getting Started

### Dependencies

cudatoolkit, pytorch, torchvision, transformers, tensorboard, allennlp, allennlp-models, openprompt.

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

In the "mention_selection" folder, it contains the pipeline component: mention selection. The mention selection part is accomplished by different models, from ranking to prompt-based generation.

##### The ranking models 

The ranking models are implemented by:

file  | model | trained model | validation data | 
------------- | ------------- | ------------- | ------------- |
train_model_f_m1_dev_conll_gold_test_three.py  | Token LSTM  | m1.pt  | conll_data/dev/ |
train_model_f_m2_dev_conll_gold_test_three.py  | Token + Mention LSTM | m2.pt  | conll_data/dev/ |
train_model_lstm_attention.py  | LSTM-attention | m_lstm_attention.pt | conll_data/dev/ |
train_transformer_2_1.py  | Coreference-modulated self-attention | m_transformer.pt | conll_data/dev_transformer/ |

The testing data information are listed:
 model | CoNLL test data | PoV mention selection data | PoV end-to-end data|
------------- | ------------- | ------------- | ------------- |
Token LSTM  | conll_data/test/  | pov_data_gold/ | pov_data_auto/ |
Token + Mention LSTM | conll_data/test/  | pov_data_gold/ | pov_data_auto/ |
LSTM-attention | conll_data/test/ | pov_data_gold/ | pov_data_auto/ |
Coreference-modulated self-attention | conll_data/test_transformer/ | pov_data_transformer_gold/ | pov_data_transformer_auto/ |

These programs can be run in two modes: training and testing, you need to specify the mode, the default is the testing mode (the training mode is 1):

--is_training 0 

In training mode, you do not need to specify the model path:

--loaded_model_path None

These models were trained on CoNLL dataset, which is too big to upload to Github, and anyone who is interested in the training data can contact the author directly by "mc277509@ohio.edu".

In order to run the program, you need to specify the path and the prefix to the training data, the directory of the development data when running in the training mode, the directory of the CoNLL and PoV testing data when running in the testing mode, number of epochs, batch size, learning rate, drop out rates, size of LSTM cell, size of hidden layer of FFN, margin for the ranking loss, patience for early stopping. 

For example, 

python train_model_lstm_attention.py 

--conll_data_dir 'change_pov/preprocessing/generate_training_data/output_training_array/conll_padding'

--development_data_dir 'conll_data/dev/'

--pov_gold_testing_data_dir 'pov_data_gold/'

--pov_auto_testing_data_dir 'pov_data_auto/test/'

--loaded_model_path None

--is_training 1

--num_epochs 200

--batch_size 512

--learning_rate 12e-5

--dropout_rate_1 0.01

--dropout_rate_2 0.01

--dropout_rate_3 0.1

--dropout_rate_4 0.1

--lstm_size 50

--hidden_size 100

--margin 0.05

In testing mode, you also need to specify the model path:

--loaded_model_path 'm_lstm_attention.pt'

***Note: The trained model m_transformer.pt is too big to upload, and anyone who is interested in getting the trained model can contact the author directly by mc277509@ohio.edu

##### The prompt-based models 

The prompt-based models are implemented by:

file  | model | trained model | validation data | 
------------- | ------------- | ------------- | ------------- |
train_plm_few_shot_learning_auto_regressive_without_entities.py  | Prompt-tuning with pre-trained T5 | plm-prompt-based.pt  | conll_data/dev/ |
train_plm_few_shot_learning_auto_regressive_without_entities.py  | Prompt-tuning with fine-tuned T5 | plm-fine-tune.pt  | conll_data/dev/ |

The testing data information are listed:
 model | CoNLL test data | PoV mention selection data | PoV end-to-end data|
------------- | ------------- | ------------- | ------------- |
Prompt-tuning with pre-trained T5 | conll_data/test/  | pov_data_gold/ | pov_data_auto/ |
Prompt-tuning with fine-tuned T5 | conll_data/test/  | pov_data_gold/ | pov_data_auto/ |

These programs can be run in two modes: training and testing, you need to specify if the current mode is training:

--is_training True 

In training mode, you do not need to specify the model path:

--model_path None

These models were trained on CoNLL dataset, which is in the folder "conll_json".

In order to run the program, you need to specify the directory of the training data, the directory of the development data when running in the training mode, the directory of the CoNLL and PoV testing data when running in the testing mode, learning rate and number of soft tokens. You also need to specify if the model is running in the pre-trained T5 or fine-tuned T5 setting: 

--tune_plm False

Then a complete running command could be:

python train_plm_few_shot_learning_auto_regressive_without_entities.py

--tune_plm False

--model_name_or_path 't5/'

--data_dir 'conll_json/'

--dev_data_dir 'conll_data/dev/'

--conll_test_data_dir 'conll_data/test/'

--pov_gold_test_data_dir 'pov_data_gold/'

--pov_auto_test_data_dir 'pov_data_auto/test/'

--loaded_model_path None

--is_training True

--prompt_lr 0.3

--soft_token_num 20

In testing mode, you also need to specify the model path:

--loaded_model_path 'plm-prompt-based.pt'

### Dataset

The annotated pov data is in the folder "pov_data".

## Miscellaneous

The results in Table 5 of the paper <a href="https://arxiv.org/abs/2103.04176">Changing the Narrative Perspective: From Deictic to Anaphoric Point of View</a> are updated. We fixed a bug related to the way dropout was used, which leads to slightly different results (overall better), as shown in the table below.

<p align="center">
<img src="https://github.com/chenmike1986/change_pov/blob/main/mention_selection/update_results.png" width="450" height="120">
</p>
