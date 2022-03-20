In order to install the Spyder IDE:
1) You need to first install anaconda: https://docs.anaconda.com/anaconda/install/
2) Then it's better to create a separate environment using the anaconda prompt to run the program: https://medium.com/analytics-vidhya/4-steps-to-install-anaconda-and-pytorch-onwindows-10-5c9cb0c80dfe
3) In order to run the program, you need to install the following packages in the anaconda prompt: pytorch (1.2.0), torchvision (0.4.0), cudatoolkit (10.0.130), pytorch-pretrained-bert (0.6.2)
4) Then you can open the anaconda navigator, switch to the newly created environment, and click "install Spyder" in the user interface.
5) Now you can run and debug the code in the Spyder IDE.



The programs in this folder are used to do mention selection for different models, which validate on the CoNLL data 
and test on the CoNLL and PoV dataset.

"train_model_f_m1_dev_conll_gold_test_three.py" is for the Only Token model.
"train_model_f_m2_dev_conll_gold_test_three.py" is for the Token + Mention model.
"train_model_f_m3_dev_conll_gold_test_three.py" is for the Only Mention model.
"train_model_f_m4_dev_conll_gold_test_three.py" is for the Token + Mention without binary features model.
"train_model_f_m5_dev_conll_gold_test_three.py" is for the Token + Mention without mention-level binary features model.
"train_model_f_m6_dev_conll_gold_test_three.py" is for the Token with token-level binary features model.

This program can be run in two modes: training and testing, you need to specify the mode, the default is the testing mode (the training mode is 1):
--is_training 0 
In training mode, you do not need to specify the model path:
--loaded_model_path None
You need to specify the path and the prefix to the training data (array of numbers of training examples and 
the default prefix is "change_pov/preprocessing/generate_training_data/output_training_array/conll_padding"). The training data is so big that we did not upload it. To generate 
the training data, you can go to the directory "preprocessing/generate_training_data/" for detailed descriptions.
You also need to specify the directory of the development data (the default directory is "conll_data/dev/"), number of epochs, batch size, learning rate, drop out rates, 
size of LSTM cell, size of hidden layer of FFN, margin for the ranking loss, patience for early stopping. 
For example, 
python train_model_f_m2_dev_conll_gold_test_three.py 
--conll_data_dir change_pov/preprocessing/generate_training_data/output_training_array/conll_padding
--development_data_dir conll_data/dev/
--loaded_model_path None
--is_training 1
--num_epochs 200
--batch_size 512
--learning_rate 12e-5
--dropout_rate_1 0.3
--dropout_rate_2 0.2
--dropout_rate_3 0.5
--dropout_rate_4 0.5
--lstm_size 50
--hidden_size 100
--margin 0.05
--patience 10
In testing mode, you do need to specify the model path:
--loaded_model_path 'm2.pt'
You also need to specify the mode:
--is_training 0
You also need to specify the directory of the CoNLL testing data (the default directory is "conll_data/test/"), the directory of the 
PoV testing data (the gold and auto setting).
For example,
python train_model_f_m2_dev_conll_gold_test_three.py 
--development_data_dir conll_data/test/
--pov_gold_testing_data_dir pov_data_gold/
--pov_auto_testing_data_dir pov_data_auto/test/
--loaded_model_path 'm2.pt'
--is_training 0



In order to run the Transformer model, you need to install the following packages: cudatoolkit (10.2.89), pytorch (1.8.1), torchvision (0.9.1), transformers (4.5.0).


The program "train_transformer_dev_conll_gold_test_three.py" is used to do mention selection using the Transformer model, which validates on the CoNLL data 
and tests on the CoNLL and PoV dataset.

This program can be run in two modes: training and testing, you need to specify the mode, the default is the training mode (the testing mode is 0):
--is_training 1
In training mode, you do not need to specify the model path:
--loaded_model_path None
You need to specify the path and the prefix to the training data (array of numbers of training examples and 
the default prefix is "change_pov/preprocessing/generate_training_data/output_training_array_transformer/conll_padding"). The training data is so big that we only upload one training batch. To generate 
the training data, you can go to the directory "preprocessing/generate_training_data/" for detailed descriptions.
You also need to specify the directory of the development data (the default directory is "conll_data/dev_transformer/"), number of epochs, batch size, learning rate, drop out rates, margin for the ranking loss, patience for early stopping. 
For example, 
python train_transformer_dev_conll_gold_test_three.py 
--conll_data_dir change_pov/preprocessing/generate_training_data/output_training_array_transformer/conll_padding
--development_data_dir conll_data/dev_transformer/
--loaded_model_path None
--is_training 1
--num_epochs 200
--batch_size 512
--learning_rate 12e-5
--dropout_rate_1 0.1
--margin 0.1
--patience 10
In testing mode, you do need to specify the model path (the trained model is not available right now):
--loaded_model_path 'm_transformer.pt'
You also need to specify the mode:
--is_training 0
You also need to specify the directory of the CoNLL testing data (the default directory is "conll_data/test_transformer/"), the directory of the 
PoV testing data (the gold and auto setting).
For example,
python train_model_f_m2_dev_conll_gold_test_three.py 
--development_data_dir conll_data/test_transformer/
--pov_gold_testing_data_dir pov_data_transformer_gold/
--pov_auto_testing_data_dir pov_data_transformer_auto/test/
--loaded_model_path 'm_transformer.pt'
--is_training 0
