The programs in this folder are used to do mention selection for different models, which validate on the CoNLL data 
and test on the PoV dataset.

"train_model_f_m1_dev_conll_gold_test_three.py" is for the Only Token model.
"train_model_f_m2_dev_conll_gold_test_three.py" is for the Token + Mention model.
"train_model_f_m3_dev_conll_gold_test_three.py" is for the Only Mention model.
"train_model_f_m4_dev_conll_gold_test_three.py" is for the Token + Mention without binary features model.
"train_model_f_m5_dev_conll_gold_test_three.py" is for the Token + Mention without mention-level binary features model.
"train_model_f_m6_dev_conll_gold_test_three.py" is for the Token with token-level binary features model.

This program can be run in two mode: training and testing, you need to specify the mode, the default is training mode:
--is_training 1
In training mode, you do not need to specify the model path:
--loaded_model_path None
You need to specify the prefix to the training data (array of numbers of training examples and 
the default prefix is "output_training_array/conll_padding"). The training data is so big that we did not upload it. To generate 
the training data, you can go to the directory "preprocess/generate_training_data/" for detailed descriptions.
You also need to specify the directory of the development data (the default directory is "conll_data/dev/"), the directory of the 
testing data (the gold and auto setting[dev and test portion]), numer of epochs, batch size, learning rate, drop out rates, 
size of LSTM cell, size of hidden layer of FFN, margin for the ranking loss, patience for early stopping. 
For example, 
python train_model_f_m2_dev_conll_gold_test_three.py 
--conll_data_dir output_training_array/conll_padding
--development_data_dir conll_data/dev/
--pov_gold_testing_data_dir pov_data_gold/
--pov_auto_dev_data_dir pov_data_auto/dev/
--pov_auto_testing_data_dir pov_data_auto/test/
--loaded_model_path None
--is_training 1
--num_epochs 200
--batch_size 512
--learning_rate 12e-5
--dropout_rate_1 0.3
--dropout_rate_2 0.2
--lstm_size 50
--hidden_size 100
--margin 0.05
--patience 10
In testing mode, you do not need to specify the model path:
--loaded_model_path 'm2.pt'
You also need to specify the mode:
--is_training 0
You also need to specify the directory of the development data (the default directory is "conll_data/dev/"), the directory of the 
testing data (the gold and auto setting[dev and test portion]), margin for the ranking loss.
For example,
python train_model_f_m2_dev_conll_gold_test_three.py 
--development_data_dir conll_data/dev/
--pov_gold_testing_data_dir pov_data_gold/
--pov_auto_dev_data_dir pov_data_auto/dev/
--pov_auto_testing_data_dir pov_data_auto/test/
--loaded_model_path 'm2.pt'
--is_training 0
--margin 0.05