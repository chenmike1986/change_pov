The programs in this folder are used to do mention selection for different models, which validate on the CoNLL data 
and test on the PoV dataset.

"train_model_f_m1_dev_conll_gold_test_three.py" is for the Only Token model.
"train_model_f_m2_dev_conll_gold_test_three.py" is for the Token + Mention model.
"train_model_f_m3_dev_conll_gold_test_three.py" is for the Only Mention model.
"train_model_f_m4_dev_conll_gold_test_three.py" is for the Token + Mention without binary features model.
"train_model_f_m5_dev_conll_gold_test_three.py" is for the Token + Mention without mention-level binary features model.
"train_model_f_m6_dev_conll_gold_test_three.py" is for the Token with token-level binary features model.

To run the program, you need to specify the prefix to the training data (array of numbers of training examples and 
the default prefix is "output/training_array/conll_padding", these training data files are generated from the previous step, 
please refer to the readme file in the folder "generate_training_data"), directory of the development data (the default directory is 
"conll_data/dev/"), directory of the testing data (the default directory is "pov_data/"), numer of epochs, batch size, 
learning rate, drop out rates, size of LSTM cell, size of hidden layer of FFN, margin for the ranking loss, patience for early stopping. 
For example, 
python train_model_f_m2_dev_conll_gold_test_three.py 
--conll_data_dir output_training_array/conll_padding
--development_data_dir conll_data/dev/
--pov_auto_testing_data_dir pov_data/
--num_epochs 200
--batch_size 512
--learning_rate 12e-5
--dropout_rate_1 0.3
--dropout_rate_2 0.2
--lstm_size 50
--hidden_size 100
--margin 0.05
--patience 10
