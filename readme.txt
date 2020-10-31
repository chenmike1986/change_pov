The programs in this folder are used to do mention selection for Only Token and Token + Mention models, under 
auto and gold settings, validated on the CoNLL and PoV dataset.

"train_model_f_m1_dev_conll_gold_test_three.py" and "train_model_f_m1_dev_pov_auto_test_pov_auto.py" are for the 
Only Token model.
"train_model_f_m2_dev_conll_gold_test_three.py" and "train_model_f_m2_dev_pov_auto_test_pov_auto.py" are for the 
Token + Mention model.

"train_model_f_m1_dev_conll_gold_test_three.py" and "train_model_f_m2_dev_conll_gold_test_three.py" validate on the 
CoNLL dataset, and test on CoNLL data, PoV data under both auto and gold setting.
"train_model_f_m1_dev_pov_auto_test_pov_auto.py" and "train_model_f_m2_dev_pov_auto_test_pov_auto.py" validate on 
the PoV dataset and test on PoV data under auto setting.

To run the program, you need to specify the prefix to the training data (array of numbers of training examples), 
directory of the development data, directory (directories) to the testing data, numer of epochs, batch size, learning rate, 
drop out rates, size of LSTM cell, size of hidden layer of FFN, margin for the ranking loss, patience for early stopping. 
For example, 
python train_model_f_m1_dev_conll_gold_test_three.py 
--conll_data_dir_scenario2 mention_predictor/data/preprocess/output_training_array/conll_padding
--development_data_dir mention_predictor/data/preprocess/preprocess_conll_data/output_conll_data/dev/
--conll_gold_testing_data_dir mention_predictor/data/preprocess/preprocess_conll_data/output_conll_data/test/
--pov_gold_testing_data_dir mention_predictor/data/preprocess/preprocess_pov_data/output_data/test_gold/
--pov_auto_testing_data_dir mention_predictor/data/preprocess/preprocess_pov_data/output_data/test_auto/
--num_epochs 200
--batch_size 512
--learning_rate 12e-5
--dropout_rate_1 0.3
--dropout_rate_2 0.2
--lstm_size 50
--hidden_size 100
--margin 0.05
--patience 10