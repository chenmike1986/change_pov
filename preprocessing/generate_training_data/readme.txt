In order to use following programs, you need to install pytorch (1.2.0), torchvision (0.4.0), cudatoolkit (10.0.130), pytorch-pretrained-bert (0.6.2).



"generate_training_data.py" is used to generate data (arrays of numbers for each training example) for training 
the mention selection system. 
"generate_training_data_m6.py" is used to generate data (arrays of numbers for each training example) for training 
the mention selection system (the Token with token-level binary features model) . 
To run the program, you need to specify the directory and the name of the CoNLL training data (the default directory
is "conll_data" and the default file name is "training_data.txt"; this file is a .rar file due to the permitted size by Github; to use it, you need to extract it).
You also need to provide the prefix of the output files (each output file contains 512 training examples and the default
prefix is "output_training_array/conll_padding"). 
For example, 
python generate_training_data.py
--conll_data_dir conll_data
--training_data training_data.txt
--output_file output_training_array/conll_padding
<<<<<<< HEAD
=======


"generate_training_data_transformer_2_1.py" is used to generate data (arrays of numbers for each training example) for training 
the mention selection system (the Transformer models) . 
To run the program, you need to specify the directory and the name of the CoNLL training data (the default directory
is "conll_data" and the default file name is "training_data_100.txt" (this file is a .rar file due to the permitted size by Github; to use it, you need to extract it).
You also need to provide the prefix of the output files (each output file contains 512 training examples and the default
prefix is "output_training_array_transformer/conll_padding"). 
For example, 
python generate_training_data_transformer_2_1.py
--conll_data_dir conll_data
--training_data training_data_100.txt
--output_file output_training_array_transformer/conll_padding

"generate_training_data_transformer_2_1_pov.py" is used to generate data (arrays of numbers for each training example) for fine-tuning 
the mention selection system (the Transformer models). 
To run the program, you need to specify the directory of the PoV data (the default directory
is "pov_data").
You also need to provide the prefix of the output files (each output file contains 512 training examples and the default
prefix is "output_training_array_transformer_pov/conll_padding"). 
For example, 
python generate_training_data_transformer_2_1_pov.py
--conll_data_dir pov_data
--output_file output_training_array_transformer_pov/conll_padding
>>>>>>> parent of c31f76c (Revert "Update readme.txt")
