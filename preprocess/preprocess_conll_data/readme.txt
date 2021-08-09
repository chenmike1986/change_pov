The program in this folder is used to preprocess the CoNLL dataset - generate fomatted data required by the mention 
selection system.

To run the program, you need to specify the input file (one of the files in 
"original_conll_data" [training, development or testing]; the files are .rar files, due to the size permitted by the Github; to use them, you need extract these files) and 
the output file (can be put in the folder "output_conll_data"; the generated files are also .rar files due to the size permitted by the Github; to review them, you need to extract these files.). For example, 
python preprocess_conll_data.py 
--input_file original_conll_data/all_development.gold_conll
--output_file output_conll_data/development_data.txt
