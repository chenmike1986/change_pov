In order to use following programs, you need to install pytorch (1.4.0), torchvision (0.5.0), cudatoolkit (10.1.243), allennlp (1.0.0rc3), allennlp-models (1.0.0rc3), pytorch-pretrained-bert (0.6.2).




The program "preprocess_annotated_original_pov_data_auto.py" in this folder is used to preprocess the PoV dataset. 
The preprocessing includes mention detection, coreference resolution, changing verb conjugation and generating S(E). 

To run the program, you need to specify the directory of the annotated PoV articles ( the default directory is "pov_data/"), 
the directory of output files, which will be used during evaluation ( the default directory is "output_data_auto/"), 
and the directory of mention string files, which contain names of the focus entities of the annotated articles (the default  
directory is "focus_mention_string/").
For example,
python preprocess_annotated_original_pov_data_auto.py 
--input_dir pov_data/ 
--output_dir output_data_auto/
--focus_mention_string_dir focus_mention_string/

"performatives.txt" is the dictionary which contains performative verbs.
"conjugations_english.tab" is the verb conjugation change dictionary.
"seed.tsv" is the relational noun dictionary.
