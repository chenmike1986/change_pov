This is a repository of the code and the data used for the experiments reported in the paper:
<a href="https://arxiv.org/abs/2103.04176">Changing the Narrative Perspective: From Deictic to Anaphoric Point of View</a>
Mika Chen and Razvan Bunescu, 
<a href="https://www.journals.elsevier.com/information-processing-and-management">Information Processing & Management</a>, Special Issue on Creative Language Processing, 2021.



This project includes two parts: preprocess and mention selection.

The usage of each component is described in the corresponding folder.

The order to use each component is as follows: preprocess, mention selection.

The annotated pov data is in the folder "pov_data".

The results in Table 5 of the paper are updated. We fixed a bug related to the way dropout was used, which leads to slightly different results (overall better), as shown in the table below.

<img src="https://github.com/chenmike1986/change_pov/blob/main/mention_selection/update_results.png" width="400" height="100">
