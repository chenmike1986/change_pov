
import os

from torch.nn.parameter import Parameter
from openprompt.utils.logging import logger



from openprompt.data_utils import InputExample, InputFeatures
from typing import *

from transformers import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from my_prompt_base import Template
from openprompt.prompts import ManualTemplate, ManualVerbalizer

import torch
from torch import nn

class SoftTemplate(Template):
    r"""This is the implementation of `The Power of Scale for Parameter-Efficient
    Prompt Tuning <https://arxiv.org/pdf/2104.08691v1.pdf>`_ . Similar to :obj:`PrefixTuningTemplate`,
    This template also does not need any textual template. Addition tokens are directly
    concatenated into the input ids. There are two initializations of the new tokens. 
    (1). random initialization. (2) initialize with the tokens of the plm (We simply take 
    the first n_tokens similar to their implementation).

    Note that this template can be simply achieved by :obj:`SoftManualTemplate`, in which
    you set `n_token` <soft> tokens template before the <text_a> will give the same result.
    """
    registered_inputflag_names = ["soft_token_ids", "loss_ids", "shortenable_ids"]

    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 text: Optional[str] = None,
                 mask_token: str = '<mask>',
                 num_tokens: int=20,
                 initialize_from_vocab: Optional[bool] = True,
                 random_range: Optional[float] = 0.5,
                 placeholder_mapping: dict = {'<text_a>':'text_a','<text_b>':'text_b'},
                ):
        super().__init__(tokenizer=tokenizer,
                         mask_token=mask_token,
                         placeholder_mapping=placeholder_mapping)
        self.raw_embedding = model.get_input_embeddings()
        self.model_is_encoder_decoder = model.config.is_encoder_decoder
        self.random_range = random_range
        self.num_tokens = num_tokens
        self.initialize_from_vocab = initialize_from_vocab

        self.embedding_size = self.raw_embedding.weight.shape[-1]
        self.num_soft_token = 5
        self.text = text 
        # self.default_text1 = {"placeholder<text_a> <mask>"
        # self.default_text2 = "<text_a> <text_b> <mask>".split()

        if self.num_tokens>0:
            self.generate_parameters()

    def get_default_soft_token_ids(self) -> List[int]:
        return self.soft_token_ids    

    def prepare(self):
        r"""get the trainable token indices for the template
        
        ``"soft_id"`` can be used to reference the previous soft token, which means these tokens use the same embeddings.
        **Note that ``"soft_id"`` should have index start from 1 but not 0**

        e.g. when self.text is ``'{"soft": None} {"soft": "the", "soft_id": 1} {"soft": None} {"soft": "it", "soft_id": 3} {"soft_id": 1} {"soft": "was"} {"mask"}'``,
        output is [1, 2, 3, 4, 2, 5, 0]

        TODO document here
        """
        num_soft_token = 0
        text = []
        soft_token_ids = []
        idx_mp = {}
        emb_mp = {}
        for d in self.text:
            if "soft" not in d and "soft_id" not in d:
                text.append(d)
                soft_token_ids.append(0)
                continue

            old_num = num_soft_token

            if "soft_id" in d:
                if not isinstance(d["soft_id"], int) or d["soft_id"] <= 0:
                    raise ValueError(f'soft_id should be integer greater than zero, but get {d["soft_id"]}')
                # if d["soft_id"] in idx_mp:
                    #id_list = idx_mp[d["soft_id"]]
                id_list = [d["soft_id"]]
                text.extend([{"soft":""} for _ in range(len(id_list))])
                soft_token_ids.extend(id_list)
                continue
                # else:
                #     if "soft" not in d: d["soft"] = None

            # if d["soft"] is None:
            #     if "duplicate" in d:
            #         if "same" in d and d["same"]:
            #             num_soft_token += 1
            #             id_list = [num_soft_token for _ in range(len(d["duplicate"]))]
            #         else:
            #             num_soft_token += d["duplicate"]
            #             id_list = list(range(old_num+1, num_soft_token+1))
            #     else:
            #         num_soft_token += 1
            #         id_list = [num_soft_token]
            #     text.extend([{"soft":""} for _ in range(len(id_list))])
            # else:
            #     token_ids = self.tokenizer(d["add_prefix_space"] + d["soft"], add_special_tokens=False)["input_ids"]
            #     surface_forms = self.tokenizer.convert_ids_to_tokens(token_ids)
            #     assert len(token_ids) == len(surface_forms)
            #     num_soft_token += len(token_ids)
            #     id_list = list(range(old_num+1, num_soft_token+1))
            #     for idx, soft_id in enumerate(id_list):
            #         emb_mp[soft_id] = token_ids[idx]

            #     text.extend([{"soft": surface_form} for surface_form in surface_forms])
            #soft_token_ids.extend(id_list)

            # if "soft_id" in d:
            #     idx_mp[d["soft_id"]] = id_list

        #self.num_soft_token = num_soft_token
        self.text = text
        self.soft_token_ids = soft_token_ids        

        if "post_processing" in d:
            if d["post_processing"] == "mlp":
                pass # TODO one mlp or more than one
            else:
                raise ValueError(f'post_processing of {d["post_processing"]} is not supported yet')
                
    def on_text_set(self):
        self.text = self.parse_text(self.text)
        self.prepare()

    def wrap_one_example(self, example) -> List[Dict]:  #TODO this automatic generated template may not be able to process diverse data format.
        if self.text is None:
            logger.warning("You didn't provide text templat efor softprompt. Using default template, is this intended?")
            # if example.text_b is None:
            #     self.text = self.default_text1
            # else:
            #     self.text = self.default_text2
        return super().wrap_one_example(example)


    def generate_parameters(self) -> None:
        """
        generate parameters needed for soft tokens embedding in soft-prompt
        for soft tokens, use a new embedding layer which is initialized with their corresponding embedding of hard tokens
        """
        if self.initialize_from_vocab:
            soft_embeds = self.raw_embedding.weight[:self.num_tokens].clone().detach()
        else:
            soft_embeds = torch.FloatTensor(self.num_tokens, self.raw_embedding.weight.size(1)).uniform_(-self.random_range, self.random_range)
        self.soft_embeds = nn.Parameter(soft_embeds, requires_grad=True)

        # Generate the embedding needed for soft tokens

        self.soft_embedding = nn.Embedding(1 + self.num_soft_token, self.embedding_size)
        # for soft_id, token_id in emb_mp.items():
        #     self.soft_embedding.weight.data[soft_id, :] = self.raw_embedding.weight.data[token_id, :].clone().detach().requires_grad_(True)
    
    def process_batch(self, batch: Union[Dict, InputFeatures]) -> Union[Dict, InputFeatures]:
        """
        Convert input_ids to inputs_embeds
        for normal tokens, use the embedding layer of PLM
        for soft tokens, use a new embedding layer which is initialized with their corresponding embedding of hard tokens
        """

        #try:
        raw_embeds = self.raw_embedding(batch['input_ids'])
        #except:
            #print(batch)
            # print(batch['tgt_text'])
            # print(batch['guid'])
        soft_embeds_1 = self.soft_embedding(batch['soft_token_ids'])
        inputs_embeds = torch.where((batch['soft_token_ids'] > 0).unsqueeze(-1), soft_embeds_1, raw_embeds)
        batch_size = inputs_embeds.size(0)
        if self.num_tokens>0:
            soft_embeds = self.soft_embeds.repeat(batch_size, 1, 1)
            inputs_embeds = torch.cat([soft_embeds, inputs_embeds], 1)

        batch['input_ids'] = None
        batch['inputs_embeds'] = inputs_embeds
        if 'attention_mask' in batch and self.num_tokens>0:
            am = batch['attention_mask']
            batch['attention_mask'] = torch.cat([torch.ones((batch_size,self.num_tokens), dtype = am.dtype,device=am.device), am], dim=-1)
        return batch


    def post_processing_outputs(self, outputs: torch.Tensor):
        r"""Post processing the outputs of language models according
        to the need of template. Most templates don't need post processing,
        The template like SoftTemplate, which appends soft template as a module
        (rather than a sequence of input tokens) to the input,
        should remove the outputs on these positions to keep the seq_len the same
        """
        if not self.model_is_encoder_decoder:
            outputs.logits = outputs.logits[:, self.num_tokens:,: ]
        return outputs
