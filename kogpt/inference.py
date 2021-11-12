import os
import logging
from typing import Optional, Union

import torch
from transformers import PreTrainedTokenizerFast
from transformers import GPTJForCausalLM


LOGGER = logging.getLogger(__name__)


class KoGPTInference:
    def __init__(self, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], device: str = 'cuda'):
        assert device in ('cuda', 'cpu')
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
            pretrained_model_name_or_path,
            bos_token='[BOS]', eos_token='[EOS]', unk_token='[UNK]', pad_token='[PAD]', mask_token='[MASK]'
        )
        LOGGER.debug('loaded tokenizer')

        model = GPTJForCausalLM.from_pretrained(
            '/data/project/rw/gpt/pretrained_models/6B-ryan1.5b',
            pad_token_id=self.tokenizer.eos_token_id,
            revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True
        )
        LOGGER.debug('loaded weights')

        model.eval()
        self.model = model.to(device=device)
        self.device = device

    def generate(self, prompt: str, temperature: float, max_length: int = 128) -> str:
        LOGGER.debug('prompt:%s, temperature:%f, max_length:%d', prompt, temperature, max_length)

        tokens = self.tokenizer.encode(prompt, return_tensors='pt').to(device=self.device, non_blocking=True)
        LOGGER.debug('prompt:%d tokens:%s', len(prompt), str(tokens.shape))

        gen_tokens = self.model.generate(tokens, do_sample=True, temperature=temperature, max_length=max_length)
        gen_text = self.tokenizer.batch_decode(gen_tokens)[0]
        LOGGER.debug('generated:%s', gen_text)
        return gen_text
