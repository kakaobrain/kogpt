import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from fastapi import FastAPI
import uvicorn
from typing import Optional

import time
app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained(
  'kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b',
  bos_token='[BOS]', eos_token='[EOS]', unk_token='[UNK]', pad_token='[PAD]', mask_token='[MASK]'
)
model = AutoModelForCausalLM.from_pretrained(
  'kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b',
  pad_token_id=tokenizer.eos_token_id,
  torch_dtype=torch.float16, low_cpu_mem_usage=True
).to(device='cuda', non_blocking=True)
_ = model.eval()

@app.post("/generate")
async def generate(
    context: Optional[
        str
    ] = "아이유같은 노래 가사를 만들어줘",
    token_max_length: Optional[int] = 64,
    temperature: Optional[float] = 1.0,
    top_p: Optional[float] = 0.9,
    stop_sequence: Optional[str] = None,
):
    start = time.time()

    with torch.no_grad():
        tokens = tokenizer.encode(context, return_tensors='pt').to(device='cuda', non_blocking=True)
        gen_tokens = model.generate(tokens, do_sample=True, temperature=temperature, max_length=token_max_length)
        generated = tokenizer.batch_decode(gen_tokens)[0]

    text = generated
    provided_ctx = len(tokens)
    if token_max_length + provided_ctx > 2048:
        return {"text": "Don't abuse the API, please."}

    response = {}
    response["model"] = "GPT-J-6B"
    response["compute_time"] = time.time() - start
    response["text"] = text
    response["prompt"] = context
    response["token_max_length"] = token_max_length
    response["temperature"] = temperature
    response["top_p"] = top_p
    response["stop_sequence"] = stop_sequence

    return response

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=5000)