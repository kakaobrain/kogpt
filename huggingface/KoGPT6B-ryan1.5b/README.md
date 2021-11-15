---
language: ko
tags:
- KakaoBrain
- KoGPT
- GPT
- GPT3
license: cc-by-nc-nd-4.0
---

# KoGPT
[![KakaoBrain](https://img.shields.io/badge/Kakao-Brain-ffcd00.svg)](http://kakaobrain.com/)
[![Github: kogpt](https://img.shields.io/badge/Github-kogpt-000000.svg)](https://github.com/kakaobrain/kogpt)
[![huggingface: KoGPT-6B](https://img.shields.io/badge/huggingface-KoGPT_6B_ryan1.5b-ffcd00.svg)](https://huggingface.co/kakaobrain/kogpt/tree/KoGPT6B-ryan1.5b)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)


KakaoBrain's Pre-Trained Language Models. 

* KoGPT (Korean Generative Pre-trained Transformer)
  * [https://github.com/kakaobrain/kogpt](https://github.com/kakaobrain/kogpt)
  * [https://huggingface.co/kakaobrain/kogpt](https://huggingface.co/kakaobrain/kogpt)


## Model Descriptions

### KoGPT6B-ryan1.5b

* [\[huggingface\]\[kakaobrain/kogpt\]\[KoGPT6B-ryan1.5b\]](https://huggingface.co/kakaobrain/kogpt/tree/KoGPT6B-ryan1.5b)

| Hyperparameter       | Value         |
|:---------------------|--------------:|
| \\(n_{parameters}\\) | 6,166,502,400 |
| \\(n_{layers}\\)     | 28            |
| \\(d_{model}\\)      | 4,096         |
| \\(d_{ff}\\)         | 16,384        |
| \\(n_{heads}\\)      | 16            |
| \\(d_{head}\\)       | 256           |
| \\(n_{ctx}\\)        | 2,048         |
| \\(n_{vocab}\\)      | 64,512        |
| Positional Encoding  | [Rotary Position Embedding (RoPE)](https://arxiv.org/abs/2104.09864) |
| RoPE Dimensions      | 64            |


## Hardware requirements

### GPU
The following is the recommended minimum GPU hardware guidance for a handful of example KoGPT.
* half-precision requires NVIDIA GPUS based on Volta, Turing or Ampere
* 32GB GPU RAM in the required minimum memory size


## Usage

### prompt
```bash
python -m kogpt --help
usage: KoGPT inference [-h] [--model MODEL] [--revision {KoGPT6B-ryan1.5b}]
                       [--device {cpu,cuda}] [-d]

KakaoBrain Korean(hangul) Generative Pre-Training Model

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         huggingface repo (default:kakaobrain/kogpt)
  --revision {KoGPT6B-ryan1.5b}
  --device {cpu,cuda}   (default:cuda)
  -d, --debug
```

```bash
python -m kogpt
prompt> 인간처럼 생각하고, 행동하는 '지능'을 통해 인류가 이제까지 풀지 못했던
temperature(0.8)> 
max_length(128)> 64
인간처럼 생각하고, 행동하는 '지능'을 통해 인류가 이제까지 풀지 못했던 문제의 해답을 찾을 수 있을 것이다. 과학기술이 고도로 발달한 21세기를 살아갈 우리 아이들에게 가장 필요한 것은 사고력 훈련이다. 사고력 훈련을 통해, 세상

prompt>  
...
```


### python
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM 

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

prompt = '인간처럼 생각하고, 행동하는 \'지능\'을 통해 인류가 이제까지 풀지 못했던'
with torch.no_grad():
  tokens = tokenizer.encode(prompt, return_tensors='pt').to(device='cuda', non_blocking=True)
  gen_tokens = model.generate(tokens, do_sample=True, temperature=0.8, max_length=64)
  generated = tokenizer.batch_decode(gen_tokens)[0]
  
print(generated)  # print: 인간처럼 생각하고, 행동하는 '지능'을 통해 인류가 이제까지 풀지 못했던 문제의 해답을 찾을 수 있을 것이다. 과학기술이 고도로 발달한 21세기를 살아갈 우리 아이들에게 가장 필요한 것은 사고력 훈련이다. 사고력 훈련을 통해, 세상
```


## Experiments

### In-context Few-Shots

| Models        | #params | NSMC (Acc.) | YNAT (F1) | KLUE-STS (F1) |
|:--------------|--------:|------------:|----------:|--------------:|
| HyperCLOVA[1] |    1.3B |        83.9 |      58.7 |          60.9 |
| HyperCLOVA[1] |    6.9B |        83.8 |      67.5 |          59.3 |
| HyperCLOVA[1] |   13.0B |        87.9 |      67.9 |          60.0 |
| HyperCLOVA[1] |   39.0B |        88.0 |      71.4 |          61.6 |
| HyperCLOVA[1] |   82.0B |    **88.2** |      72.7 |      **65.1** |
| **Ours**      |    6.0B |        87.8 |  **78.0** |          64.3 |


### Finetuning / P-Tuning

| Models                    | #params | method       | NSMC (Acc.) | KorSTS(spearman) |
|:--------------------------|--------:|:-------------|------------:|-----------------:|
| SKT-AI/KoGPT-2 2.0[2]     |    125M | `finetuning` |        93.3 |             78.4 |
| SKT-AI/KoGPT-2 Trinity[3] |    1.2B | `finetuning` |        93.2 |             83.4 |
| HyperCLOVA[1]             |    1.3B | `p-tuning`   |        91.7 |                - |
| HyperCLOVA[1]             |   39.0B | `p-tuning`   |        93.0 |                - |
| **Ours**                  |    135M | `finetuning` |        95.1 |             83.0 |
| **Ours**                  |    6.0B | `finetuning` |    **95.7** |         **85.3** |

We conducted this experiments using [4], with same hyperparameters.


## Limitations

KakaoBrain KoGPT was trained on `rayn dataset`, a dataset known to contain profanity, lewd, political changed, and other harsh language. Therefore, KoGPT can generate socially unacceptable texts. As with all language models, It is difficult to predict in advance how KoGPT will response to particular prompts and offensive content without warning.

Primarily Korean: koGPT is primarily trained on Korean texts, and is best for classifying, searching, summarizing or generating such texts. KoGPT by default perform worse on inputs that are different from the data distribution it is trained on, including non-Korean as well as specific dialects of Korean that are not well represented in the training data.



## Citation

If you apply this library or model to any project and research, please cite our code:

```
@misc{kakaobrain2021kogpt,
  title         = {KoGPT: KakaoBrain Korean(hangul) Generative Pre-trained Transformer}
  author        = {Ildoo Kim and Gunsoo Han and Jiyeon Ham and Woonhyuk Baek},
  year          = {2021},
  howpublished  = {\url{https://github.com/kakaobrain/kogpt}},
}
```


## Contact

This is released as an open source in the hope that it will be helpful to many research institutes and startups for research purposes. We look forward to contacting us from various places who wish to cooperate with us. 

contact@kakaobrain.com


## License

The `source code` of KakaoBrain `KoGPT` are licensed under [Apache 2.0](LICENSE.apache-2.0) License.   
The `pretrained wieghts` of KakaoBrain `KoGPT` are licensed under [CC-BY-NC-ND 4.0 License](https://creativecommons.org/licenses/by-nc-nd/4.0/) License.

카카오브레인 `KoGPT`의 `소스코드(source code)`는 [Apache 2.0](LICENSE.apache-2.0) 라이선스 하에 공개되어 있습니다.   
카카오브레인 `KoGPT`의 `사전학습된 가중치(pretrained weights)`는 [CC-BY-NC-ND 4.0 라이선스](https://creativecommons.org/licenses/by-nc-nd/4.0/) 라이선스 하에 공개되어 있습니다.   
모델 및 코드, 사전학습된 가중치를 사용할 경우 라이선스 내용을 준수해 주십시오. 라이선스 전문은 [Apache 2.0](LICENSE.apache-2.0), [LICENSE.cc-by-nc-nd-4.0](LICENSE.cc-by-nc-nd-4.0) 파일에서 확인하실 수 있습니다.


## References

[1] [HyperCLOVA](https://arxiv.org/abs/2109.04650): Kim, Boseop, et al. "What changes can large-scale language models bring? intensive study on hyperclova: Billions-scale korean generative pretrained transformers." arXiv preprint arXiv:2109.04650 (2021).   
[2] [SKT-AI/KoGPT-2 2.0](https://github.com/SKT-AI/KoGPT2): "SKT-AI/KoGPT2: Korean GPT-2 pretrained cased (KoGPT2)." https://github.com/SKT-AI/KoGPT2 (2021).   
[3] [SKT-AI/KoGPT-2 Trinity](https://huggingface.co/skt/ko-gpt-trinity-1.2B-v0.5): "Ko-GPT-Trinity 1.2B." https://huggingface.co/skt/ko-gpt-trinity-1.2B-v0.5 (2021).   
[4] [KoGPT2-subtasks](https://github.com/haven-jeon/KoGPT2-subtasks): "KoGPT2 v2.0 한국어 평가 모듈" https://github.com/haven-jeon/KoGPT2-subtasks (2021).
