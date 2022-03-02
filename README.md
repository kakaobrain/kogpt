# KoGPT
[![KakaoBrain](https://img.shields.io/badge/Kakao-Brain-ffcd00.svg)](http://kakaobrain.com/)
[![Github: kogpt](https://img.shields.io/badge/Github-kogpt-000000.svg)](https://github.com/kakaobrain/kogpt)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)   
[![huggingface: KoGPT-6B](https://img.shields.io/badge/huggingface-KoGPT_6B_ryan1.5b-ffcd00.svg)](https://huggingface.co/kakaobrain/kogpt/tree/KoGPT6B-ryan1.5b)
[![huggingface: KoGPT-6B](https://img.shields.io/badge/huggingface-KoGPT_6B_ryan1.5b_(float16)-ffcd00.svg)](https://huggingface.co/kakaobrain/kogpt/tree/KoGPT6B-ryan1.5b-float16)
[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)


* KoGPT (Korean Generative Pre-trained Transformer)
  * [https://github.com/kakaobrain/kogpt](https://github.com/kakaobrain/kogpt)
  * [https://huggingface.co/kakaobrain/kogpt](https://huggingface.co/kakaobrain/kogpt)


## Model Descriptions

### KoGPT6B-ryan1.5b

* [\[huggingface\]\[kakaobrain/kogpt\]\[KoGPT6B-ryan1.5b\]](https://huggingface.co/kakaobrain/kogpt/tree/KoGPT6B-ryan1.5b)
* [\[huggingface\]\[kakaobrain/kogpt\]\[KoGPT6B-ryan1.5b-float16\]](https://huggingface.co/kakaobrain/kogpt/tree/KoGPT6B-ryan1.5b-float16)

| Hyperparameter       | Value         |
|:---------------------|--------------:|
| <img src="https://render.githubusercontent.com/render/math?math=n_{parameters}"> | 6,166,502,400 |
| <img src="https://render.githubusercontent.com/render/math?math=n_{layers}">     | 28            |
| <img src="https://render.githubusercontent.com/render/math?math=d_{model}">      | 4,096         |
| <img src="https://render.githubusercontent.com/render/math?math=d_{ff}">         | 16,384        |
| <img src="https://render.githubusercontent.com/render/math?math=n_{heads}">      | 16            |
| <img src="https://render.githubusercontent.com/render/math?math=d_{head}">       | 256           |
| <img src="https://render.githubusercontent.com/render/math?math=n_{ctx}">        | 2,048         |
| <img src="https://render.githubusercontent.com/render/math?math=n_{vocab}">      | 64,512        |
| Positional Encoding  | [Rotary Position Embedding (RoPE)](https://arxiv.org/abs/2104.09864) |
| RoPE Dimensions      | 64            |


## Hardware requirements

### KoGPT6B-ryan1.5b

#### GPU
The following is the recommended minimum GPU hardware guidance for a handful of example KoGPT.
* `32GB GPU RAM` in the required minimum memory size

### KoGPT6B-ryan1.5b-float16

#### GPU
The following is the recommended minimum GPU hardware guidance for a handful of example KoGPT.
* half-precision requires NVIDIA GPUS based on Volta, Turing or Ampere
* `16GB GPU RAM` in the required minimum memory size


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
  'kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16',  # or float32 version: revision=KoGPT6B-ryan1.5b
  bos_token='[BOS]', eos_token='[EOS]', unk_token='[UNK]', pad_token='[PAD]', mask_token='[MASK]'
)
model = AutoModelForCausalLM.from_pretrained(
  'kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16',  # or float32 version: revision=KoGPT6B-ryan1.5b
  pad_token_id=tokenizer.eos_token_id,
  torch_dtype='auto', low_cpu_mem_usage=True
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

We have been reported to have issues(https://github.com/kakaobrain/kogpt/issues/17) with our downstream evaluation.

The previously published performance evaluation table was deleted because it was difficult to see it as a fair comparison because the comparison target algorithm was different and the performance measurement method could not be confirmed.

You can refer to the above issue link for the existing performance evaluation table and troubleshooting results.


## Limitations

KakaoBrain `KoGPT` was trained on raw data, a dataset known to contain profanity, lewd, political changed, and other harsh language.
Therefore, `KoGPT` can generate socially unacceptable texts. As with all language models, It is difficult to predict in advance how `KoGPT` will response to particular prompts and offensive content without warning.

Primarily Korean: `KoGPT` is primarily trained on Korean texts, and is best for classifying, searching, summarizing or generating such texts.
`KoGPT` by default perform worse on inputs that are different from the data distribution it is trained on, including non-Korean as well as specific dialects of Korean that are not well represented in the training data.

If abnormal or socially unacceptable text is generated during testing, please send a "prompt" and the "generated text" to [opensource+kogpt@kakaobrain.com](mailto:opensource+kogpt@kakaobrain.com).  



카카오브레인 `KoGPT`는 AI커뮤니티를 위한 연구용으로 욕설, 음란, 정치적 내용 및 기타 거친 언어에 대한 처리를 하지 않은 원시 데이터로 학습하였습니다.
따라서 `KoGPT`는 사회적으로 용인되지 않은 텍스트를 생성할 수 있습니다. 다른 언어 모델과 마찬가지로 특정 프롬프트와 공격적인 콘텐츠에 어떠한 결과를 생성할지 사전에 파악하기 어렵습니다.

`KoGPT`는 주로 한국어 텍스트로 학습을 하였으며 이러한 텍스트를 분류, 검색, 요약 또는 생성하는데 가장 적합합니다.
기본적으로 `KoGPT`는 학습 데이터에 잘 나타나지 않는 방언뿐만아니라 한국어가 아닌 경우와 같이 학습 데이터에서 발견하기 어려운 입력에서 좋지 않은 성능을 보입니다.

본 KoGPT를 활용한 연구, 개발, 테스트 등에 있어 위의 부분을 꼭 유의하시기 바랍니다.  
테스트중에 발생한 비정상적인 혹은 사회적으로 용인되지 않는 텍스트가 생성된 경우 [opensource+kogpt@kakaobrain.com](mailto:opensource+kogpt@kakaobrain.com)로 "prompt"와 "생성된 문장"을 함께 보내주시기 바랍니다.


## Citation

If you apply this library or model to any project and research, please cite our code:

```
@misc{kakaobrain2021kogpt,
  title         = {KoGPT: KakaoBrain Korean(hangul) Generative Pre-trained Transformer},
  author        = {Ildoo Kim and Gunsoo Han and Jiyeon Ham and Woonhyuk Baek},
  year          = {2021},
  howpublished  = {\url{https://github.com/kakaobrain/kogpt}},
}
```


## Contact

This is released as an open source in the hope that it will be helpful to many research institutes and startups for research purposes. We look forward to contacting us from various places who wish to cooperate with us. 

[contact@kakaobrain.com](mailto:contact@kakaobrain.com)


## License

The `source code` of KakaoBrain `KoGPT` are licensed under [Apache 2.0](LICENSE.apache-2.0) License.   
The `pretrained weights` of KakaoBrain `KoGPT` are licensed under [CC-BY-NC-ND 4.0 License](https://creativecommons.org/licenses/by-nc-nd/4.0/) License.

카카오브레인 `KoGPT`의 `소스코드(source code)`는 [Apache 2.0](LICENSE.apache-2.0) 라이선스 하에 공개되어 있습니다.   
카카오브레인 `KoGPT`의 `사전학습된 가중치(pretrained weights)`는 [CC-BY-NC-ND 4.0 라이선스](https://creativecommons.org/licenses/by-nc-nd/4.0/) 라이선스 하에 공개되어 있습니다.   
모델 및 코드, 사전학습된 가중치를 사용할 경우 라이선스 내용을 준수해 주십시오. 라이선스 전문은 [Apache 2.0](LICENSE.apache-2.0), [LICENSE.cc-by-nc-nd-4.0](LICENSE.cc-by-nc-nd-4.0) 파일에서 확인하실 수 있습니다.

### Obligation to use

While Open Source software may be free to use, that does not mean it is free of obligation. To determine whether your intended use of KoGPT is suitable for the Apache 2.0 (or CC-BY-NC-ND 4.0), please consider the license guide. If you violate the license, you may be subject to legal action such as prohibition of use or claim for damages depending on the use.

오픈소스 소프트웨어는 무료로 사용할 수 있지만 이것이 의무가 없다는 의미는 아닙니다. KoGPT의 사용에 앞서 라이선스 가이드를 살펴보고 예정한 사용이 Apache 2.0 (또는 CC-BY-NC-ND 4.0)를 준수하는지 여부를 먼저 확인하시기 바랍니다. 라이선스를 위반하는 경우, 내용에 따라 사용금지, 손해배상 청구 등의 법적 조치를 취할 수 있습니다.


## References

[1] [HyperCLOVA](https://arxiv.org/abs/2109.04650): Kim, Boseop, et al. "What changes can large-scale language models bring? intensive study on hyperclova: Billions-scale korean generative pretrained transformers." arXiv preprint arXiv:2109.04650 (2021).   


----


## Contribution

### Disclaimer
The contribution section is not an official KakaoBrain product.

### AK391's Web Demo on Huggingface Spaces
* see demo: https://huggingface.co/spaces/akhaliq/kogpt
  * Web Demo is integrated to [Huggingface Spaces](https://huggingface.co/spaces) with [Gradio](https://github.com/gradio-app/gradio).
  * Contributors: [AK391](https://github.com/AK391)


