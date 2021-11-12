# KoGPT
[![KakaoBrain](https://img.shields.io/badge/kakao-brain-ffcd00.svg)](http://kakaobrain.com/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![License: Commercial](https://img.shields.io/badge/License-Commercial-ffcd00.svg)](LICENSE.commercial)
[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)

KakaoBrain Korean Generative Pre-Training Models


## Usage





## Experiments

### In-context Few-Shots
| Models        | #params | NSMC (Acc.) | YNAT (F1) | KLUE-STS (F1) |
|:--------------|--------:|------------:|----------:|--------------:|
| HyperCLOVA[1] |    1.3B |        83.9 |      58.7 |          60.9 |
| HyperCLOVA[1] |    6.9B |        83.8 |      67.5 |          59.3 |
| HyperCLOVA[1] |   13.0B |        87.9 |      67.9 |          60.0 |
| HyperCLOVA[1] |   39.0B |        88.0 |      71.4 |          61.6 |
| HyperCLOVA[1] |   82.0B |    **88.2** |      72.7 |      **65.1** |
| Ours          |    6.0B |        87.8 |  **78.0** |          64.3 |


### Finetuning
| Models           | #params | method     | NSMC (Acc.) |
|:-----------------|--------:|:-----------|------------:|
| SKT-AI/KoGPT2[2] |    125M | `finetuning` |        93.3 |
| HyperCLOVA[1]    |    1.3B | `p-tuning`   |        91.7 |
| HyperCLOVA[1]    |   39.0B | `p-tuning`   |        93.0 |
| Ours             |    6.0B | `finetuning` |    **95.7** |


## Citation

If you apply this library or model to any project and research, please cite our code:

```
@article{kakaobrain2021kogpt,
  title         = {KakaoBrain Korean Generative Pre-Training}
  author        = {Ildoo Kim and Gunsoo Han and Jiyeon Ham and Woonhyuk Baek},
  year          = {2021},
  howpublished  = {\url{https://github.com/kakaobrain/kogpt}},
}
```


## References

[1] [HyperCLOVA](https://arxiv.org/abs/2109.04650): Kim, Boseop, et al. "What changes can large-scale language models bring? intensive study on hyperclova: Billions-scale korean generative pretrained transformers." arXiv preprint arXiv:2109.04650 (2021).   
[2] [SKT-AI/KoGPT2](https://github.com/SKT-AI/KoGPT2): "SKT-AI/KoGPT2: Korean GPT-2 pretrained cased (KoGPT2)." https://github.com/SKT-AI/KoGPT2 (2021).     


## License

The `source code` of KakaoBrain `KoGPT` are licensed under [AGPL 3.0](LICENSE.agpl-3.0) and [Commercial](LICENSE.commercial) License.   
The `pretrained wieghts` of KakaoBrain `KoGPT` are licensed under [CC-BY-NC-ND 4.0 라이선스](https://creativecommons.org/licenses/by-nc-nd/4.0/) License.

카카오브레인 `KoGPT`의 `소스코드(source code)`는 [AGPL 3.0](LICENSE.agpl-3.0) 과 [Commercial](LICENSE.commercial) 라이선스 하에 공개되어 있습니다.   
카카오브레인 `KoGPT`의 `사전학습된 가중치(pretrained weights)`는 [CC-BY-NC-ND 4.0 라이선스](https://creativecommons.org/licenses/by-nc-nd/4.0/) 라이선스 하에 공개되어 있습니다.   
모델 및 코드, 사전학습된 가중치를 사용할 경우 라이선스 내용을 준수해 주십시오. 라이선스 전문은 [LICENSE.agpl-3.0](LICENSE.agpl-3.0), [LICENSE.commercial](LICENSE.commercial), [LICENSE.cc-by-nc-nd-4.0](LICENSE.cc-by-nc-nd-4.0) 파일에서 확인하실 수 있습니다.
