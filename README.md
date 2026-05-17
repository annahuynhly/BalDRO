<h1 align="center">
Evaluating BalDRO for LLM Unlearning: Alternative Forget Objectives and Cross-Model Generalisation
</h1>

<div align="center">
  <a href="https://annahuynhly.github.io/pages/projects/bluedot_unlearning.html"><img src="https://img.shields.io/badge/Project-Page-blue"></a> &nbsp;
  <a href="https://github.com/nxZhai/BalDRO"><img src="https://img.shields.io/badge/Original-BalDRO-94c320?logo=github"></a> &nbsp;
  <a href="https://arxiv.org/pdf/2601.09172"><img src="https://img.shields.io/badge/arXiv-2601.09172-red?logo=arXiv"></a> &nbsp;
</div>

---

This repository is a fork of [BalDRO](https://github.com/nxZhai/BalDRO) (Shao et al., WWW 2026), extended as part of a BlueDot AI Safety technical project. The original repo evaluates BalDRO-DV and BalDRO-G on Llama-2-7B using NPO, SimNPO, and SatImp as forget objectives. This fork investigates two additional directions:

1. **Alternative forget objectives.** We test WGA (Weighted Gradient Ascent) and TNPO (Token-wise NPO) as drop-in replacements for NPO within the BalDRO-DV framework, evaluating whether the robustness gains transfer to different loss functions.

2. **Cross-model generalisation.** We extend experiments beyond Llama-2-7B to four additional models: Llama-3.2-1B-Instruct, Llama-3.1-8B-Instruct, Qwen3-8B, and Mistral-7B-Instruct-v0.3.

All experiments use the TOFU benchmark (`forget01` split) and report forget quality, model utility, and gibberish rate.

## What's added in this fork

- `src/trainer/unlearn/wga.py` — WGA and BalDRO-DV + WGA trainers
- `src/trainer/unlearn/tnpo.py` — TNPO and BalDRO-DV + TNPO trainers
- `scripts/unlearn/tofu/train_tofu_wga_fast.sh` — WGA baseline
- `scripts/unlearn/tofu/train_tofu_drwga_fast.sh` — BalDRO-DV + WGA
- `scripts/unlearn/tofu/train_tofu_tnpo_fast.sh` — TNPO baseline
- `scripts/unlearn/tofu/train_tofu_drtnpo_fast.sh` — BalDRO-DV + TNPO (β_DV=2.0)
- `scripts/unlearn/tofu/train_tofu_drtnpo_fast_bdv0.5.sh` — BalDRO-DV + TNPO (β_DV=0.5)
- `scripts/unlearn/tofu/train_tofu_drnpo_fast.sh` — BalDRO-DV + NPO on Llama-2-7B
- Per-model fast scripts for NPO and BalDRO-DV + NPO on Llama-3.2-1B, Llama-3.1-8B, Qwen3-8B, and Mistral-7B
- Finetuning scripts for Qwen3-8B and Mistral-7B on TOFU (no OpenUnlearning checkpoints exist for these models)

## Setup

Setup instructions are unchanged from the original repo. See [BalDRO](https://github.com/nxZhai/BalDRO) for environment setup, dataset preparation, and model downloads.

For Llama-3.x models, use the pretrained TOFU checkpoints from [OpenUnlearning](https://huggingface.co/open-unlearning). For Qwen3-8B and Mistral-7B, run the finetuning scripts in `scripts/unlearn/tofu/` before unlearning.

## Acknowledgements

This work builds on [BalDRO](https://github.com/nxZhai/BalDRO) by Shao et al. and [Open-Unlearning](https://github.com/locuslab/open-unlearning).

## Citation

```bibtex
@inproceedings{shao2026baldro,
  title={Baldro: A distributionally robust optimization based framework for large language model unlearning},
  author={Shao, Pengyang and Zhai, Naixin and Chen, Lei and Yang, Yonghui and Zhu, Fengbin and Yang, Xun and Wang, Meng},
  booktitle={Proceedings of the ACM Web Conference 2026},
  pages={8874--8884},
  year={2026}
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
