<!-- # Kimi-Dev -->

<div align="center">
  <img src="./assets/main_logo.png" alt="Kimi Logo" width="400" />
<h2><a href="https://moonshotai.github.io/Kimi-Dev/">
Introducing Kimi-Dev: <br>A Strong and Open-source Coding LLM for Issue Resolution</a></h2>
</a></h2>
<b>Kimi-Dev Team</b>
<br>
</div>

<div align="center">
  <a href="https://arxiv.org/abs/2509.23045">
    <b>📄 Tech Report (Arxiv)</b>
  </a> &nbsp;|&nbsp;
  <a href="https://huggingface.co/moonshotai/Kimi-Dev-72B">
    <b>🤗 Huggingface</b>
  </a> &nbsp;|&nbsp;
  <a href="https://huggingface.co/spaces/moonshotai/Kimi-Dev-72B">
    <b>💻 Demo (HF Space)</b>
  </a> &nbsp;
</div>
<br>
<br>


We introduce Kimi-Dev-72B, our new open-source coding LLM for software engineering tasks. Kimi-Dev-72B achieves a new state-of-the-art on SWE-bench Verified among open-source models.

- Kimi-Dev-72B achieves 60.4% performance on SWE-bench Verified. It surpasses the runner-up, setting a new state-of-the-art result among open-source models.


- Kimi-Dev-72B is optimized via large-scale reinforcement learning. It autonomously patches real repositories in Docker and gains rewards only when the entire test suite passes. This ensures correct and robust solutions, aligning with real-world development standards.


- Kimi-Dev-72B is available for download and deployment on Hugging Face and GitHub. We welcome developers and researchers to explore its capabilities and contribute to development.


<div align="center">
  <img src="./assets/open_performance_white.png" alt="Kimi Logo" width="600" />
  <p><b>Performance of Open-source Models on SWE-bench Verified.</b></p>

</div>


<!-- ## 💡 Introduction -->

<!-- ## 🔥 News -->



## ⚙️ Installation

```bash
# clone repo
git clone https://github.com/MoonshotAI/Kimi-Dev.git
# create env
conda create -n kimidev python=3.12
# local install
pip install -e .
```

## 🛠️ How to use

### Prepare repo structure [From [Agentless](https://github.com/OpenAutoCoder/Agentless/)]
Since for each issue in the benchmark (both SWE-Bench Lite and SWE-Bench Verified) we need to checkout the repository and process the files, you might want to save some time by downloading the preprocessed data here: [swebench_repo_structure.zip](https://drive.google.com/file/d/15-4XjTmY48ystrsc_xcvtOkMs3Fx8RoW/view). After downloading, please unzip and export the location as such 
```bash
export PROJECT_FILE_LOC={folder which you saved}
``` 

### Deploy vLLM Model

#### Installation
```
# Install vLLM with CUDA 12.8.
# If you are using pip.
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu128
# If you are using uv.
uv pip install vllm --torch-backend=auto
```

#### Serving
```
vllm serve Kimi-Dev-72B --served-model-name kimi-dev --host 0.0.0.0 --port 8000 --gpu-memory-utilization 0.95 --max-seq-len-to-capture 131072 --tensor-parallel-size 8
```

### Rollout
Kimi-Dev adopts a simplified two-stage framework for handling code repair and test writing tasks:

1. **File Localization**: Intelligently identify key files that need modification based on problem descriptions and repository structure
2. **Code Editing**: Perform precise code modifications on the located files, including bug fixes or unit test insertions

Compared to multi-step localization methods, we perform localization at the file level and then pass the complete file to the repair step for more detailed reasoning.

Run rollout script:

```
conda activate kimidev
# Bugfixer
python kimidev/examples/rollout_messages_bugfixer.py --model_name {vllm_serve_model}
# Testwriter
python kimidev/examples/rollout_messages_testwriter.py --model_name {vllm_serve_model}
```

## 👀 Example Results
We provide some example result files as well as the files required for test-time scaling [here](./resources/).

You can also download these files from [Google Drive](https://drive.google.com/file/d/1Tv4u9_CjCAOIhyZC1pOmpHFx3q0Ui8ru/view?usp=drive_link).

## 💪 Contributing

Welcome to submit Pull Requests or create Issues to help improve the project.


## 😺 Contact

If you have any questions, please feel free to submit a GitHub issue or contact zhuhan@moonshot.cn.

## 📝 Citation
If you find our code and models useful, please kindly cite the following information.
```
@misc{yang2025kimidevagentlesstrainingskill,
      title={Kimi-Dev: Agentless Training as Skill Prior for SWE-Agents}, 
      author={Zonghan Yang and Shengjie Wang and Kelin Fu and Wenyang He and Weimin Xiong and Yibo Liu and Yibo Miao and Bofei Gao and Yejie Wang and Yingwei Ma and Yanhao Li and Yue Liu and Zhenxing Hu and Kaitai Zhang and Shuyi Wang and Huarong Chen and Flood Sung and Yang Liu and Yang Gao and Zhilin Yang and Tianyu Liu},
      year={2025},
      eprint={2509.23045},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2509.23045}, 
}
