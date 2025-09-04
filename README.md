## Installation

Create a new environment:
```bash
conda create -n meanflow python=3.12
```

Install PyTorch:
```bash
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.4 -c pytorch -c nvidia
```

Install the rest of the dependencies:
```bash
pip install -r requirements.txt
```
