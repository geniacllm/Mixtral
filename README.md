# MoE Recipes
現状Colabの実装までです

環境構築方法はColabのnotebookに記載しています

事前学習とfinetuningを行うことができます。

## 配布環境での環境構築

### condaを使えるようにする
既に使える場合は飛ばしてください

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.10.0-1-Linux-x86_64.sh
bash Miniconda3-py310_23.10.0-1-Linux-x86_64.sh

#自分のダウンロードしたminiconda3のパス
echo 'export PATH="/home/ext_kumamama6_gmail_com/miniconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
conda init
source ~/.bashrc
```

### MoE Recipes用の仮想環境の構築

conda create -n mixtralenv python=3.11
conda activate mixtralenv
conda install nvidia/label/cuda-11.8.0::cuda-toolkit
