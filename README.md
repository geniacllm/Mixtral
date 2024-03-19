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

```bash
conda create -n mixtralenv python=3.11
conda activate mixtralenv
```

#### ライブラリのインストール
ここから(mixtralenv)内です

```bash
conda install nvidia/label/cuda-11.8.0::cuda-toolkit

#gitのuser名とアクセストークンを入れてください
git clone https://{user}:{password}@github.com/kumagai6/moe.git moe-recipes

pip install --upgrade pip setuptools wheel

cd ~/moe-recipes
bash install_gcp_1.sh

#全てyで不要なファイルを削除してメモリを解放します。
conda clean --all
#pipでmpi4pyをインストールしたいですが、libopenmpi-devをapt-getでinstallすることを求められました。こちらsudo権限がなく使えないコマンドなので
#condaでinstallしています。かなり時間がかかります。もっと早い方法があるかもしれません。ChatGPTにはmambaでやる事を勧められました。
conda install -c conda-forge mpi4py

bash install_gcp_2.sh
#将来的に不要です。
git clone https://github.com/hotsuyuki/Megatron-DeepSpeed

%cd ~/moe-recipes
git clone https://github.com/NVIDIA/apex
%cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
```

#### コードのコンパイル
```bash
%cd ~/moe-recipes/megatron_lm/megatron/core/datasets
python setup.py build_ext --inplace

%cd ~/moe-recipes

```
