# Mixtral7*8B zero3
現状Colab(A100)の実装までです

環境構築方法はColabのnotebookに記載しています
* トークナイザーの読み込み
* 事前学習
* finetuning(独自モデルのHuggingfaceからは追加実装必要)
* モデルのチェックポイントの変換
* wandbへの書き込み
* Huggingfaceへのアップロード
を確認済みです。

## 配布環境での環境構築

### 1.condaを使えるようにする
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

### 2.Mixtral7*8B zero3用の仮想環境の構築

```bash
conda create -n mixtralenv python=3.11

mkdir -p ~/miniconda3/envs/mixtralenv/etc/conda/activate.d
echo 'export ORIGINAL_LD_LIBRARY_PATH=$LD_LIBRARY_PATH' > ~/miniconda3/envs/mixtralenv/etc/conda/activate.d/edit_environment_variable.sh
echo 'export LD_LIBRARY_PATH="$HOME/miniconda3/envs/mixtralenv/lib:$LD_LIBRARY_PATH"' >> ~/miniconda3/envs/mixtralenv/etc/conda/activate.d/edit_environment_variable.sh
chmod +x ~/miniconda3/envs/mixtralenv/etc/conda/activate.d/edit_environment_variable.sh

mkdir -p ~/miniconda3/envs/mixtralenv/etc/conda/deactivate.d
echo 'export LD_LIBRARY_PATH=$ORIGINAL_LD_LIBRARY_PATH' > ~/miniconda3/envs/mixtralenv/etc/conda/deactivate.d/rollback_environment_variable.sh
echo 'unset ORIGINAL_LD_LIBRARY_PATH' >> ~/miniconda3/envs/mixtralenv/etc/conda/deactivate.d/rollback_environment_variable.sh
chmod +x ~/miniconda3/envs/mixtralenv/etc/conda/deactivate.d/rollback_environment_variable.sh

conda activate mixtralenv
```

#### 2.1 ライブラリのインストール
ここから(mixtralenv)内です

```bash
conda install nvidia/label/cuda-11.8.0::cuda-toolkit

#gitのuser名とアクセストークンを入れてください
git clone https://{user}:{password}@github.com/kumagai6/moe.git moe-recipes

pip install --upgrade pip setuptools wheel
```
##### (候補1)condaでのinstall、エラーでた場合教えてください
```bash
cd ~/moe-recipes
bash install_gcp.sh
```

##### (候補2)mpi4pyがinstallできなくて、2つに分けました。(動作未確認、試行錯誤していたら候補1でinstallされました。)
mpi4pyのinstallにはシステムレベルでのopenmpiのインストールが必要です。sudo権限がなく現在の環境ではできないので、$HOME/openmpiのローカルにopenmpiをインストールします。
```bash
cd ~/moe-recipes
bash install_gcp_1.sh

wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.6.tar.gz
gunzip -c openmpi-4.1.6.tar.gz | tar xf -
cd ~/moe-recipes/openmpi-4.1.6
./configure --prefix=$HOME/openmpi
make all install

export PATH=$HOME/openmpi/bin:$PATH
export LD_LIBRARY_PATH=$HOME/openmpi/lib:$LD_LIBRARY_PATH

cd ~/moe-recipes
bash install_gcp_2.sh
```

#### 2.2 apexのインストール
```bash
cd ~/moe-recipes
#将来的に不要です。学習データの保存先、こちらのディレクトリのまま、学習コードを作ってしまいました。
git clone https://github.com/hotsuyuki/Megatron-DeepSpeed

cd ~/moe-recipes
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
```

#### 2.3 コードのコンパイル
```bash
%cd ~/moe-recipes/megatron_lm/megatron/core/datasets
python setup.py build_ext --inplace

%cd ~/moe-recipes/tools/tokenizer

```

## wandbログイン
wandb login

## 事前学習の実行
```bash
%cd ~/moe-recipes/scripts/abci/mixtral
sbatch mixtral-7bx8_pretrain_GCP.sh --nodes=1 --gpus-per-node=1 --time=06:00:00
```
