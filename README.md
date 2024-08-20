# ONNX RuntimeとCUDAのバージョンが合わない時

## はじめに
すでに学習済みの`ONNX`モデルを使用する時、`CUDAライブラリ`のバージョンが合わないときがあります。
具体的には「`libcudart.so.11.0`などが見つからないエラーが出力される」などです。
`ONNX`モデルが作成されたときの`CUDAバージョン`と、現在のシステム上の`CUDAバージョン`が異なる場合にこのようなエラーが発生します。

例えば日本人用顔学習モデル[`JAPANESE FACE V1`](https://github.com/yKesamaru/FACE01_trained_models)を`CUDA 12.x`環境で使用した時、このようなエラーが出力されることがあります。

https://github.com/yKesamaru/FACE01_trained_models

通常、ある程度`CUDAバージョン`が異なっていても`ONNX Runtime`と`CUDA`の互換性が保証されています。
`ONNX Runtime`と`CUDA`のバージョンの互換性は以下のサイトから確認できます。

[CUDA Execution Provider: Requirements ](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements)

![](https://raw.githubusercontent.com/yKesamaru/onnxruntime-gpu_cuda/master/assets/2024-08-07-18-43-08.png)

ところが先のようなエラーが発生してコードが動作しない場合は、依存ライブラリを特定してシンボリックリンクを作成する方法でこの問題を回避できます。

ここではシステムのCUDAバージョンとONNX Runtimeが求めるCUDAバージョンが異なっている場合の対処法を解説します。

### 注意
[公式ドキュメント](https://onnxruntime.ai/docs/install/#install-onnx-runtime-gpu-cuda-12x)では以下の記述があります。

> Install ONNX Runtime GPU (CUDA 12.x)
> 
> For Cuda 12.x, please use the following instructions to install from ORT Azure Devops Feed
> ```bash
> pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
> ```

この操作は未検証です。この記事ではシンボリックリンクをはる方法を解説します。

![](https://raw.githubusercontent.com/yKesamaru/onnxruntime-gpu_cuda/master/assets/eye-catch.png)

## 環境
```bash
$ inxi -SG --filter
System:
  Kernel: 6.8.0-40-generic x86_64 bits: 64 Desktop: Unity
    Distro: Ubuntu 22.04.4 LTS (Jammy Jellyfish)
Graphics:
  Device-1: NVIDIA TU116 [GeForce GTX 1660 Ti] driver: nvidia v: 555.42.06
  Display: x11 server: X.Org v: 1.21.1.4 driver: X:
    loaded: modesetting,nouveau,nvidia unloaded: fbdev,vesa gpu: nvidia
    resolution: 2560x1440~60Hz
  OpenGL: renderer: NVIDIA GeForce GTX 1660 Ti/PCIe/SSE2
    v: 4.6.0 NVIDIA 555.42.06

$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2021 NVIDIA Corporation
Built on Thu_Nov_18_09:45:30_PST_2021
Cuda compilation tools, release 11.5, V11.5.119
Build cuda_11.5.r11.5/compiler.30672275_0

$ nvidia-smi
Tue Aug 20 18:06:06 2024
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 555.42.06              Driver Version: 555.42.06      CUDA Version: 12.5     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce GTX 1660 Ti     Off |   00000000:09:00.0  On |                  N/A |
| 41%   38C    P8             16W /  120W |    1727MiB /   6144MiB |     21%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

(FACE01_DEV) user@user:~/bin/FACE01_DEV$ pip list | grep -i "onnx"
onnx                          1.16.1
onnxruntime-gpu               1.18.1
```


## `nvidia-cuda-toolkit`がインストールされているか確認
`libcudart.so.11.0`は`CUDA`ランタイムライブラリです。まず`CUDA`がシステムに正しくインストールされているか確認してください。
```bash
sudo apt update
sudo apt install -y nvidia-cuda-toolkit
```

## `シンボリックリンク`の作成
`libonnxruntime_providers_cuda.so`などが必要とする（依存する）ライブラリがすべて正しい場所に存在することを確認します。
```bash
ldd /home/user/bin/FACE01/lib/python3.10/site-packages/onnxruntime/capi/libonnxruntime_providers_cuda.so
```
上記の出力結果から、たとえば以下のようにシンボリックリンクを作成します。
```bash
# libcufft.so.10 のシンボリックリンク作成
sudo ln -s /usr/lib/x86_64-linux-gnu/libcufft.so.10 /usr/local/cuda-11.8/lib64/libcufft.so.10

# libcublas.so.11 のシンボリックリンク作成
sudo ln -s /usr/lib/x86_64-linux-gnu/libcublas.so.11 /usr/local/cuda-11.8/lib64/libcublas.so.11

# libcublasLt.so.11 のシンボリックリンク作成
sudo ln -s /usr/lib/x86_64-linux-gnu/libcublasLt.so.11 /usr/local/cuda-11.8/lib64/libcublasLt.so.11
```
## `環境変数`に正しいパスを記述して永続化させる
`CUDA`ライブラリが正しいパスに設定されているか確認します。
`~/.bashrc`に以下の記述を行います。
```bash
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-11.8
```
`.bashrc`を再読込してください。
```bash
source ~/.bashrc
```
## `ONNX Runtime`を再インストール
```bash
pip uninstall onnxruntime-gpu
pip install onnxruntime-gpu==1.18.1
```
以上で必要な`CUDAライブラリ`が正しくロードされるはずです。

## おわりに
今回はシステムのCUDAバージョンとONNX Runtimeが要求するCUDAバージョンが異なった場合の対処法を解説しました。
どなたかの助けになれば幸いです。

## 参考文献
- [CUDA Execution Provider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#cuda-execution-provider)
- [Install ONNX Runtime GPU (CUDA 12.x) ](https://onnxruntime.ai/docs/install/#install-onnx-runtime-gpu-cuda-12x)
- [FACE01_trained_models](https://github.com/yKesamaru/FACE01_trained_models)
  - 日本人専用の顔認証モデル
- [FACE01](https://github.com/yKesamaru/FACE01_DEV)
  - `JAPANESE FACE V1`を使うためのPythonで書かれたオープンソースのリファレンス実装

