# Track engine

A documentation for single object tracking engine.

# SERVER PREPARATION

## STEP 1. Prepare your server environment

We use `PyTrorch` deep learning framework as our tracking engine core.
In order to ensure successful operation, the following conditions need to be met.

-   `PyTorch >= 1.7.0`

```bash
# You must meet the requirement of cudatoolkit & YOUR_SERVER_NVIDIA_Quadro (https://developer.nvidia.com/cuda-gpus#compute)
# NVIDIA_Quadro: 8.x <=> cudatoolkit=11.x
conda install -y pytorch==1.7.0 torchvision==0.8.1 cudatoolkit=10.2 -c pytorch
```

Create a new virtual env for project (Recommend using [Miniconda3](https://docs.conda.io/en/latest/miniconda.html)) and install the required packages.

```bash
conda create -n ENV_NAME python=3.8

bash install.sh
```

## STEP 2. Set project paths

Run the following command to set paths for this project

```bash
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir .
```

After running this command, you can also modify paths by editing these two files

```bash
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

## STEP 3. Prepare trained tracker engine model

```bash
mkdir model_zoo
cp /data/ckpt/... .
mv .pth.tar tracking_engine_1_online_22k.pth.tar
```

## Socket communication interface

整个过程大致如下：

1. 服务端 tracking engine 运行之后，开启 12345 端口（自行更改），一直等待客户端的连接（默认设置最多 5 个连接）；

2. 如果有客户端发送连接请求，服务端进行处理。其中先接收客户端的消息，第一次请求携带的参数包括 tracking_video_path（相对于客户端本地的路径）

3. server 先判断本地是否在视频库中包含这个视频，如果没有，向客户端发送上传视频的请求，客户端接收到服务端发送的消息后，将视频进行上传；如果有，等待下一步指令；

4. 视频准备好之后，向客户端发送初始化第一帧的请求，客户端接收到消息之后，将跟踪目标的初始位置信息发送给服务端；

5. 服务端接收到客户端发送到的初始位置信息之后，执行跟踪推理，并实时将每一帧的跟踪结果返回给客户端，直到客户端返回一个确认收到 ACK 的消息之后，才执行后续的跟踪；

6. 客户端接收到跟踪结果之后，给客户端发送一个已收到的信息，等待服务器将下一帧的跟踪结果返回。

# CLIENT PREPARATION

video_path

通信流程！
