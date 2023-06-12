# Track engine

A documentation for single object tracking engine.





# SERVER PREPARATION

## STEP 1. Prepare your server environment

We use `PyTrorch` deep learning framework as our tracking engine core. In order to ensure successful operation, the following conditions need to be met.

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

The entire processing procedure is as follows:

1. After the server's tracking engine is running, it starts listening on port 12345 (customizable) and continuously waits for client connections (with a default maximum of 5 connections, you can also customize it).

2. If a client sends a connection request, the server handles it according to the following steps.

   

   Firstly, it receives a message from the client, and the initial request includes the parameter "video_name".

   

   Then, the server checks whether the video is included in the local video library. If it is not, it sends a request to the client to upload the video. Upon receiving the server's message, the client uploads the video. If the video already exists, it waits for the next instruction.

   

   Once the video is prepared, the server sends a request to the client to initialize the first frame. Upon receiving the message, the client sends the initial position information of the tracked target to the server.

   

   After receiving the initial position information sent by the client, the server performs tracking inference and continuously sends the tracking results of each frame back to the client in real-time. The server proceeds with further tracking only after receiving a message from the client confirming the receipt of the ACK of the received data.

   

   After receiving the tracking results, the client sends a message to acknowledge that it has received the results and waits for the server to return the tracking results of the next frame.





# CLIENT PREPARATION

## STEP 1. Prepare your environment

You need install the follwing packages:

```bash
pip install opencv-python numpy tqdm
```



## STEP 2. Prepare your test video

Just prepare the video files you are going to tracking.



## STEP 3. Run your tracking instance!

```bash
python track_demo.py --video_path="VIDEO_PATH" --save_img=True --server_ip="YOUR_SERVER_IP" --server_port=12345
```

After selecting an ROI area, please be patient as it may take some time to perform tracking.

If you have set the variable save_img True, the default dir structure is `track_res/{YOUR_TEST_VIDEO_NAME}/{images},result.txt`.

```
├── track_res
│   └── {YOUR_TEST_VIDEO_NAME}
│       ├── images (each tracked frame with bbox)
│       └── result.txt
```


## Socket communication interface

Communicating with server,

1. Send `video_name` message to server.

2. Always monitor the message returned by the client,

   If msg="upload", upload your video.

   If msg="draw_bbox", select a ROI for the first frame for tracking.

   If msg="bbox", save the tracking result to result.txt and visualize the tracking process.

   If msg="bbox", save the tracking result to result.txt and visualize the tracking process.

   if msg="status", data["status"] == "TRACK_END", terminate tracking program.









