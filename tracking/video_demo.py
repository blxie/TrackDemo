import json
import os
import socket
import sys
import argparse
import threading

import cv2
import numpy as np
import base64

from pathlib import Path

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation import Tracker


def is_video_file_valid(video_path):
    if not os.path.exists(video_path):
        return False
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False

        cap.release()
        return True

    except cv2.error as e:
        print(f"Error: {e}")
        return False


def tracking(
    tracker,
    client_socket=None,
    optional_box=None,
    video_path=None,
    debug=0,
    track_res_file=None,
):
    params = tracker.params

    _debug = debug
    if debug is None:
        _debug = getattr(params, 'debug', 0)
    params.debug = _debug

    params.tracker_name = tracker.name
    params.param_name = tracker.parameter_name
    _tracker = tracker.create_tracker(params)

    assert os.path.isfile(video_path), f"Invalid param {video_path}, video path must be a valid video file!"

    output_boxes = []
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()

    if success is not True:
        print("Read frame from {} failed.".format(video_path))
        exit(-1)

    def _build_init_info(box):
        return {'init_bbox': box}

    if optional_box is not None:
        assert isinstance(optional_box, (list, tuple))
        assert len(optional_box) == 4, "valid box's foramt is [x,y,w,h]"

        _tracker.initialize(frame, _build_init_info(optional_box))
        output_boxes.append(optional_box)
    else:
        raise NotImplementedError("We haven't support cv_show now.")

    print("Hold on, tracking...")
    while True:
        ret, frame = cap.read()

        if frame is None:
            break

        # Draw box
        out = _tracker.track(frame)
        bbox_data = [int(s) for s in out['target_bbox']]
        output_boxes.append(bbox_data)

        # Send the tracking results to the client in real time
        msg = {'bbox': bbox_data, 'score': out['score']}  # data to encode
        msg_base64 = base64.b64encode(json.dumps(msg).encode())
        client_socket.sendall(msg_base64)

        # Waiting for confirmation message from client
        ack_message = client_socket.recv(1024)
        if ack_message == b'ACK':
            # Client has received the confirmation message and continues to process the next frame
            # print(ack_message)
            continue
        else:
            # Client did not send the correct acknowledgment message, disconnected
            break

    print(f"Tracking video finished: {video_path}")
    # tracked_bb = np.array(output_boxes).astype(int)
    # np.savetxt(track_res_file, tracked_bb, delimiter=',', fmt='%d')

    # Send "TRACK_END" signal to client
    msg = {'status': 'TRACK_END'}
    msg_base64 = base64.b64encode(json.dumps(msg).encode())
    client_socket.sendall(msg_base64)

    # release resources
    cap.release()
    cv2.destroyAllWindows()
    client_socket.close()


def run_video(
    client_socket: socket.socket,
    tracker_name,
    tracker_param,
    tracker_params=None,
):
    """Run the tracker on your webcam.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        debug: Debug level.
    """
    response_base64 = client_socket.recv(1024)  # Receive data
    response_data = base64.b64decode(response_base64).decode()
    js_data = json.loads(response_data)

    if "video_name" in js_data and js_data["video_name"]:  ## STEP1: Prepare video files
        video_name = js_data['video_name']  # Video file name
        data_path = os.path.join("database/usrdata", Path(video_name).stem)
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        video_path = os.path.join(data_path, video_name)

        # Check if the file already exists
        upload_video = False
        if not is_video_file_valid(video_path):
            upload_video = True

        # Send a message to the client whether to upload the video
        msg = {'upload': upload_video}
        msg_base64 = base64.b64encode(json.dumps(msg).encode())
        client_socket.sendall(msg_base64)

        if upload_video:
            file_size_bytes = client_socket.recv(4)  # Receive file size information
            file_size = int.from_bytes(file_size_bytes, 'big')  # Parse file size
            received_size = 0

            with open(video_path, "wb") as file:
                while received_size < file_size:
                    data = client_socket.recv(1024)  # Receive data from client
                    if not data:
                        break
                    file.write(data)
                    received_size += len(data)

        print("视频已准备好！\n")

    ## STEP2: ready init_bbox
    # Video data has been prepared, and the init_bbox request is initiated to the client
    msg = {"draw_bbox": True}
    msg_base64 = base64.b64encode(json.dumps(msg).encode())
    client_socket.sendall(msg_base64)

    response_base64 = client_socket.recv(1024)  # Receive data from client
    response_data = base64.b64decode(response_base64).decode()
    js_data = json.loads(response_data)
    selected_roi = js_data['roi']  # Initial bounding box: first frame
    
    # STEP3: Execute trace
    tracker_params['videofile'] = video_path
    tracker_params['optional_box'] = selected_roi

    tracker = Tracker(tracker_name, tracker_param, "video", tracker_params=tracker_params)
    track_res_file = os.path.join(data_path, 'tracked_{}_res.txt'.format(Path(video_name).stem))

    tracking(
        tracker=tracker,
        client_socket=client_socket,
        optional_box=selected_roi,
        video_path=video_path,
        debug=0,
        track_res_file=track_res_file,
    )


def main():
    parser = argparse.ArgumentParser(description='Run the tracker on your webcam.')
    # parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    # parser.add_argument('tracker_param', type=str, help='Name of parameter file.')
    # parser.add_argument('videofile', type=str, help='path to a video file.')
    parser.add_argument('--optional_box', type=float, default=None, nargs="+", help='optional_box with format x y w h.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--save_results', dest='save_results', action='store_true', help='Save bounding boxes')
    parser.set_defaults(save_results=True)

    parser.add_argument('--params__model', type=str, default=None, help="Tracking model path.")
    parser.add_argument('--params__update_interval', type=int, default=None, help="Update interval of online tracking.")
    parser.add_argument('--params__online_sizes', type=int, default=None)
    parser.add_argument('--params__search_area_scale', type=float, default=None)
    parser.add_argument('--params__max_score_decay', type=float, default=1.0)
    parser.add_argument('--params__vis_attn',
                        type=int,
                        choices=[0, 1],
                        default=0,
                        help="Whether visualize the attention maps.")

    args = parser.parse_args()

    args.tracker_name = 'mixformer_cvt_online'
    args.tracker_param = 'baseline'
    args.params__model = 'mixformer_online_22k.pth.tar'
    args.videofile = ''
    args.debug = 0
    args.params__search_area_scale = 4.5
    args.params__update_interval = 10
    args.params__online_sizes = 5

    tracker_params = {}
    for param in list(filter(lambda s: s.split('__')[0] == 'params' and getattr(args, s) != None, args.__dir__())):
        tracker_params[param.split('__')[1]] = getattr(args, param)

    # print(tracker_params)

    host = "0.0.0.0"  # listen on all network interfaces
    port = 12345  # Server port
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(5)

    print(f"Server listening on {host}:{port}")

    while True:
        client_socket, addr = server_socket.accept()  # Receive client connection
        print(f"Client connected from {addr[0]}:{addr[1]}")

        # Create a new thread to handle client connections
        client_thread = threading.Thread(target=run_video,
                                         args=(
                                             client_socket,
                                             args.tracker_name,
                                             args.tracker_param,
                                             tracker_params,
                                         ))
        client_thread.start()


if __name__ == '__main__':
    main()
