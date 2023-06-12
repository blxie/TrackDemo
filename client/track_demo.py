"""
Client functions:
    1. Upload data data
        - video path in client
        - initial bbox
    2. Receive the data returned by the server
        - communication message
        - bbox
"""
import argparse
import base64
import cv2
import os
import socket
import json
import tqdm
import shutil

from pathlib import Path


def run_track(
    client_socket: socket.socket,
    video_capture: cv2.VideoCapture,
    video_path,
    save_img,
    win_name,
):
    selected_roi = None  # Store the coordinates of the rectangle selected by the user
    video_path = Path(video_path)

    res_dir = os.path.join("track_res", Path(video_path).stem)
    img_save_path = os.path.join(res_dir, "images")
    if os.path.exists(img_save_path):
        shutil.rmtree(img_save_path)
    else:
        os.makedirs(img_save_path)

    data = {"video_name": Path(video_path).name}
    data = json.dumps(data).encode()
    client_socket.sendall(base64.b64encode(data))

    i = 1
    while True:
        response_base64 = client_socket.recv(1024)
        response_data = base64.b64decode(response_base64).decode()
        js_data = json.loads(response_data)
        # print(js_data)

        if "upload" in js_data and js_data["upload"] == True:
            file_size = os.path.getsize(video_path)
            file_size_bytes = file_size.to_bytes(4, "big")  # Convert file size to byte sequence of 4 bytes
            client_socket.sendall(file_size_bytes)  # Send file size information

            with open(video_path, "rb") as file:
                progress = tqdm.tqdm(
                    range(file_size),
                    f"Uploading {Path(video_path).name}",
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                )
                while True:
                    data = file.read(1024)  # 1024 bytes of data are read each time
                    if not data:
                        break

                    client_socket.sendall(data)  # Send data
                    progress.update(len(data))

                progress.close()
                print("Upload complete!")
        if i == 1 and "draw_bbox" in js_data and js_data["draw_bbox"]:
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win_name, 960, 540)
            ret, frame = video_capture.read()
            if frame is None:
                break
            cv2.putText(
                frame,
                "Select an ROI and then press SPACE or ENTER button!",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            cv2.imshow(win_name, frame)
            cv2.waitKey(1)

            selected_roi = cv2.selectROI(win_name, frame)
            # Send rectangle coordinates and video file path
            data = {"roi": selected_roi}
            data = json.dumps(data).encode()
            client_socket.sendall(base64.b64encode(data))

            x, y, w, h = selected_roi
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imwrite(f"{img_save_path}/{1:04d}.jpg", frame)
            with open(f"{res_dir}/results.txt", "w") as file:
                bbox_str = f"{x},{y},{w},{h}\n"
                file.write(bbox_str)

        if "bbox" in js_data and js_data["bbox"]:
            i += 1
            ret, frame = video_capture.read()
            if frame is None:
                break
            if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                break
            with open(f"{res_dir}/results.txt", "a") as file:
                bbox_str = f"{x},{y},{w},{h}\n"
                file.write(bbox_str)

            if js_data["score"] > 0.8:
                print("frame@{} >> Track target info: {:.4f} @ {}".format(i, js_data["score"], js_data["bbox"]))
                x, y, w, h = js_data["bbox"]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                cv2.putText(
                    frame,
                    "Tracking failed: the target may be out of view or occluded!",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )

            if save_img:
                cv2.imwrite(f"{img_save_path}/{i:04d}.jpg", frame)

            cv2.imshow(win_name, frame)
            cv2.waitKey(1)
            # After receiving the bbox information, send a confirmation message to the server
            client_socket.sendall(b"ACK")  # send confirmation message

        if "status" in js_data and js_data["status"] == "TRACK_END":
            break  # Tracking finished, end the loop


def main():
    parser = argparse.ArgumentParser(description="Run the tracker on your webcam.")
    parser.add_argument(
        "--video_path",
        type=str,
        default="test.mp4",
        help="The absolute or relative path of the video you want to track.",
    )
    parser.add_argument("--server_ip", type=str, default="", help="Server IP.")
    parser.add_argument("--server_port", type=int, default=12345, help="Server port.")
    parser.add_argument("--save_img", type=bool, default=True, help="If save tracking images.")
    parser.add_argument("--win_name", type=str, default="TrackDemo", help="Windows name.")

    args = parser.parse_args()
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((args.server_ip, args.server_port))
    video_capture = cv2.VideoCapture(args.video_path)

    run_track(
        client_socket,
        video_capture,
        args.video_path,
        args.save_img,
        args.win_name,
    )

    client_socket.close()
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
