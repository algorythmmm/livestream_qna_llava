# ------------------- IMPORT REQUIRED MODULES -------------------

import logging
import cv2
import numpy as np
import yt_dlp
import os
import time
import subprocess
from PIL import Image
from collections import defaultdict
from transformers import pipeline
from ultralytics import YOLO
from flask import Flask, render_template, request, jsonify
from pyngrok import ngrok
from qna import generate_answer_from_frames



# Initialize YOLO model
model = YOLO("yolo11l.pt")  # Load the YOLO model
class_names = model.names   # Get class names from YOLO

# Initialize depth estimation pipeline
depth_pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-base-hf")

# Flask app setup
app = Flask(__name__)

# ------------------- FUNCTION TO FETCH FRAMES -------------------

def fetch_live_frames(url, duration=15, frame_count=3):
        """
    Fetches `frame_count` frames evenly spaced over the given `duration` from the video URL.

    Args:
        url (str): The video URL.
        duration (int): The duration (in seconds) to capture frames.
        frame_count (int): The number of frames to return.

    Returns:
        tuple: (list of PIL.Image frames, subprocess process object)
    """
    
    ydl_opts = {'format': '232', 'quiet': True}  # 720p format
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=False)
            video_stream_url = info_dict['url']

            # Video properties
            width, height = info_dict.get('width', 1280), info_dict.get('height', 720)
            fps = info_dict.get('fps', 30)

            # Start ffmpeg to capture frames
            command = [
                'ffmpeg', '-i', video_stream_url, '-f', 'image2pipe', '-vcodec', 'rawvideo', '-pix_fmt', 'bgr24', '-'
            ]
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8)

            frames_collected = []
            total_frames = int(duration * fps)
            frame_indices = np.linspace(0, total_frames - 1, frame_count, dtype=int)

            for i in range(total_frames):
                raw_frame = process.stdout.read(width * height * 3)
                if not raw_frame:
                    break
                if i in frame_indices:
                    frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3))
                    pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    frames_collected.append(pil_frame)
                if len(frames_collected) == frame_count:
                    break
            
            return frames_collected, process

    except yt_dlp.DownloadError as e:
        print(f"Error downloading video: {e}")
        return [], None

# ------------------- QNA FUNCTION -------------------

def qna_function(video_url, question):
    """Process a YouTube video stream and answer a given question."""
    
    frames, process = fetch_live_frames(video_url, duration=5)
    if frames:
        answer = generate_answer_from_frames(question, frames)
        
        # Close ffmpeg process
        process.stdout.close()
        process.stderr.close()
        process.wait()
        
        return { "Answer": answer }
    
    return {"status": "error", "message": "No frames received."}

# ------------------- CAPTURE FUNCTION -------------------

def capture_function(video_url):
    """Extract a frame from a YouTube video and perform object detection + depth estimation."""
    
    frames, process = fetch_live_frames(video_url, duration=1)
    
    if frames:
        middle_frame = frames[len(frames) // 2]
        results = model.track(source=np.array(middle_frame))
        
        detected_classes = results[0].boxes.cls.cpu().numpy()
        detected_centers = results[0].boxes.xywh.cpu().numpy()

        image = Image.fromarray(cv2.cvtColor(np.array(middle_frame), cv2.COLOR_BGR2RGB))
        depth_result = depth_pipe(image)
        depth_map_disparity = np.array(depth_result["depth"])
        
        # Convert disparity map to actual depth values
        depth_map = (1) / (depth_map_disparity + 1.0)
        normalized_depth_map = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map)) * 10

        log_info = []
        class_counts = defaultdict(int)

        for idx, class_id in enumerate(detected_classes):
            class_name = class_names[int(class_id)]
            center_x, center_y = detected_centers[idx][0], detected_centers[idx][1]

            if 0 <= int(center_x) < normalized_depth_map.shape[1] and 0 <= int(center_y) < normalized_depth_map.shape[0]:
                depth_value = normalized_depth_map[int(center_y), int(center_x)]
            else:
                depth_value = 0

            class_counts[class_name] += 1
            log_info.append(f"{class_name.capitalize()} no. {class_counts[class_name]} at ({center_x:.2f}, {center_y:.2f}) with depth {depth_value:.2f}m")

        # Close ffmpeg process
        process.stdout.close()
        process.stderr.close()
        process.wait()

        return { "Scene description stats": log_info }
    
    return {"status": "error", "message": "No frames received."}

# ------------------- FLASK ROUTES -------------------

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        video_url = request.form.get("video_url")
        action = request.form.get("action")
        question = request.form.get("question", None)

        if action == "qna":
            return jsonify(qna_function(video_url, question))
        elif action == "capture":
            return jsonify(capture_function(video_url))
        
        return jsonify({"status": "error", "message": "Invalid action."})
    
    return render_template("index.html")


# ------------------- RUN FLASK -------------------

if __name__ == "__main__":
    app.run(debug=True)
