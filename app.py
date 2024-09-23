import argparse
import io
from PIL import Image
import datetime
import torch
import cv2
import numpy as np
import tensorflow as tf
from re import DEBUG, sub
from flask import Flask, render_template, request, redirect, send_file, url_for, Response
from werkzeug.utils import secure_filename, send_from_directory
import os
import subprocess
from subprocess import Popen
import re
import requests
import shutil
import time
import glob

from ultralytics import YOLO

app = Flask(__name__)

# Initialize YOLO model globally
model = YOLO('best.pt')

@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route("/", methods=["GET", "POST"])
def predict_img():
    if request.method == "POST":
        if 'file' in request.files:
            f = request.files['file']
            basepath = os.path.dirname(__file__)

            # Ensure the 'uploads' directory exists
            upload_folder = os.path.join(basepath, 'uploads')
            if not os.path.exists(upload_folder):
                os.makedirs(upload_folder)

            filepath = os.path.join(upload_folder, secure_filename(f.filename))
            print("Upload folder is:", filepath)

            f.save(filepath)  # Save the uploaded file to the 'uploads' directory

            global imgpath
            predict_img.imgpath = f.filename
            print("Printing predict_img :::::: ", predict_img)

            file_extension = f.filename.rsplit('.', 1)[1].lower()

            if file_extension in ['jpg', 'jpeg', 'png']:
                img = cv2.imread(filepath)

                # Perform the detection
                detections = model(img, save=True)

                return display(f.filename)

            elif file_extension == 'mp4':
                return detect_video(filepath)

    return render_template('index.html')

@app.route("/video_feed")
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(get_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def get_frame():
    """Capture frames from the webcam and apply YOLO object detection."""
    cap = cv2.VideoCapture(0)  # Open webcam (0 for default camera)

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Apply YOLO object detection on the frame
        results = model(frame)

        # Draw bounding boxes and labels on the frame
        frame_with_detections = results[0].plot()

        ret, jpeg = cv2.imencode('.jpg', frame_with_detections)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

    cap.release()

@app.route('/<path:filename>')
def display(filename):
    folder_path = 'runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
    directory = folder_path + '/' + latest_subfolder
    print("printing directory: ", directory)
    files = os.listdir(directory)
    latest_file = files[0]

    filename = os.path.join(folder_path, latest_subfolder, latest_file)

    file_extension = filename.rsplit('.', 1)[1].lower()

    environ = request.environ
    if file_extension == 'jpg':
        return send_from_directory(directory, latest_file, environ)  # shows the result in separate tab
    else:
        return "Invalid file format"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing YOLO models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=args.port)
