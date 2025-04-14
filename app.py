from flask import Flask, render_template, Response, request, send_file
import cv2
import numpy as np
import time
import os

app = Flask(__name__, template_folder='my_html')

# Define color ranges (HSV format) and corresponding BGR values
myColors = [
    [105, 180, 50, 130, 255, 255],  # Blue
    [0, 150, 120, 10, 255, 255],    # Red
    [50, 100, 100, 80, 255, 255],   # Green
    [10, 150, 120, 25, 255, 255],   # Orange
    [40, 70, 70, 80, 255, 255]      # Light Green
]
myColorValues = [
    [177, 77, 9],   # Blue
    [0, 0, 255],    # Red
    [0, 255, 0],    # Green
    [0, 140, 255],  # Orange
    [144, 238, 144] # Light Green
]

myPoints = []
brushThickness = 10
cap = cv2.VideoCapture(0)

# Check if webcam is working
if not cap.isOpened():
    print("Error: Could not access the webcam.")

def getContours(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # print(f"Contours found: {len(contours)}")  # Debugging line
    x = y = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            return x + w // 2, y
    return 0, 0

def findColor(img):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    newPoints = []
    for i, color in enumerate(myColors):
        lower = np.array(color[0:3])
        upper = np.array(color[3:6])
        mask = cv2.inRange(imgHSV, lower, upper)

        # Debug: Show the mask for each color detection
        cv2.imshow(f"Mask {i}", mask)

        x, y = getContours(mask)
        if x != 0 and y != 0:
            newPoints.append([x, y, i])
            cv2.circle(img, (x, y), brushThickness, myColorValues[i], cv2.FILLED)
    return newPoints

def drawCanvas(img):
    for x, y, colorId in myPoints:
        cv2.circle(img, (x, y), brushThickness, myColorValues[colorId], cv2.FILLED)

def generate_frames():
    global myPoints
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)  # Flip the frame to mirror the video
        newPoints = findColor(frame)
        if newPoints:
            myPoints.extend(newPoints)
        drawCanvas(frame)

        cv2.putText(frame, f'Brush: {brushThickness}px', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

        # Encode the frame in JPEG format for streaming
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    try:
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print(f"Error while streaming video: {e}")


        return "Error while streaming video"

@app.route('/clear')
def clear():
    myPoints.clear()
    return "Canvas Cleared"

@app.route('/set_brush')
def set_brush():
    global brushThickness
    brushThickness = int(request.args.get('size', 10))
    print(f"Brush size updated to: {brushThickness}")  # Debugging line
    return "Brush size updated"

@app.route('/custom_color')
def custom_color():
    global myColorValues, myColors
    hex = request.args.get('hex')
    bgr = [int(hex[i:i+2], 16) for i in (4, 2, 0)]
    myColorValues.append(bgr)
    myColors.append([0, 0, 0, 179, 255, 255])  # broad HSV range for color tracking
    return "Custom color added"

@app.route('/set_eraser')
def set_eraser():
    myColorValues.append([255, 255, 255])  # white
    myColors.append([0, 0, 0, 179, 255, 255])
    return "Eraser mode set"

@app.route('/snapshot')
def snapshot():
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        drawCanvas(frame)
        filename = f"snapshot_{int(time.time())}.png"
        cv2.imwrite(filename, frame)
        return send_file(filename, mimetype='image/png')
    return "Snapshot failed"

if __name__ == '__main__':
    app.run(debug=True)
