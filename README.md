# 🎭 Texture Overlay Prank Camera

> **Turn real-world objects in your webcam into portals to another texture — live.**

Texture Prank Camera is a real-time computer-vision project built with **YOLO segmentation + OpenCV + Python + OBS Virtual Camera**.

Point your webcam at a tie, person, bicycle, backpack, etc., select the object, hit `T`, and watch it slowly transform into an image from **left to right** over ~18 seconds.

The rest of the webcam stays completely normal.

The processed video can be exposed as a virtual webcam and used in applications such as Google Meet, Zoom, Discord, etc.

---

## 🎬 What does it look like?

Imagine you're on a video call wearing a normal tie.

You press `T`.

Over the next ~18 seconds:

```text
REAL TIE

████░░░░░░
████░░░░░░

██████░░░░
██████░░░░

████████░░
████████░░

██████████
██████████

TEXTURE COMPLETE
```

Except the `████` isn't a simple color — it's your chosen texture image.

Your friend sees the transformation happening live.

The rest of the video remains your normal webcam.

---

# ✨ Features

* 🧠 YOLO real-time object segmentation
* 🎯 Select from YOLO's detectable object classes
* 👔 Defaults to `tie`
* 🔄 Change the target object while the program is running
* 🎨 Replace the detected object with any texture/image
* ⏳ Slow ~18-second left-to-right transformation
* ↩️ Instant reset with `R`
* 🎥 Outputs through **OBS Virtual Camera**
* 💻 Works with webcam applications that support virtual cameras
* 🎛️ Local controller/dropdown
* 🔒 Controller UI is **not included in the outgoing video**
* 🪶 Temporal smoothing helps stabilize the segmentation mask

---

# 🧩 How it works

The core pipeline is:

```text
                Physical Webcam
                       │
                       ▼
                ┌────────────┐
                │    YOLO    │
                │ Segmentation│
                └──────┬─────┘
                       │
                       ▼
                Selected Object
                       │
                       ▼
              ┌─────────────────┐
              │ Texture Engine  │
              │                 │
              │ Left → Right    │
              │ ~18 seconds     │
              └────────┬────────┘
                       │
                       ▼
                 Clean Video
                       │
                       ▼
              OBS Virtual Camera
                       │
             ┌─────────┼─────────┐
             ▼         ▼         ▼
           Meet       Zoom     Discord
```

The controller is separate:

```text
┌──────────────────────────────┐
│ Object: [ tie             ▼ ]│
│                              │
│       Webcam Preview         │
│                              │
│ T = transform                │
│ R = reset                    │
└──────────────────────────────┘
```

The dropdown and controls are drawn **only on the local controller window**.

They are never sent to the virtual camera.

---

# 🖥️ Requirements

## Hardware

You'll need:

* A webcam
* A Windows PC
* A reasonably modern CPU
* Enough performance to run YOLO in real time

A dedicated GPU is helpful, but isn't strictly required.

Performance will depend heavily on your hardware, camera resolution, and YOLO model.

---

# 🐍 Software

Recommended:

* Windows 10/11
* Python 3.11
* OBS Studio
* A webcam application that supports virtual cameras

This project currently uses:

* Python
* Ultralytics YOLO
* OpenCV
* NumPy
* pyvirtualcam

---

# 📦 Installation

## 1. Clone the repository

```bash
git clone <YOUR-REPOSITORY-URL>
cd texture-prank-camera
```

Or download the repository as a ZIP and extract it.

---

## 2. Create a virtual environment

Highly recommended.

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\activate
```

You should see something like:

```text
(.venv) PS C:\...\texture-prank-camera>
```

---

## 3. Install Python dependencies

```powershell
pip install ultralytics opencv-python numpy pyvirtualcam
```

You can verify the installation:

```powershell
python -c "import cv2, numpy, ultralytics, pyvirtualcam; print('Everything works!')"
```

---

# 📹 Install OBS Studio

Install OBS Studio from:

[OBS Studio](https://obsproject.com/?utm_source=chatgpt.com)

For this project, **you don't need to create an OBS scene**.

You don't need to:

* Add your webcam to OBS
* Add a camera source
* Add the Python window
* Screen capture anything
* Build an OBS layout

OBS is being used for its **Virtual Camera**.

The Python program sends its processed frames directly to the OBS Virtual Camera through `pyvirtualcam`.

---

# 📁 Project structure

Your project should look approximately like:

```text
texture-prank-camera/
│
├── Tie_textureFill.py
├── yolov8n-seg.pt
├── _1.png
├── README.md
└── .venv/
```

### `Tie_textureFill.py`

The main application.

### `yolov8n-seg.pt`

YOLO segmentation model.

### `_1.png`

The texture displayed inside detected objects.

You can replace this with your own image.

---

# 🤖 YOLO model

The default model is:

```text
yolov8n-seg.pt
```

The `n` version is the small/fast model and is useful for real-time webcam applications.

You can experiment with larger segmentation models if your hardware can handle them.

Larger models may improve detection but can reduce real-time performance.

---

# ▶️ Running the prank camera

Start your virtual environment:

```powershell
.venv\Scripts\activate
```

Then:

```powershell
python -m Tie_textureFill
```

You should see:

```text
Loading YOLO model...
YOLO loaded.
Texture loaded.
Default object: tie
Opening webcam...
Webcam resolution: 1280x720

Starting OBS Virtual Camera...

Virtual camera started: OBS Virtual Camera

============================================
 PRANK CAMERA READY
============================================

T = Transform
R = Reset
Q = Quit

Select 'OBS Virtual Camera' in Google Meet.
```

The important line is:

```text
Virtual camera started: OBS Virtual Camera
```

If you see that, Python successfully connected to the virtual camera.

---

# 🎛️ Controls

## `T` — Transform

Press:

```text
T
```

to begin the transformation.

The selected object gradually changes from the webcam image to the texture.

The effect takes approximately **18 seconds**.

---

## `R` — Reset

Press:

```text
R
```

to immediately restore the normal webcam appearance.

---

## `Q` — Quit

Press:

```text
Q
```

to stop the application.

The webcam and virtual camera will be released automatically.

---

# 🔽 Selecting an object

The controller contains a dropdown of YOLO's available classes.

The default target is:

```text
tie
```

Depending on the model, you may see classes such as:

```text
person
bicycle
car
backpack
handbag
bottle
cup
tie
...
```

Select an object from the dropdown and press `T`.

You can switch objects while the application is running.

---

# 🎨 Changing the texture

Replace:

```text
_1.png
```

with your own image.

For example:

```text
texture.png
```

Then change:

```python
TEXTURE_PATH = "_1.png"
```

to:

```python
TEXTURE_PATH = "texture.png"
```

You can use:

* Photos
* Patterns
* Brick walls
* Wood
* Space
* Memes
* Game textures
* Screenshots
* Abstract art
* Basically anything you want to make appear "inside" the object

The texture is resized to fill the entire camera frame.

---

# 🎥 Using it in Google Meet

Start the Python application first.

Then open:

[Google Meet](https://meet.google.com/?utm_source=chatgpt.com)

When selecting your camera, choose:

```text
OBS Virtual Camera
```

Your video pipeline is now:

```text
Webcam
   ↓
Texture Prank Camera
   ↓
OBS Virtual Camera
   ↓
Google Meet
   ↓
Everyone else
```

You can test this privately before joining an actual call.

---

# 🎥 Using it in Zoom / Discord / other applications

The same principle applies to any application that allows you to select a webcam.

Look for a camera option named:

```text
OBS Virtual Camera
```

Select it instead of your physical webcam.

The application receives the processed video.

---

# 🧪 Recommended first test

Before attempting the prank during a call:

### 1. Run the program

```powershell
python -m Tie_textureFill
```

### 2. Confirm

```text
Virtual camera started: OBS Virtual Camera
```

### 3. Select `person`

Stand in front of the webcam.

### 4. Press `T`

You should see the texture gradually sweep across your body.

### 5. Press `R`

Everything should return to normal.

### 6. Try `tie`

Stand relatively still and make sure your tie is visible.

### 7. Try other objects

Experiment with:

```text
bicycle
backpack
handbag
cup
bottle
```

---

# ⚠️ Object detection limitations

This is important.

The dropdown contains **YOLO classes**, not guaranteed targets.

Selecting:

```text
cup
```

doesn't mean YOLO will necessarily detect every cup.

The texture effect only works when YOLO successfully produces a segmentation mask.

### Generally easier targets

Large, obvious objects tend to work better:

```text
person       🟢
car          🟢
bicycle      🟢/🟡
tie          🟢/🟡
```

### Potentially difficult targets

Small or partially obscured objects:

```text
cup          🟡/🔴
handbag      🟡/🔴
small items  🔴
```

Detection depends on:

* Lighting
* Object size
* Camera resolution
* Object angle
* Occlusion
* Motion blur
* Background
* YOLO model
* Confidence threshold

So if `person` works perfectly but `cup` doesn't, **the virtual-camera system isn't necessarily broken**.

YOLO may simply not be producing a usable mask.

---

# 🔧 Improving detection

The current configuration uses:

```python
CONFIDENCE = 0.4
```

If you're having trouble detecting an object, try:

```python
CONFIDENCE = 0.3
```

or:

```python
CONFIDENCE = 0.25
```

Lowering the threshold can make weaker detections appear.

However, there's a tradeoff:

```text
Lower confidence
       ↓
More detections
       ↓
More false positives
```

For a prank, a little experimentation is part of the fun.

---

# 🧍 Tips for a reliable prank

For the best results:

### Stay relatively still

The current version uses temporal smoothing, but sudden movement can still cause the mask to jump.

### Make the object large

Objects closer to the camera are easier to segment.

### Use decent lighting

YOLO doesn't perform miracles in a dark room.

### Don't cover the object

If your tie is hidden behind your jacket, don't expect segmentation magic.

### Test beforehand

Don't discover that your particular object isn't detected **after** you've already started the call.

---

# 😈 The prank recipe

Here's the intended workflow:

```text
1. Start the program

2. Select OBS Virtual Camera
   in your call application

3. Verify your normal webcam works

4. Select "tie"

5. Join your call

6. Act completely normal

7. Wait...

8. Press T

9. Let the transformation happen

10. Enjoy the confusion

11. Press R

12. Pretend nothing happened
```

For maximum effect, **don't announce that you're running an effect**.

The slow transformation is the entire joke.

---

# 🧠 Why the effect works

This isn't actually replacing the physical object.

The program is creating a **mask** around the detected object.

Conceptually:

```text
Original frame:

AAAAAAAAAAAAAAAAAAAA
AAAAAAA  OBJECT AAAAA
AAAAAAA  OBJECT AAAAA
AAAAAAA  OBJECT AAAAA
AAAAAAAAAAAAAAAAAAAA
```

YOLO produces:

```text
00000000000000000000
00000001111100000000
00000001111100000000
00000001111100000000
00000000000000000000
```

The program then performs:

```text
output[mask] = texture[mask]
```

So:

```text
Everything outside mask
        ↓
normal webcam

Everything inside mask
        ↓
texture
```

That's why the rest of your camera remains completely normal.

---

# 🪄 How the animation works

The current transformation isn't an instant replacement.

The program calculates a moving vertical boundary:

```text
              transformation →

████░░░░░░░░░
████░░░░░░░░░
████░░░░░░░░░

       ↓

████████░░░░░
████████░░░░░
████████░░░░░

       ↓

█████████████
█████████████
█████████████
```

Only the portion of the detected object that has been crossed by the animation is replaced.

The result is a gradual left-to-right "conversion."

---

# 🚧 Current limitations

This project is intentionally a prototype.

Current limitations include:

* YOLO can lose an object during fast movement
* Small objects may not be detected
* Partially hidden objects may fail
* Thin objects can be difficult to segment
* Poor lighting reduces reliability
* The current effect isn't true 3D surface mapping
* The texture is a 2D image projected through the object's mask
* CPU performance may limit frame rate
* Different YOLO models have different class support
* The transformation currently uses the object's image-space bounding region

In other words:

**This is computer vision doing a magic trick, not actual object tracking/replacement.**

And that's part of the fun.

---

# 🚀 Ideas for future versions

There are a ridiculous number of ways this could be improved.

### Better tracking

Use object tracking between YOLO detections so the effect doesn't disappear when detection briefly fails.

### Better models

Try larger segmentation models for difficult objects.

### Custom textures per object

```text
tie       → galaxy.jpg
person    → brick.jpg
backpack  → lava.jpg
bicycle   → ocean.jpg
```

### Random prank mode

Automatically select an object and texture.

### Delayed activation

Wait several seconds before starting the transformation.

### More dramatic transitions

Instead of a simple sweep:

```text
left → right
```

try:

```text
wave
glitch
pixelation
liquid
burn
scanline
dissolve
```

### Multiple objects

Transform several detected objects simultaneously.

### Persistent tracking

Keep the effect attached to an object even when YOLO temporarily loses it.

### Better geometric mapping

Instead of treating the texture as a flat window, warp it according to the object's shape and movement.

---

# 🛠️ Troubleshooting

## `Could not open webcam`

Check that:

* Your webcam isn't being used exclusively by another application.
* `CAMERA_INDEX` is correct.
* Windows has granted Python camera permissions.

---

## `Could not load texture image`

Make sure the texture exists:

```text
texture.png
```

and that:

```python
TEXTURE_PATH
```

matches the filename.

---

## `Could not load YOLO model`

Make sure:

```text
yolov8n-seg.pt
```

is present or that the configured model path is correct.

---

## `OBS Virtual Camera` doesn't appear

Make sure OBS Studio is installed correctly and that the Virtual Camera component is available.

Then restart the Python program.

If necessary, restart OBS/Windows and try again.

---

## Object isn't transforming

First test:

```text
person
```

If `person` works, your pipeline is probably fine.

Then try moving closer to the camera and improving lighting.

You can also lower:

```python
CONFIDENCE = 0.4
```

to:

```python
CONFIDENCE = 0.3
```

---

# 📜 License

Add your preferred license here.

For example:

```text
MIT License
```

If you intend to publish this publicly, make sure you also comply with the licenses of the models and libraries you're using.

---

# ⚠️ Disclaimer

This project is intended for **harmless experimentation, computer-vision demos, and consensual pranks**.

Don't use it to impersonate someone, deceive people in situations where deception could cause harm, or bypass security/identity checks.

Have fun. Don't become the reason someone has to explain a fake texture-covered person to HR.

---

# ❤️ Credits

Built with:

* **Ultralytics YOLO** — object detection and segmentation
* **OpenCV** — webcam processing and rendering
* **NumPy** — image/mask operations
* **pyvirtualcam** — virtual camera output
* **OBS Studio** — virtual camera infrastructure

---

## 🎭 Have fun

The basic idea is simple:

> **Detect something → turn it into something else → pretend nothing happened.**

That's it.

Point camera.

Select object.

Press `T`.

Wait.

Watch the confusion begin. 😈
