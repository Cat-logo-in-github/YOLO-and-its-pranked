# 🪄 Virtual Prank Camera

> **Turn ordinary webcam calls into questionable reality.**
> YOLO segmentation + OpenCV + OBS Virtual Camera = silly computer-vision pranks for Zoom, Google Meet, and other webcam apps.

This project lets you selectively manipulate objects detected by your webcam in real time.

You can make:

* 👔 A tie become a completely different tie
* 🍎 An object become another object/texture
* 🪄 A subtle "something is wrong with that..." effect appear on an object
* 🌀 A small region of an object gradually invert its colors

The processed camera feed can be sent directly to **OBS Virtual Camera**, allowing the effect to appear inside video-call applications.

---

## ⚠️ Disclaimer

This project is intended for **harmless jokes and experimentation with computer vision**.

Don't use it to impersonate people, deceive someone in a consequential situation, or interfere with meetings, exams, interviews, or other situations where the deception could cause harm.

The best prank is one where everyone gets a laugh when you reveal it. 😄

---

# ✨ What This Project Does

The basic pipeline is:

```text
                ┌──────────────────┐
                │   Physical       │
                │   Webcam         │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ Python + OpenCV  │
                │                  │
                │ YOLO Segmentation│
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │  Prank Effect    │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ OBS Virtual      │
                │ Camera           │
                └────────┬─────────┘
                         │
                ┌────────┴────────┐
                ▼                 ▼
             Zoom             Google Meet
```

The important part is that **your normal webcam remains the background**.

Only the pixels belonging to the detected object are modified.

---

# 🎭 The Three Effects

This repository contains three different approaches.

## 1. Texture Fill

### File

```text
Tie_textureFill.py
```

This is the simplest and probably the most reliable effect.

YOLO detects an object and the program replaces the detected object's pixels with a full-screen texture.

For example:

```text
Normal webcam

       👔
       ↓

YOLO detects tie

       👔
       ↓

Texture appears inside tie
```

The rest of the webcam remains completely normal.

### Features

* Real-time object segmentation
* Selectable target object
* Defaults to `tie`
* Live dropdown
* `T` to activate
* `R` to reset
* `Q` to quit
* OBS Virtual Camera support

---

# 2. Object → Object Replacement

### File

```text
tie_replace.py
```

This version takes the idea further.

You can choose:

### Target object

The object in your webcam that should be modified.

For example:

```text
person
tie
cup
handbag
apple
...
```

### Replacement object

The object/appearance that should be placed onto it.

For example:

```text
apple
tie
cup
...
```

This allows things like:

```text
             webcam
               │
               ▼
            👤 PERSON
               │
               ▼
         ┌─────────────┐
         │ segmentation│
         └──────┬──────┘
                │
                ▼
             🍎 APPLE
```

So theoretically you can make:

> "Why does your entire body look like an apple?"

which is, scientifically speaking, extremely important research.

### Important

The first dropdown can use **all classes available to the YOLO model**.

The replacement source is restricted to objects that actually exist in the supplied source image.

Detection quality depends heavily on YOLO.

If YOLO cannot confidently segment:

```text
handbag
cup
small object
partially hidden object
```

the effect may be unstable or may not trigger.

That's a limitation of the detection model rather than the virtual-camera system.

---

# 3. The "Something Is Wrong With Your Tie" Effect

### File

```text
subtle_prank.py
```

This is the sneaky one.

Instead of replacing the entire object, the program creates a small irregular region **inside the detected object**.

The current effect gradually inverts the colors in that region.

Example:

```text
Normal:

       👔
       │
       │
       │


After several seconds:

       👔
       │
       ▓
       │
```

The goal isn't to make the whole object look like a filter.

It's supposed to produce a:

> "Wait... is there something weird on your tie?"

moment.

The effect gradually appears over approximately 15 seconds.

### Controls

```text
T = Start effect
R = Reset
Q = Quit
```

The inversion is constrained by the YOLO segmentation mask, so it only affects the selected object.

---

# 🖥️ Requirements

## Hardware

A normal webcam is sufficient.

Recommended:

* 720p webcam or better
* reasonably modern CPU/GPU
* stable lighting
* relatively little motion

GPU acceleration is helpful but not strictly required.

---

# 🐍 Python

Python **3.11** is recommended for this project.

Create a virtual environment:

```powershell
python -m venv tflowplay
```

Activate it on Windows:

```powershell
.\tflowplay\Scripts\Activate.ps1
```

You should then see something similar to:

```text
(tflowplay) PS C:\...
```

---

# 📦 Install Dependencies

Install the required packages:

```powershell
pip install ultralytics opencv-python numpy pyvirtualcam
```

If you're using the OBS Virtual Camera pipeline, `pyvirtualcam` is required:

```powershell
pip install pyvirtualcam
```

---

# 🤖 YOLO Model

The project uses an Ultralytics YOLO segmentation model.

For example:

```text
yolov8n-seg.pt
```

Place the model in the project directory:

```text
your-project/
│
├── yolov8n-seg.pt
├── Tie_textureFill.py
├── tie_replace.py
├── subtle_prank.py
└── texture.png
```

The model needs to be a **segmentation** model.

A normal detection-only YOLO model won't provide the pixel masks required by these effects.

---

# 🖼️ Texture Images

For the texture effect, place your desired texture image in the project directory.

For example:

```text
texture.png
```

Then configure:

```python
TEXTURE_PATH = "texture.png"
```

You can use practically anything:

```text
marble
wood
grass
the moon
a brick wall
a horrible floral pattern
```

The texture is resized to the webcam resolution and only sampled through the detected object's mask.

---

# 📷 Running Without Zoom/Meet

Before involving OBS, test the effect by itself.

For example:

```powershell
python Tie_textureFill.py
```

or:

```powershell
python tie_replace.py
```

or:

```powershell
python subtle_prank.py
```

You should see the processed webcam output.

---

# 🎥 OBS Virtual Camera

The magic that makes this work in Zoom/Meet is **OBS Virtual Camera**.

Install OBS Studio.

You don't need to build a complicated OBS scene for the `pyvirtualcam` versions.

The Python program itself sends frames directly to:

```text
OBS Virtual Camera
```

When the Python script starts successfully, you should see something similar to:

```text
Starting OBS Virtual Camera...

Virtual camera started: OBS Virtual Camera
```

At that point, your processed video is available as a camera device.

---

# 🟢 Using It With Google Meet

Start the prank program:

```powershell
python subtle_prank.py
```

Wait until you see:

```text
Virtual camera started: OBS Virtual Camera
```

Then open Google Meet.

In the camera settings, select:

```text
OBS Virtual Camera
```

Your call should now receive the processed feed.

The local controller window is separate from the camera output.

Therefore:

```text
Your screen:

┌───────────────────────────────┐
│ Prank Controller              │
│                               │
│ Target: tie                   │
│ T = effect                    │
│ R = reset                     │
└───────────────────────────────┘


Google Meet receives:

┌───────────────────────────────┐
│                               │
│       CLEAN PROCESSED         │
│       WEBCAM VIDEO            │
│                               │
└───────────────────────────────┘
```

The dropdown/controller isn't supposed to appear in the call.

---

# 🔵 Using It With Zoom

Same basic procedure.

Run the Python program first:

```powershell
python subtle_prank.py
```

Then in Zoom:

```text
Settings
   ↓
Video
   ↓
Camera
   ↓
OBS Virtual Camera
```

Zoom will receive the processed webcam feed.

---

# 🎛️ Controls

The current versions use:

| Key | Action               |
| --- | -------------------- |
| `T` | Trigger prank effect |
| `R` | Reset effect         |
| `Q` | Quit                 |

The target object can be changed using the on-screen dropdown.

For example:

```text
TARGET OBJECT

┌──────────────────────────┐
│ tie                   ▼  │
└──────────────────────────┘
```

Click it and select another YOLO class.

Depending on the model, available classes can include things such as:

```text
person
tie
cup
handbag
bottle
apple
chair
laptop
cell phone
book
...
```

Not every object will work equally well.

---

# 🧪 Recommended Prank Workflow

For maximum comedy, don't immediately start with something ridiculous.

Instead:

### Step 1 — Start normal

Join the call normally.

Don't mention the software.

### Step 2 — Let the camera stabilize

YOLO needs a reasonably clear view of the object.

Good lighting helps considerably.

### Step 3 — Pick your target

For example:

```text
tie
```

### Step 4 — Start the effect

Press:

```text
T
```

### Step 5 — Act completely normal

This is important.

Don't stare at your friend's reaction.

Continue talking normally.

### Step 6 — Let them notice

The subtle effect is deliberately slow.

The intended reaction is:

> "Wait..."

followed by:

> "What is wrong with your tie?"

### Step 7 — Reveal

After they've had their moment:

> "It's computer vision."

Then turn the effect off with:

```text
R
```

---

# 🧠 Why Detection Quality Matters

These effects rely on YOLO segmentation.

The system isn't actually "understanding" the object like a human does.

It's producing a pixel mask such as:

```text
0 0 0 0 0
0 1 1 1 0
0 1 1 1 0
0 1 1 1 0
0 0 0 0 0
```

The prank effect is then applied only where the mask contains the object.

Therefore:

### Good conditions

```text
Bright room
+
Large object
+
Object facing camera
+
Little movement
=
Excellent results
```

### Bad conditions

```text
Dark room
+
Tiny object
+
Object partially hidden
+
Fast movement
=
YOLO having an existential crisis
```

---

# 🎯 Tips For Better Detection

## Lighting

Use decent lighting.

Avoid putting the target completely in shadow.

## Distance

Objects should occupy a reasonable portion of the camera frame.

A tiny cup 4 meters away isn't an ideal YOLO target.

## Movement

The current system uses temporal smoothing:

```python
SMOOTHING_ALPHA = 0.7
```

This helps reduce jitter but also means extremely fast movement can cause the mask to lag.

## Confidence

The default is:

```python
CONFIDENCE = 0.4
```

If an object isn't being detected reliably, you can experiment with:

```python
CONFIDENCE = 0.3
```

or:

```python
CONFIDENCE = 0.5
```

Lower isn't necessarily better—it can also produce incorrect detections.

---

# 🛠️ Troubleshooting

## "Could not open webcam"

Check:

* Another application isn't exclusively using the camera.
* Windows has granted camera permission.
* Your camera index is correct.

Try:

```python
CAMERA_INDEX = 0
```

If you have multiple cameras, you may need:

```python
CAMERA_INDEX = 1
```

---

## OBS Virtual Camera doesn't appear

Make sure OBS Studio is installed correctly.

On Windows, restart OBS/the Python application if necessary.

The program should report:

```text
Virtual camera started: OBS Virtual Camera
```

---

## Zoom/Meet shows a black screen

First test the Python output **without opening Zoom/Meet**.

Then check that Zoom/Meet is using:

```text
OBS Virtual Camera
```

and not the physical webcam.

---

## The object isn't detected

This is probably YOLO rather than OBS.

Try:

* better lighting
* moving closer to the camera
* reducing `CONFIDENCE`
* keeping the object unobstructed
* reducing rapid movement

---

## The effect jitters

Try increasing:

```python
SMOOTHING_ALPHA
```

For example:

```python
SMOOTHING_ALPHA = 0.8
```

Higher values generally produce smoother masks but can make tracking feel more sluggish.

---

## The effect is too subtle

For the inversion effect:

```python
MAX_STRENGTH = 0.85
```

Increase toward:

```python
MAX_STRENGTH = 1.0
```

For testing, you can also temporarily shorten:

```python
EFFECT_DURATION = 3.0
```

Once you've confirmed the effect works, restore the longer duration.

---

# 📁 Suggested Project Structure

```text
virtual-prank-camera/
│
├── yolov8n-seg.pt
│
├── Tie_textureFill.py
├── tie_replace.py
├── subtle_prank.py
│
├── texture.png
│
├── README.md
│
└── requirements.txt
```

A basic `requirements.txt` could contain:

```text
ultralytics
opencv-python
numpy
pyvirtualcam
```

Then installation becomes:

```powershell
pip install -r requirements.txt
```

---

# 🚀 Future Ideas

This project could get considerably more ridiculous.

Potential future versions:

* Global keyboard controls
* Animated left-to-right texture replacement
* Multiple simultaneous target objects
* Object-specific effects
* Slowly changing object colors
* "Reality glitch" effects
* Custom segmentation models
* Better object tracking
* GPU acceleration
* A proper prank dashboard
* One-click launcher
* Automatic camera selection
* More convincing material transformations

The ultimate goal:

```text
Normal webcam
      ↓
"Something seems slightly weird..."
      ↓
"Wait."
      ↓
"WHAT THE HELL?"
      ↓
😂
```

---

# ⭐ Credits / Technologies

This project is built around:

* **Ultralytics YOLO** — object detection and segmentation
* **OpenCV** — webcam processing and image manipulation
* **NumPy** — pixel-level operations
* **pyvirtualcam** — sending processed frames to a virtual camera
* **OBS Virtual Camera** — exposing the processed stream to video-call applications

---

# 🧙 Final Prankster Rule

**The less you react, the better the prank works.**

Don't say:

> "Hey guys, look at my crazy camera!"

Say nothing.

Let someone notice.

Then, when they finally ask:

> "Why is your tie doing that?"

just respond:

> "Doing what?"

And continue the conversation.

**Happy pranking. 🪄**
