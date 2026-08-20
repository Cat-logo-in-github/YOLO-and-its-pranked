# Texture Window Prank Camera 🎭📹

A real-time computer-vision prank camera that detects objects using YOLO segmentation and replaces the detected object with a texture from `texture.png`.

The rest of the image remains the **live webcam feed**.

For example:

```text
              LIVE WEBCAM

       ┌─────────────────────────┐
       │                         │
       │         face            │
       │                         │
       │        shirt            │
       │          ███            │
       │          ███ ← texture  │
       │          ███            │
       │                         │
       └─────────────────────────┘
```

The object effectively becomes a **window into the texture image**.

---

## ✨ Features

* Real-time webcam processing
* YOLO segmentation
* Detects all classes supported by the YOLO segmentation model
* Interactive object dropdown
* Defaults to `tie`
* Change the selected object while the program is running
* Multiple detected instances can be textured simultaneously
* Full-frame texture mapping
* Temporal mask smoothing
* Designed to work as a virtual webcam for Zoom, Discord, Teams, etc.
* Optional slow transformation effect

---

# 1. What you need

## Hardware

You need:

* A webcam
* A reasonably capable computer
* Ideally an NVIDIA GPU, although YOLO can run on CPU

A GPU is strongly recommended for a smooth real-time effect.

---

# 2. Install Python

Use Python 3.10 or newer.

Check your Python installation:

```bash
python --version
```

or:

```bash
py --version
```

---

# 3. Create a project folder

For example:

```text
texture-prank/
│
├── Tie_textureFill.py
├── yolov8n-seg.pt
├── texture.png
└── README.md
```

`texture.png` is the image that will appear through the detected object.

It can be anything:

* another room
* a photograph
* a pattern
* a meme
* a different shirt
* a bizarre texture
* a video-game texture
* a picture of something completely unrelated

---

# 4. Install the Python packages

Open PowerShell or a terminal inside the project directory.

Install Ultralytics:

```bash
pip install ultralytics
```

Install OpenCV:

```bash
pip install opencv-python
```

Install NumPy:

```bash
pip install numpy
```

Then verify:

```bash
python -c "import cv2, numpy, ultralytics; print('Everything works!')"
```

---

# 5. Get the YOLO segmentation model

The project currently uses:

```text
yolov8n-seg.pt
```

Place the model file in the same directory as the Python script.

The `-seg` version is important because we need **segmentation masks**, not just bounding boxes.

A bounding box would look like:

```text
┌───────────────┐
│               │
│      tie      │
│               │
└───────────────┘
```

We instead need:

```text
        /\
       /  \
      / ██ \
     / ████ \
    /  ████  \
       ███
       ███
```

Only the actual object pixels should be replaced.

---

# 6. Add your texture

Put an image called:

```text
texture.png
```

in the project directory.

The texture is resized to cover the entire webcam frame.

This means the texture is NOT treated as a photograph of a tie.

Instead:

```text
texture.png
┌──────────────────────────────┐
│                              │
│          TEXTURE             │
│                              │
│                              │
│                              │
└──────────────────────────────┘
```

becomes a virtual background behind the webcam image.

The detected object simply acts as a window through which the texture can be seen.

---

# 7. Run the prank camera

Run:

```bash
python Tie_textureFill.py
```

You should see a webcam window.

The normal webcam remains visible.

Only the selected object is replaced by the texture.

---

# 8. Selecting an object

The dropdown starts with:

```text
tie
```

Click the dropdown to select another YOLO class.

For example:

```text
person
bicycle
car
motorcycle
airplane
bus
train
truck
cat
dog
backpack
handbag
bottle
tie
...
```

The exact list depends on the YOLO model.

Changing the selection happens while the camera is running.

There is no need to restart the program.

---

# 9. Making the prank work with Zoom

The OpenCV window itself is not automatically a webcam.

To make the processed video available to Zoom, route it through a virtual camera.

The easiest setup is:

```text
Webcam
   ↓
Python + YOLO
   ↓
Processed video
   ↓
OBS Virtual Camera
   ↓
Zoom
```

Install OBS Studio from the official OBS website.

OBS provides a Virtual Camera feature that allows its output to be used by webcam applications such as Zoom.

After installing OBS:

1. Open OBS.
2. Add the appropriate video source/output.
3. Start **Virtual Camera**.
4. Open Zoom.
5. Open Zoom's camera selection.
6. Select **OBS Virtual Camera**.

Zoom allows you to switch between available cameras from its video settings/camera menu.

### Important

Test the effect in a private Zoom test meeting first.

Do not discover that your virtual camera is broken five seconds after joining an important meeting.

---

# 10. The "OH NO" transformation mode

The really fun version of this project is the slow transformation.

Instead of immediately doing:

```text
NORMAL → TEXTURE
```

the object gradually transforms:

```text
NORMAL
   ↓
10%
   ↓
25%
   ↓
50%
   ↓
75%
   ↓
100% TEXTURE
```

The recommended transformation time is approximately:

```text
18 seconds
```

---

# 11. Left-to-right transformation

The transformation should be based on the object's position in the webcam frame.

Conceptually:

```text
Object mask

┌─────────────────────────┐
│                         │
│       REAL              │
│       REAL              │
│       REAL              │
│                         │
└─────────────────────────┘
          ↓

┌─────────────────────────┐
│                         │
│       ████ REAL         │
│       ████ REAL         │
│       ████ REAL         │
│                         │
└─────────────────────────┘
          ↓

┌─────────────────────────┐
│                         │
│       ████████ REAL     │
│       ████████ REAL     │
│       ████████ REAL     │
│                         │
└─────────────────────────┘
          ↓

┌─────────────────────────┐
│                         │
│       █████████████     │
│       █████████████     │
│       █████████████     │
│                         │
└─────────────────────────┘

100% TEXTURE
```

Only pixels belonging to the detected object are affected.

The rest of the webcam remains completely normal.

---

# 12. Suggested prank controls

A polished version could use:

| Key      | Action               |
| -------- | -------------------- |
| `T`      | Start transformation |
| `R`      | Reset transformation |
| `SPACE`  | Pause/resume         |
| `Q`      | Quit                 |
| Dropdown | Change target object |

The dropdown remains available throughout the prank.

---

# 13. Making it more convincing

The basic version uses a hard transition:

```text
REAL | TEXTURE
```

A better version uses a soft transition:

```text
REAL → slightly blurred → texture
```

For example:

```text
REAL REAL REAL
REAL REAL REAL
REAL ░░▒▒▓▓▓
REAL ░▒▓▓▓▓▓
REAL ▓▓▓▓▓▓▓
```

This makes the transformation feel more like the object is being "infected" by the texture.

---

# 14. Extra prank ideas

Once the basic system works, several effects become possible.

### The slow corruption

The object slowly turns into the texture over 18 seconds.

### The sudden corruption

Everything is normal until the entire object instantly changes.

### The flicker

The texture briefly appears and disappears:

```text
REAL
TEXTURE
REAL
TEXTURE
REAL
TEXTURE
```

before finally becoming permanent.

### The reverse

The texture slowly disappears and the original object returns.

### The wrong-object prank

Select something unexpected.

For example:

```text
Selected object: PERSON
```

Now the entire person becomes a texture window.

### Multiple objects

Select `person` and every detected person can become a texture window.

### The "infection"

If several objects are present, the effect could spread:

```text
Object 1 → Object 2 → Object 3 → Object 4
```

rather than transforming everything simultaneously.

---

# 15. Recommended final effect

The ultimate version of the prank would behave like this:

```text
00:00
Normal webcam.

00:05
Nothing unusual.

00:06
A tiny portion of the selected object
starts showing the texture.

00:09
The texture has spread roughly 25%.

00:13
The object is approximately half texture.

00:17
Almost completely transformed.

00:18
100% texture.

00:18+
The object stays transformed.

[R]
Reset.

[T]
Do it again.
```

The best part is that the transformation happens gradually enough that someone on a call might initially think:

> "Wait... is my video glitching?"

before realizing what happened.

---

# 16. Troubleshooting

## The webcam is black

First check that the webcam is actually turned on.

Yes, seriously.

Then try changing:

```python
CAMERA_INDEX = 0
```

to:

```python
CAMERA_INDEX = 1
```

or:

```python
CAMERA_INDEX = 2
```

Another application may also already be using the webcam.

---

## The object isn't detected

Make sure the object is one of the classes supported by the model.

Try increasing the object visibility or lowering:

```python
CONFIDENCE = 0.4
```

to:

```python
CONFIDENCE = 0.25
```

Lower confidence can improve detection but may also produce false positives.

---

## The texture isn't appearing

Check:

```text
texture.png
```

exists in the same directory as the script.

The program should not silently continue if the texture cannot be loaded.

---

## Zoom doesn't show the processed camera

Make sure your virtual-camera software is running before opening Zoom.

In Zoom, open the camera selection menu and choose the virtual camera rather than the physical webcam.

Zoom's own troubleshooting guidance recommends checking the selected camera when multiple cameras are available.

---

# 17. Safety / responsible prank use

This project is intended for harmless visual jokes.

Use it with friends or in situations where a temporary visual prank is appropriate.

Do not use manipulated video to impersonate someone, deceive people about important real-world events, or misrepresent yourself in situations where authenticity matters.

---

# 18. Future upgrades

Possible future versions:

* [ ] Slow left-to-right transformation
* [ ] Soft transition edge
* [ ] Transformation sound effects
* [ ] Keyboard controls
* [ ] Object-specific textures
* [ ] Random texture mode
* [ ] Multiple texture slots
* [ ] Texture animation
* [ ] Texture scrolling
* [ ] Object "infection" effect
* [ ] Delayed transformation
* [ ] Reverse transformation
* [ ] OBS/virtual-camera integration
* [ ] One-click prank mode
* [ ] Custom GUI
* [ ] GPU acceleration
* [ ] Per-object transformation state

---

## The final goal

The finished project should feel less like a computer-vision demo and more like a **normal webcam that has a secret superpower**.

The person on the other end should see:

```text
              NORMAL VIDEO

                   ↓

             something weird...

                   ↓

             "Wait... what?"

                   ↓

          THE OBJECT IS CHANGING

                   ↓

              WHAT THE HELL

                   ↓

          OBJECT = TEXTURE
```

And then you casually say:

> "Huh. That's weird."
