from ultralytics import YOLO
import cv2
import numpy as np
import pyvirtualcam
import random
import time


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "yolov8n-seg.pt"
BACKGROUND_PATH = "bg.png"
CRACK_PATH = "crack.png"

CAMERA_INDEX = 0

CONFIDENCE = 0.25

FPS = 30

WINDOW_NAME = "Haunted Camera"


# ============================================================
# EFFECT CONFIGURATION
# ============================================================

# Object movement
TELEPORT_DURATION = 2.0

TELEPORT_DISTANCE_MIN = 45
TELEPORT_DISTANCE_MAX = 90


# Object disappearance
DISAPPEAR_DURATION = 1.5


# Lighting
LIGHT_DURATION = 0.45


# Visual glitch
GLITCH_DURATION = 0.20


# Camera crack
CRACK_BUILD_DURATION = 10.0

# Crack opacity.
# Lower = more subtle.
CRACK_OPACITY = 0.65


# ============================================================
# OBJECT FILTERING
# ============================================================

# Ignore tiny detections.
MIN_OBJECT_WIDTH = 35
MIN_OBJECT_HEIGHT = 35
MIN_OBJECT_AREA = 2500


# ============================================================
# EFFECT LIST
# ============================================================

EFFECTS = [
    "teleport",
    "disappear",
    "lighting",
    "glitch",
    "crack",
]


# ============================================================
# LOAD YOLO
# ============================================================

print()
print("Loading YOLO model...")

model = YOLO(
    MODEL_PATH
)

print("YOLO loaded.")


# ============================================================
# LOAD BACKGROUND
# ============================================================

background = cv2.imread(
    BACKGROUND_PATH
)

if background is None:

    raise ValueError(
        f"Could not load {BACKGROUND_PATH}"
    )

print("Background loaded.")


# ============================================================
# LOAD CRACK ASSET
# ============================================================

crack_asset = cv2.imread(
    CRACK_PATH,
    cv2.IMREAD_UNCHANGED
)

if crack_asset is None:

    raise ValueError(
        f"Could not load {CRACK_PATH}"
    )

if len(crack_asset.shape) != 3:

    raise ValueError(
        "crack.png must be a PNG image."
    )

if crack_asset.shape[2] != 4:

    raise ValueError(
        "crack.png must contain transparency "
        "(RGBA / 4 channels)."
    )

print("Crack overlay loaded.")


# ============================================================
# GLOBAL EFFECT STATE
# ============================================================

active_effect = None

effect_start_time = 0.0

selected_object = None
selected_mask = None
selected_bbox = None


# ============================================================
# TELEPORT STATE
# ============================================================

teleport_direction = 1
teleport_distance = 60


# ============================================================
# CRACK STATE
# ============================================================

crack_active = False
crack_start_time = None


# ============================================================
# RESET
# ============================================================

def reset_effect():

    global active_effect
    global effect_start_time

    global selected_object
    global selected_mask
    global selected_bbox

    global teleport_direction
    global teleport_distance

    global crack_active
    global crack_start_time

    active_effect = None
    effect_start_time = 0.0

    selected_object = None
    selected_mask = None
    selected_bbox = None

    teleport_direction = 1
    teleport_distance = 60

    crack_active = False
    crack_start_time = None

    print()
    print("Everything reset.")
    print()


# ============================================================
# DETECT OBJECTS
# ============================================================

def detect_objects(frame):

    results = model(
        frame,
        conf=CONFIDENCE,
        verbose=False
    )[0]

    objects = []

    if (
        results.boxes is None
        or results.masks is None
    ):
        return objects

    h, w = frame.shape[:2]

    for i in range(
        len(results.boxes)
    ):

        cls_id = int(
            results.boxes.cls[i]
        )

        class_name = model.names[
            cls_id
        ]

        confidence = float(
            results.boxes.conf[i]
        )

        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        box = (
            results.boxes.xyxy[i]
            .cpu()
            .numpy()
        )

        x1, y1, x2, y2 = (
            box.astype(int)
        )

        x1 = max(
            0,
            min(x1, w - 1)
        )

        x2 = max(
            0,
            min(x2, w)
        )

        y1 = max(
            0,
            min(y1, h - 1)
        )

        y2 = max(
            0,
            min(y2, h)
        )

        box_w = x2 - x1
        box_h = y2 - y1

        if box_w < MIN_OBJECT_WIDTH:
            continue

        if box_h < MIN_OBJECT_HEIGHT:
            continue

        if (
            box_w * box_h
            < MIN_OBJECT_AREA
        ):
            continue

        # ----------------------------------------------------
        # Segmentation mask
        # ----------------------------------------------------

        mask = (
            results.masks.data[i]
            .cpu()
            .numpy()
        )

        mask = cv2.resize(
            mask,
            (w, h),
            interpolation=cv2.INTER_NEAREST
        )

        mask = (
            mask > 0.5
        ).astype(np.uint8)

        # ----------------------------------------------------
        # Ignore extremely small masks
        # ----------------------------------------------------

        if np.count_nonzero(mask) < 500:
            continue

        objects.append({
            "class_id": cls_id,
            "name": class_name,
            "confidence": confidence,
            "bbox": (
                x1,
                y1,
                x2,
                y2
            ),
            "mask": mask
        })

    return objects


# ============================================================
# CHOOSE RANDOM OBJECT
# ============================================================

def choose_random_object(objects):

    if not objects:
        return None

    return random.choice(
        objects
    )


# ============================================================
# CHOOSE RANDOM EFFECT
# ============================================================

def choose_random_effect():

    return random.choice(
        EFFECTS
    )


# ============================================================
# START OBJECT EFFECT
# ============================================================

def start_object_effect(
    effect,
    obj
):

    global active_effect
    global effect_start_time

    global selected_object
    global selected_mask
    global selected_bbox

    global teleport_direction
    global teleport_distance

    if obj is None:

        print(
            "No suitable object detected."
        )

        return False

    active_effect = effect

    effect_start_time = time.time()

    selected_object = obj[
        "name"
    ]

    selected_mask = obj[
        "mask"
    ].copy()

    selected_bbox = obj[
        "bbox"
    ]

    # --------------------------------------------------------
    # Pick teleport direction ONCE.
    # --------------------------------------------------------

    if effect == "teleport":

        teleport_direction = random.choice(
            [-1, 1]
        )

        teleport_distance = random.randint(
            TELEPORT_DISTANCE_MIN,
            TELEPORT_DISTANCE_MAX
        )

        direction_name = (
            "LEFT"
            if teleport_direction < 0
            else "RIGHT"
        )

        print()
        print(
            "HAUNTED EVENT"
        )

        print(
            f"Object    : {selected_object}"
        )

        print(
            f"Effect    : {effect}"
        )

        print(
            f"Direction : {direction_name}"
        )

        print(
            f"Distance  : {teleport_distance}px"
        )

        print()

    else:

        print()
        print(
            "HAUNTED EVENT"
        )

        print(
            f"Object : {selected_object}"
        )

        print(
            f"Effect : {effect}"
        )

        print()

    return True


# ============================================================
# START FULLSCREEN EFFECT
# ============================================================

def start_fullscreen_effect(
    effect
):

    global active_effect
    global effect_start_time

    active_effect = effect

    effect_start_time = time.time()

    print()
    print(
        "HAUNTED EVENT"
    )

    print(
        f"Effect : {effect}"
    )

    print()


# ============================================================
# START CRACK
# ============================================================

def start_crack():

    global active_effect
    global effect_start_time

    global crack_active
    global crack_start_time

    if crack_active:

        return

    active_effect = "crack"

    effect_start_time = time.time()

    crack_active = True

    crack_start_time = time.time()

    print()
    print(
        "============================================"
    )
    print(
        " CAMERA CRACK STARTED"
    )
    print(
        " Developing over 10 seconds..."
    )
    print(
        " It will remain until R is pressed."
    )
    print(
        "============================================"
    )
    print()


# ============================================================
# PREPARE CRACK ASSET
# ============================================================

def prepare_crack_asset(
    asset,
    width,
    height
):

    original_h, original_w = (
        asset.shape[:2]
    )

    # --------------------------------------------------------
    # Scale while preserving aspect ratio.
    #
    # We want the crack to cover the camera,
    # but not distort it.
    # --------------------------------------------------------

    scale = max(
        width / original_w,
        height / original_h
    )

    new_w = int(
        original_w * scale
    )

    new_h = int(
        original_h * scale
    )

    resized = cv2.resize(
        asset,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    # --------------------------------------------------------
    # Center crop to camera dimensions.
    # --------------------------------------------------------

    x_offset = (
        new_w - width
    ) // 2

    y_offset = (
        new_h - height
    ) // 2

    resized = resized[
        y_offset:y_offset + height,
        x_offset:x_offset + width
    ]

    return resized


# ============================================================
# APPLY CRACK
# ============================================================

def apply_crack_overlay(
    frame,
    prepared_crack,
    progress
):

    h, w = frame.shape[:2]

    # --------------------------------------------------------
    # RGBA crack
    # --------------------------------------------------------

    crack_rgb = prepared_crack[
        :, :, :3
    ]

    crack_alpha = (
        prepared_crack[
            :, :, 3
        ].astype(
            np.float32
        )
        / 255.0
    )

    # --------------------------------------------------------
    # Reveal from center outward.
    #
    # This makes the crack appear to develop rather
    # than simply fading onto the entire screen.
    # --------------------------------------------------------

    yy, xx = np.mgrid[
        0:h,
        0:w
    ]

    center_x = w * 0.50
    center_y = h * 0.50

    normalized_distance = np.sqrt(
        (
            (xx - center_x)
            / w
        ) ** 2
        +
        (
            (yy - center_y)
            / h
        ) ** 2
    )

    # Maximum useful radius.
    max_radius = 0.55

    radius = (
        progress
        * max_radius
    )

    # Soft transition.
    edge_width = 0.025

    reveal = np.clip(
        (
            radius
            - normalized_distance
        )
        / edge_width
        + 0.5,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # Apply reveal.
    # --------------------------------------------------------

    alpha = (
        crack_alpha
        * reveal
        * CRACK_OPACITY
    )

    # --------------------------------------------------------
    # Slightly strengthen the crack after it is developed.
    # --------------------------------------------------------

    if progress >= 1.0:

        alpha = (
            crack_alpha
            * CRACK_OPACITY
        )

    alpha = alpha[
        ..., None
    ]

    # --------------------------------------------------------
    # Composite.
    # --------------------------------------------------------

    output = (
        frame.astype(
            np.float32
        )
        * (1.0 - alpha)
        +
        crack_rgb.astype(
            np.float32
        )
        * alpha
    )

    return np.clip(
        output,
        0,
        255
    ).astype(
        np.uint8
    )


# ============================================================
# TELEPORT EFFECT
# ============================================================

def apply_teleport(
    frame,
    clean_background,
    mask,
    bbox,
    progress
):

    output = frame.copy()

    if mask is None or bbox is None:

        return output

    x1, y1, x2, y2 = bbox

    # --------------------------------------------------------
    # Smooth movement:
    #
    # 0%   = original position
    # 50%  = maximum displacement
    # 100% = original position
    #
    # sin(pi * progress)
    # --------------------------------------------------------

    movement = np.sin(
        np.pi * progress
    )

    dx = int(
        teleport_direction
        * teleport_distance
        * movement
    )

    # --------------------------------------------------------
    # Remove object from original position.
    # --------------------------------------------------------

    object_pixels = (
        mask > 0
    )

    output[
        object_pixels
    ] = clean_background[
        object_pixels
    ]

    # --------------------------------------------------------
    # Extract object.
    # --------------------------------------------------------

    object_crop = frame[
        y1:y2,
        x1:x2
    ].copy()

    object_mask = mask[
        y1:y2,
        x1:x2
    ].copy()

    if object_crop.size == 0:

        return output

    # --------------------------------------------------------
    # New position.
    # --------------------------------------------------------

    new_x1 = x1 + dx
    new_x2 = x2 + dx

    new_y1 = y1
    new_y2 = y2

    h, w = frame.shape[:2]

    # --------------------------------------------------------
    # Keep object inside frame.
    # --------------------------------------------------------

    if new_x1 < 0:

        correction = -new_x1

        new_x1 += correction
        new_x2 += correction

    if new_x2 > w:

        correction = new_x2 - w

        new_x1 -= correction
        new_x2 -= correction

    new_x1 = int(
        new_x1
    )

    new_x2 = int(
        new_x2
    )

    # --------------------------------------------------------
    # Check dimensions.
    # --------------------------------------------------------

    if (
        new_x2 <= new_x1
        or new_y2 <= new_y1
    ):

        return output

    roi = output[
        new_y1:new_y2,
        new_x1:new_x2
    ]

    if (
        roi.shape[:2]
        != object_crop.shape[:2]
    ):

        return output

    # --------------------------------------------------------
    # Composite.
    # --------------------------------------------------------

    alpha = (
        object_mask
        .astype(
            np.float32
        )
        [..., None]
    )

    blended = (
        roi.astype(
            np.float32
        )
        * (1.0 - alpha)
        +
        object_crop.astype(
            np.float32
        )
        * alpha
    )

    output[
        new_y1:new_y2,
        new_x1:new_x2
    ] = blended.astype(
        np.uint8
    )

    return output


# ============================================================
# DISAPPEAR EFFECT
# ============================================================

def apply_disappear(
    frame,
    clean_background,
    mask,
    progress
):

    if mask is None:

        return frame.copy()

    # --------------------------------------------------------
    # Fade object out, hold, then return.
    # --------------------------------------------------------

    if progress < 0.20:

        strength = (
            progress
            / 0.20
        )

    elif progress < 0.70:

        strength = 1.0

    else:

        strength = (
            1.0
            -
            (
                progress
                - 0.70
            )
            / 0.30
        )

    strength = max(
        0.0,
        min(
            1.0,
            strength
        )
    )

    alpha = (
        mask.astype(
            np.float32
        )
        * strength
    )

    alpha = alpha[
        ..., None
    ]

    output = (
        frame.astype(
            np.float32
        )
        * (1.0 - alpha)
        +
        clean_background.astype(
            np.float32
        )
        * alpha
    )

    return np.clip(
        output,
        0,
        255
    ).astype(
        np.uint8
    )


# ============================================================
# LIGHTING EFFECT
# ============================================================

def apply_lighting(
    frame,
    progress
):

    output = frame.astype(
        np.float32
    )

    if progress < 0.20:

        factor = 0.70

    elif progress < 0.40:

        factor = 1.25

    elif progress < 0.65:

        factor = 0.55

    else:

        factor = 1.0

    output *= factor

    return np.clip(
        output,
        0,
        255
    ).astype(
        np.uint8
    )


# ============================================================
# GLITCH EFFECT
# ============================================================

def apply_glitch(
    frame
):

    output = frame.copy()

    h, w = frame.shape[:2]

    # --------------------------------------------------------
    # Chromatic displacement.
    # --------------------------------------------------------

    shift = random.randint(
        5,
        20
    )

    red = output[
        :, :, 2
    ].copy()

    blue = output[
        :, :, 0
    ].copy()

    output[
        :, :, 2
    ] = np.roll(
        red,
        shift,
        axis=1
    )

    output[
        :, :, 0
    ] = np.roll(
        blue,
        -shift,
        axis=1
    )

    # --------------------------------------------------------
    # Horizontal tearing.
    # --------------------------------------------------------

    for _ in range(4):

        y = random.randint(
            0,
            h - 1
        )

        thickness = random.randint(
            2,
            10
        )

        offset = random.randint(
            -30,
            30
        )

        y2 = min(
            h,
            y + thickness
        )

        output[
            y:y2
        ] = np.roll(
            output[
                y:y2
            ],
            offset,
            axis=1
        )

    return output


# ============================================================
# MAIN
# ============================================================

print()
print("Opening webcam...")

cap = cv2.VideoCapture(
    CAMERA_INDEX
)

if not cap.isOpened():

    raise RuntimeError(
        "Could not open webcam."
    )


# ============================================================
# FIRST FRAME
# ============================================================

ret, first_frame = cap.read()

if not ret:

    cap.release()

    raise RuntimeError(
        "Could not read webcam."
    )

h, w = first_frame.shape[:2]

print(
    f"Webcam resolution: {w}x{h}"
)


# ============================================================
# PREPARE BACKGROUND
# ============================================================

clean_background = cv2.resize(
    background,
    (w, h),
    interpolation=cv2.INTER_LINEAR
)


# ============================================================
# PREPARE CRACK
# ============================================================

prepared_crack = prepare_crack_asset(
    crack_asset,
    w,
    h
)


# ============================================================
# VIRTUAL CAMERA
# ============================================================

print()
print(
    "Starting OBS Virtual Camera..."
)

with pyvirtualcam.Camera(
    width=w,
    height=h,
    fps=FPS,
    device="OBS Virtual Camera"
) as virtual_camera:

    print()
    print(
        "Virtual camera started:",
        virtual_camera.device
    )

    print()
    print(
        "============================================"
    )
    print(
        "           HAUNTED CAMERA READY"
    )
    print(
        "============================================"
    )
    print()
    print(
        "T = Random haunted event"
    )
    print(
        "1 = Subtle object movement"
    )
    print(
        "2 = Object disappearance"
    )
    print(
        "3 = Lighting glitch"
    )
    print(
        "4 = Visual glitch"
    )
    print(
        "5 = Camera crack"
    )
    print(
        "R = Reset"
    )
    print(
        "B = Reload background"
    )
    print(
        "Q = Quit"
    )
    print()
    print(
        "Select 'OBS Virtual Camera'"
    )
    print(
        "in Zoom / Google Meet."
    )
    print()

    # ========================================================
    # MAIN LOOP
    # ========================================================

    while True:

        ret, frame = cap.read()

        if not ret:

            break

        # ----------------------------------------------------
        # Detect objects.
        # ----------------------------------------------------

        objects = detect_objects(
            frame
        )

        # ----------------------------------------------------
        # Normal camera image.
        # ----------------------------------------------------

        output = frame.copy()

        # ====================================================
        # ACTIVE EFFECT
        # ====================================================

        if active_effect == "teleport":

            elapsed = (
                time.time()
                - effect_start_time
            )

            progress = min(
                elapsed
                / TELEPORT_DURATION,
                1.0
            )

            output = apply_teleport(
                frame,
                clean_background,
                selected_mask,
                selected_bbox,
                progress
            )

            if elapsed >= TELEPORT_DURATION:

                reset_effect()


        elif active_effect == "disappear":

            elapsed = (
                time.time()
                - effect_start_time
            )

            progress = min(
                elapsed
                / DISAPPEAR_DURATION,
                1.0
            )

            output = apply_disappear(
                frame,
                clean_background,
                selected_mask,
                progress
            )

            if elapsed >= DISAPPEAR_DURATION:

                reset_effect()


        elif active_effect == "lighting":

            elapsed = (
                time.time()
                - effect_start_time
            )

            progress = min(
                elapsed
                / LIGHT_DURATION,
                1.0
            )

            output = apply_lighting(
                frame,
                progress
            )

            if elapsed >= LIGHT_DURATION:

                reset_effect()


        elif active_effect == "glitch":

            elapsed = (
                time.time()
                - effect_start_time
            )

            output = apply_glitch(
                frame
            )

            if elapsed >= GLITCH_DURATION:

                reset_effect()


        elif active_effect == "crack":

            # ------------------------------------------------
            # The crack persists.
            # ------------------------------------------------

            elapsed = (
                time.time()
                - crack_start_time
            )

            progress = min(
                elapsed
                / CRACK_BUILD_DURATION,
                1.0
            )

            output = apply_crack_overlay(
                frame,
                prepared_crack,
                progress
            )

            # NO automatic reset.
            #
            # R is required to remove it.


        # ====================================================
        # SEND TO OBS VIRTUAL CAMERA
        # ====================================================

        rgb = cv2.cvtColor(
            output,
            cv2.COLOR_BGR2RGB
        )

        virtual_camera.send(
            rgb
        )

        virtual_camera.sleep_until_next_frame()


        # ====================================================
        # LOCAL PREVIEW
        # ====================================================

        cv2.imshow(
            WINDOW_NAME,
            output
        )


        # ====================================================
        # KEYBOARD
        # ====================================================

        key = (
            cv2.waitKey(1)
            & 0xFF
        )


        # ====================================================
        # T — RANDOM EFFECT
        # ====================================================

        if key == ord("t"):

            # Don't interrupt an existing event.
            if active_effect is not None:

                continue

            effect = choose_random_effect()

            if effect in [
                "teleport",
                "disappear"
            ]:

                obj = choose_random_object(
                    objects
                )

                if obj is not None:

                    start_object_effect(
                        effect,
                        obj
                    )

                else:

                    print(
                        "T chose an object effect, "
                        "but no suitable object was detected."
                    )


            elif effect == "lighting":

                start_fullscreen_effect(
                    "lighting"
                )


            elif effect == "glitch":

                start_fullscreen_effect(
                    "glitch"
                )


            elif effect == "crack":

                start_crack()


        # ====================================================
        # 1 — OBJECT MOVEMENT
        # ====================================================

        elif key == ord("1"):

            if active_effect is None:

                obj = choose_random_object(
                    objects
                )

                if obj is not None:

                    start_object_effect(
                        "teleport",
                        obj
                    )

                else:

                    print(
                        "No suitable object detected."
                    )


        # ====================================================
        # 2 — DISAPPEAR
        # ====================================================

        elif key == ord("2"):

            if active_effect is None:

                obj = choose_random_object(
                    objects
                )

                if obj is not None:

                    start_object_effect(
                        "disappear",
                        obj
                    )

                else:

                    print(
                        "No suitable object detected."
                    )


        # ====================================================
        # 3 — LIGHTING
        # ====================================================

        elif key == ord("3"):

            if active_effect is None:

                start_fullscreen_effect(
                    "lighting"
                )


        # ====================================================
        # 4 — GLITCH
        # ====================================================

        elif key == ord("4"):

            if active_effect is None:

                start_fullscreen_effect(
                    "glitch"
                )


        # ====================================================
        # 5 — CRACK
        # ====================================================

        elif key == ord("5"):

            if not crack_active:

                start_crack()


        # ====================================================
        # R — RESET
        # ====================================================

        elif key == ord("r"):

            reset_effect()


        # ====================================================
        # B — RELOAD BACKGROUND
        # ====================================================

        elif key == ord("b"):

            new_background = cv2.imread(
                BACKGROUND_PATH
            )

            if new_background is not None:

                clean_background = cv2.resize(
                    new_background,
                    (w, h),
                    interpolation=cv2.INTER_LINEAR
                )

                print(
                    "Background reloaded."
                )

            else:

                print(
                    "Could not reload bg.png."
                )


        # ====================================================
        # Q — QUIT
        # ====================================================

        elif key == ord("q"):

            break


# ============================================================
# CLEANUP
# ============================================================

print()
print(
    "Shutting down..."
)

cap.release()

cv2.destroyAllWindows()

print(
    "Done."
)