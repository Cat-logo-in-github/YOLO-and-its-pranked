from ultralytics import YOLO
import cv2
import numpy as np
import time


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "yolov8n-seg.pt"
TEXTURE_PATH = "_1.png"

CAMERA_INDEX = 0

CONFIDENCE = 0.4
SMOOTHING_ALPHA = 0.7

# How long the transformation takes
TRANSFORM_DURATION = 18.0

# Width of the soft transition edge
FEATHER_WIDTH = 25

WINDOW_NAME = "Virtual Texture"


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading YOLO model...")

model = YOLO(MODEL_PATH)

print("YOLO loaded.")


# ============================================================
# LOAD TEXTURE
# ============================================================

texture = cv2.imread(TEXTURE_PATH)

if texture is None:
    raise ValueError(
        f"Could not load texture image: {TEXTURE_PATH}"
    )

print("Texture loaded.")


# ============================================================
# GET ALL YOLO CLASSES
# ============================================================

class_names = model.names

if isinstance(class_names, dict):

    class_ids = sorted(class_names.keys())

    class_list = [
        class_names[i]
        for i in class_ids
    ]

else:

    class_ids = list(
        range(len(class_names))
    )

    class_list = list(class_names)


# ============================================================
# DEFAULT OBJECT
# ============================================================

selected_object = "tie"

if selected_object in class_list:

    selected_class_id = (
        class_ids[
            class_list.index(selected_object)
        ]
    )

else:

    selected_class_id = class_ids[0]
    selected_object = class_list[0]


print(
    f"Selected object: {selected_object}"
)


# ============================================================
# DROPDOWN STATE
# ============================================================

dropdown_open = False

dropdown_x = 20
dropdown_y = 20

dropdown_width = 280
dropdown_height = 45

option_height = 32

visible_options = 10

dropdown_scroll = 0


# ============================================================
# ANIMATION STATE
# ============================================================

animation_active = False

animation_complete = False

animation_start_time = None


# ============================================================
# RESET ANIMATION
# ============================================================

def reset_animation():

    global animation_active
    global animation_complete
    global animation_start_time

    animation_active = False
    animation_complete = False
    animation_start_time = None


# ============================================================
# START ANIMATION
# ============================================================

def start_animation():

    global animation_active
    global animation_complete
    global animation_start_time

    animation_active = True
    animation_complete = False

    animation_start_time = time.perf_counter()

    print(
        f"Transformation started: "
        f"{selected_object}"
    )


# ============================================================
# MOUSE HANDLER
# ============================================================

def mouse_callback(
    event,
    x,
    y,
    flags,
    param
):

    global dropdown_open
    global selected_object
    global selected_class_id
    global dropdown_scroll

    # --------------------------------------------------------
    # LEFT CLICK
    # --------------------------------------------------------

    if event == cv2.EVENT_LBUTTONDOWN:

        # Main dropdown
        if (
            dropdown_x <= x
            <= dropdown_x + dropdown_width

            and

            dropdown_y <= y
            <= dropdown_y + dropdown_height
        ):

            dropdown_open = not dropdown_open

            return

        # ----------------------------------------------------
        # OPTION CLICK
        # ----------------------------------------------------

        if dropdown_open:

            options_top = (
                dropdown_y
                + dropdown_height
            )

            if (
                dropdown_x <= x
                <= dropdown_x + dropdown_width

                and

                options_top <= y
            ):

                option_index = (
                    (y - options_top)
                    // option_height
                    + dropdown_scroll
                )

                if (
                    0
                    <= option_index
                    < len(class_list)
                ):

                    selected_object = (
                        class_list[
                            option_index
                        ]
                    )

                    selected_class_id = (
                        class_ids[
                            option_index
                        ]
                    )

                    print(
                        f"Selected object: "
                        f"{selected_object}"
                    )

                    # Changing the object
                    # resets the transformation
                    reset_animation()

                    dropdown_open = False

    # --------------------------------------------------------
    # MOUSE WHEEL
    # --------------------------------------------------------

    elif event == cv2.EVENT_MOUSEWHEEL:

        if dropdown_open:

            if flags > 0:

                dropdown_scroll -= 1

            else:

                dropdown_scroll += 1

            max_scroll = max(
                0,
                len(class_list)
                - visible_options
            )

            dropdown_scroll = max(
                0,
                min(
                    dropdown_scroll,
                    max_scroll
                )
            )


# ============================================================
# OPEN WEBCAM
# ============================================================

print("Opening webcam...")

# DirectShow is generally reliable on Windows
cap = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_DSHOW
)

# Fallback
if not cap.isOpened():

    print(
        "DirectShow failed. "
        "Trying default camera backend..."
    )

    cap = cv2.VideoCapture(
        CAMERA_INDEX
    )


if not cap.isOpened():

    raise RuntimeError(
        "Could not open webcam."
    )


# Try to request a decent resolution
cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)


print("Webcam opened.")


# ============================================================
# CREATE WINDOW
# ============================================================

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)

cv2.setMouseCallback(
    WINDOW_NAME,
    mouse_callback
)


# ============================================================
# TEMPORAL MASK
# ============================================================

prev_mask = None


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    # --------------------------------------------------------
    # READ CAMERA
    # --------------------------------------------------------

    ret, frame = cap.read()

    if not ret or frame is None:

        print(
            "Could not read webcam frame."
        )

        continue


    h, w = frame.shape[:2]


    # --------------------------------------------------------
    # FULL SCREEN TEXTURE
    # --------------------------------------------------------

    texture_full = cv2.resize(
        texture,
        (w, h),
        interpolation=cv2.INTER_LINEAR
    )


    # --------------------------------------------------------
    # YOLO SEGMENTATION
    # --------------------------------------------------------

    results = model(
        frame,
        conf=CONFIDENCE,
        verbose=False
    )[0]


    # Start with an empty mask
    mask_full = np.zeros(
        (h, w),
        dtype=np.float32
    )


    # --------------------------------------------------------
    # FIND SELECTED OBJECT
    # --------------------------------------------------------

    if results.masks is not None:

        for i in range(
            len(results.boxes)
        ):

            cls_id = int(
                results.boxes.cls[i]
            )

            # Only selected object
            if cls_id != selected_class_id:

                continue


            mask = (
                results.masks
                .data[i]
                .cpu()
                .numpy()
            )


            # Resize segmentation mask
            mask = cv2.resize(
                mask,
                (w, h),
                interpolation=cv2.INTER_NEAREST
            )


            # Combine multiple instances
            mask_full = np.maximum(
                mask_full,
                mask
            )


    # --------------------------------------------------------
    # TEMPORAL SMOOTHING
    # --------------------------------------------------------

    if prev_mask is None:

        stable_mask = mask_full

    else:

        stable_mask = (
            SMOOTHING_ALPHA
            * prev_mask

            +

            (1.0 - SMOOTHING_ALPHA)
            * mask_full
        )


    prev_mask = stable_mask


    # --------------------------------------------------------
    # BASE OBJECT MASK
    # --------------------------------------------------------

    mask_bool = (
        stable_mask > 0.5
    )


    # --------------------------------------------------------
    # OUTPUT STARTS AS NORMAL WEBCAM
    # --------------------------------------------------------

    output = frame.copy()


    # ========================================================
    # ANIMATION
    # ========================================================

    if animation_active:

        elapsed = (
            time.perf_counter()
            - animation_start_time
        )

        progress = (
            elapsed
            / TRANSFORM_DURATION
        )

        progress = np.clip(
            progress,
            0.0,
            1.0
        )


        # ----------------------------------------------------
        # FIND OBJECT BOUNDING BOX
        # ----------------------------------------------------

        ys, xs = np.where(
            mask_bool
        )


        if len(xs) > 0:

            object_left = xs.min()
            object_right = xs.max()

            object_width = (
                object_right
                - object_left
            )


            # ------------------------------------------------
            # LEFT → RIGHT SWEEP
            # ------------------------------------------------

            sweep_x = (
                object_left
                +
                progress * object_width
            )


            # ------------------------------------------------
            # HARD PART OF SWEEP
            # ------------------------------------------------

            texture_mask = (
                mask_bool
                &
                (
                    np.arange(w)[
                        np.newaxis,
                        :
                    ]
                    <= sweep_x
                )
            )


            # ------------------------------------------------
            # SOFT TRANSITION EDGE
            # ------------------------------------------------

            if FEATHER_WIDTH > 0:

                distance_from_edge = (
                    sweep_x
                    -
                    np.arange(w)[
                        np.newaxis,
                        :
                    ]
                )

                feather = np.clip(
                    (
                        distance_from_edge
                        /
                        FEATHER_WIDTH
                    )
                    + 0.5,
                    0.0,
                    1.0
                )

                # Only object pixels
                feather = (
                    feather
                    *
                    mask_bool.astype(
                        np.float32
                    )
                )

                # Convert to 3-channel
                alpha = feather[
                    :, :, np.newaxis
                ]

                # Blend webcam and texture
                blended = (
                    frame.astype(
                        np.float32
                    )
                    *
                    (1.0 - alpha)

                    +

                    texture_full.astype(
                        np.float32
                    )
                    *
                    alpha
                )

                blended = np.clip(
                    blended,
                    0,
                    255
                ).astype(
                    np.uint8
                )

                # Only modify the object
                output[mask_bool] = (
                    blended[mask_bool]
                )

            else:

                output[
                    texture_mask
                ] = texture_full[
                    texture_mask
                ]


        # ----------------------------------------------------
        # FINISHED
        # ----------------------------------------------------

        if progress >= 1.0:

            animation_active = False
            animation_complete = True

            print(
                "Transformation complete."
            )


    # ========================================================
    # IF ANIMATION IS COMPLETE
    # ========================================================

    elif animation_complete:

        output[mask_bool] = (
            texture_full[mask_bool]
        )


    # ========================================================
    # DRAW DROPDOWN
    # ========================================================

    # Main dropdown
    cv2.rectangle(
        output,
        (
            dropdown_x,
            dropdown_y
        ),
        (
            dropdown_x + dropdown_width,
            dropdown_y + dropdown_height
        ),
        (35, 35, 35),
        -1
    )


    # Border
    cv2.rectangle(
        output,
        (
            dropdown_x,
            dropdown_y
        ),
        (
            dropdown_x + dropdown_width,
            dropdown_y + dropdown_height
        ),
        (255, 255, 255),
        2
    )


    # Selected object text
    cv2.putText(
        output,
        selected_object,
        (
            dropdown_x + 15,
            dropdown_y + 30
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


    # ASCII arrow
    arrow = (
        "^"
        if dropdown_open
        else "v"
    )

    cv2.putText(
        output,
        arrow,
        (
            dropdown_x
            + dropdown_width
            - 30,

            dropdown_y + 30
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


    # ========================================================
    # DROPDOWN OPTIONS
    # ========================================================

    if dropdown_open:

        for display_index in range(
            visible_options
        ):

            actual_index = (
                display_index
                + dropdown_scroll
            )

            if (
                actual_index
                >= len(class_list)
            ):

                break


            option_y = (
                dropdown_y
                + dropdown_height
                +
                display_index
                * option_height
            )


            # Selected option
            if (
                actual_index
                ==
                class_list.index(
                    selected_object
                )
            ):

                color = (
                    70,
                    130,
                    70
                )

            else:

                color = (
                    45,
                    45,
                    45
                )


            # Background
            cv2.rectangle(
                output,
                (
                    dropdown_x,
                    option_y
                ),
                (
                    dropdown_x
                    + dropdown_width,

                    option_y
                    + option_height
                ),
                color,
                -1
            )


            # Border
            cv2.rectangle(
                output,
                (
                    dropdown_x,
                    option_y
                ),
                (
                    dropdown_x
                    + dropdown_width,

                    option_y
                    + option_height
                ),
                (100, 100, 100),
                1
            )


            # Text
            cv2.putText(
                output,
                class_list[
                    actual_index
                ],
                (
                    dropdown_x + 12,
                    option_y + 22
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )


    # ========================================================
    # STATUS TEXT
    # ========================================================

    if animation_active:

        elapsed = (
            time.perf_counter()
            - animation_start_time
        )

        remaining = max(
            0.0,
            TRANSFORM_DURATION
            - elapsed
        )

        status = (
            f"TRANSFORMING  "
            f"{remaining:.1f}s"
        )

        status_color = (
            0,
            220,
            255
        )

    elif animation_complete:

        status = "TRANSFORMED"

        status_color = (
            80,
            255,
            80
        )

    else:

        status = (
            "Press T to transform"
        )

        status_color = (
            255,
            255,
            255
        )


    cv2.putText(
        output,
        status,
        (
            20,
            h - 25
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        status_color,
        2,
        cv2.LINE_AA
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        WINDOW_NAME,
        output
    )


    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    # Start transformation
    if key == ord("t"):

        start_animation()


    # Reset
    elif key == ord("r"):

        reset_animation()

        print(
            "Transformation reset."
        )


    # Quit
    elif key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()