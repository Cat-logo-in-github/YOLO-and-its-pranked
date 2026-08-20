from ultralytics import YOLO
import cv2
import numpy as np
import time
import pyvirtualcam


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "yolov8n-seg.pt"
TEXTURE_PATH = "_1.png"

CAMERA_INDEX = 0

CONFIDENCE = 0.4
SMOOTHING_ALPHA = 0.7

# Transformation duration
TRANSFORM_DURATION = 18.0

# Softness of the transformation edge
FEATHER_WIDTH = 25

# Virtual camera FPS
FPS = 30

WINDOW_NAME = "Prank Controller"


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
# GET YOLO CLASSES
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
# DEFAULT OBJECT = TIE
# ============================================================

selected_object = "tie"

if selected_object in class_list:

    selected_class_id = class_ids[
        class_list.index(selected_object)
    ]

else:

    selected_class_id = class_ids[0]
    selected_object = class_list[0]

print(
    f"Default object: {selected_object}"
)


# ============================================================
# DROPDOWN
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


def reset_animation():

    global animation_active
    global animation_complete
    global animation_start_time

    animation_active = False
    animation_complete = False
    animation_start_time = None


def start_animation():

    global animation_active
    global animation_complete
    global animation_start_time

    animation_active = True
    animation_complete = False

    animation_start_time = time.perf_counter()

    print(
        f"Transformation started: {selected_object}"
    )


# ============================================================
# MOUSE HANDLER
# ============================================================

def mouse_callback(event, x, y, flags, param):

    global dropdown_open
    global selected_object
    global selected_class_id
    global dropdown_scroll

    if event == cv2.EVENT_LBUTTONDOWN:

        # -----------------------------------------------
        # MAIN DROPDOWN
        # -----------------------------------------------

        if (
            dropdown_x <= x <= dropdown_x + dropdown_width
            and
            dropdown_y <= y <= dropdown_y + dropdown_height
        ):

            dropdown_open = not dropdown_open
            return


        # -----------------------------------------------
        # DROPDOWN OPTION
        # -----------------------------------------------

        if dropdown_open:

            options_top = (
                dropdown_y
                + dropdown_height
            )

            if (
                dropdown_x <= x <= dropdown_x + dropdown_width
                and
                y >= options_top
            ):

                option_index = (
                    (y - options_top)
                    // option_height
                    + dropdown_scroll
                )

                if (
                    0 <= option_index < len(class_list)
                ):

                    selected_object = (
                        class_list[option_index]
                    )

                    selected_class_id = (
                        class_ids[option_index]
                    )

                    print(
                        f"Selected object: "
                        f"{selected_object}"
                    )

                    # Switching objects resets
                    # the current transformation.
                    reset_animation()

                    dropdown_open = False


    # -----------------------------------------------
    # MOUSE WHEEL
    # -----------------------------------------------

    elif event == cv2.EVENT_MOUSEWHEEL:

        if dropdown_open:

            if flags > 0:
                dropdown_scroll -= 1
            else:
                dropdown_scroll += 1

            max_scroll = max(
                0,
                len(class_list) - visible_options
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

cap = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_DSHOW
)

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


# Request 1280x720
cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)


# Read one frame so we know the actual resolution
ret, frame = cap.read()

if not ret:

    cap.release()

    raise RuntimeError(
        "Could not read initial webcam frame."
    )


h, w = frame.shape[:2]

print(
    f"Webcam resolution: {w}x{h}"
)


# ============================================================
# START VIRTUAL CAMERA
# ============================================================

print()
print("Starting OBS Virtual Camera...")
print()

try:

    virtual_camera = pyvirtualcam.Camera(
        width=w,
        height=h,
        fps=FPS,
        device="OBS Virtual Camera",
        fmt=pyvirtualcam.PixelFormat.BGR
    )

except Exception as e:

    cap.release()

    print()
    print("ERROR: Could not open OBS Virtual Camera.")
    print()
    print("Make sure:")
    print("1. OBS Studio is installed.")
    print("2. OBS Virtual Camera is available.")
    print("3. You are not already using the virtual camera elsewhere.")
    print()
    print("Detailed error:")
    print(e)

    raise


print(
    f"Virtual camera started: "
    f"{virtual_camera.device}"
)

print()
print("============================================")
print(" PRANK CAMERA READY")
print("============================================")
print()
print("T = Transform")
print("R = Reset")
print("Q = Quit")
print()
print("Select 'OBS Virtual Camera' in Google Meet.")
print()


# ============================================================
# CONTROLLER WINDOW
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

try:

    while True:

        # ----------------------------------------------------
        # READ WEBCAM
        # ----------------------------------------------------

        ret, frame = cap.read()

        if not ret:

            print(
                "Could not read webcam frame."
            )

            continue


        # ----------------------------------------------------
        # FULL-SCREEN TEXTURE
        # ----------------------------------------------------

        texture_full = cv2.resize(
            texture,
            (w, h),
            interpolation=cv2.INTER_LINEAR
        )


        # ----------------------------------------------------
        # YOLO
        # ----------------------------------------------------

        results = model(
            frame,
            conf=CONFIDENCE,
            verbose=False
        )[0]


        mask_full = np.zeros(
            (h, w),
            dtype=np.float32
        )


        # ----------------------------------------------------
        # FIND SELECTED OBJECT
        # ----------------------------------------------------

        if results.masks is not None:

            for i in range(
                len(results.boxes)
            ):

                cls_id = int(
                    results.boxes.cls[i]
                )

                if cls_id != selected_class_id:
                    continue


                mask = (
                    results.masks
                    .data[i]
                    .cpu()
                    .numpy()
                )


                mask = cv2.resize(
                    mask,
                    (w, h),
                    interpolation=cv2.INTER_NEAREST
                )


                mask_full = np.maximum(
                    mask_full,
                    mask
                )


        # ----------------------------------------------------
        # TEMPORAL SMOOTHING
        # ----------------------------------------------------

        if prev_mask is None:

            stable_mask = mask_full

        else:

            stable_mask = (
                SMOOTHING_ALPHA * prev_mask
                +
                (1.0 - SMOOTHING_ALPHA)
                * mask_full
            )


        prev_mask = stable_mask


        mask_bool = (
            stable_mask > 0.5
        )


        # ====================================================
        # CLEAN VIDEO OUTPUT
        #
        # IMPORTANT:
        # This frame NEVER gets the dropdown drawn onto it.
        #
        # This is what gets sent to Meet.
        # ====================================================

        clean_output = frame.copy()


        # ====================================================
        # TRANSFORMATION
        # ====================================================

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


            # ------------------------------------------------
            # OBJECT BOUNDING BOX
            # ------------------------------------------------

            ys, xs = np.where(
                mask_bool
            )


            if len(xs) > 0:

                object_left = xs.min()
                object_right = xs.max()

                object_width = max(
                    1,
                    object_right
                    - object_left
                )


                # --------------------------------------------
                # LEFT → RIGHT POSITION
                # --------------------------------------------

                sweep_x = (
                    object_left
                    +
                    progress
                    * object_width
                )


                # ------------------------------------------------
                # SOFT TRANSITION
                # ------------------------------------------------

                # Create an X coordinate for EVERY pixel
                x_coords = np.arange(
                    w,
                    dtype=np.float32
                )[np.newaxis, :]

                # Broadcast it vertically across the whole frame
                x_coords = np.repeat(
                    x_coords,
                    h,
                    axis=0
                )

                distance_from_edge = (
                    sweep_x - x_coords
                )

                alpha = np.clip(
                    (
                        distance_from_edge
                        /
                        FEATHER_WIDTH
                    )
                    + 0.5,
                    0.0,
                    1.0
                )

                # Only allow the detected object to be affected
                alpha *= mask_bool.astype(
                    np.float32
                )

                # Convert HxW -> HxWx1 for RGB/BGR blending
                alpha = alpha[
                    :, :, np.newaxis
                ]


                # --------------------------------------------
                # BLEND
                # --------------------------------------------

                blended = (
                    frame.astype(
                        np.float32
                    )
                    * (1.0 - alpha)

                    +

                    texture_full.astype(
                        np.float32
                    )
                    * alpha
                )


                blended = np.clip(
                    blended,
                    0,
                    255
                ).astype(
                    np.uint8
                )


                clean_output[
                    mask_bool
                ] = blended[
                    mask_bool
                ]


            # --------------------------------------------
            # COMPLETE
            # --------------------------------------------

            if progress >= 1.0:

                animation_active = False
                animation_complete = True

                print(
                    "Transformation complete."
                )


        # ====================================================
        # FULLY TRANSFORMED STATE
        # ====================================================

        elif animation_complete:

            clean_output[
                mask_bool
            ] = texture_full[
                mask_bool
            ]


        # ====================================================
        # SEND CLEAN VIDEO TO VIRTUAL CAMERA
        # ====================================================

        virtual_camera.send(
            clean_output
        )

        virtual_camera.sleep_until_next_frame()


        # ====================================================
        # LOCAL CONTROLLER
        #
        # This is a COPY.
        # The dropdown is ONLY drawn here.
        # ====================================================

        controller = clean_output.copy()


        # ====================================================
        # DROPDOWN
        # ====================================================

        cv2.rectangle(
            controller,
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


        cv2.rectangle(
            controller,
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


        cv2.putText(
            controller,
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


        arrow = "^" if dropdown_open else "v"

        cv2.putText(
            controller,
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


        # ====================================================
        # DROPDOWN OPTIONS
        # ====================================================

        if dropdown_open:

            selected_index = (
                class_ids.index(
                    selected_class_id
                )
            )

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


                if (
                    actual_index
                    == selected_index
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


                cv2.rectangle(
                    controller,
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


                cv2.rectangle(
                    controller,
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


                cv2.putText(
                    controller,
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


        # ====================================================
        # LOCAL STATUS
        # ====================================================

        if animation_active:

            elapsed = (
                time.perf_counter()
                - animation_start_time
            )

            progress = np.clip(
                elapsed
                / TRANSFORM_DURATION,
                0.0,
                1.0
            )

            status = (
                f"TRANSFORMING "
                f"{progress * 100:.0f}%"
            )

        elif animation_complete:

            status = "TRANSFORMED"

        else:

            status = "T = transform   R = reset"


        cv2.putText(
            controller,
            status,
            (
                20,
                h - 25
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


        # ====================================================
        # SHOW LOCAL CONTROLLER
        # ====================================================

        cv2.imshow(
            WINDOW_NAME,
            controller
        )


        # ====================================================
        # KEYBOARD
        # ====================================================

        key = (
            cv2.waitKey(1)
            & 0xFF
        )


        if key == ord("t"):

            start_animation()


        elif key == ord("r"):

            reset_animation()

            print(
                "Transformation reset."
            )


        elif key == ord("q"):

            break


finally:

    # ========================================================
    # CLEANUP
    # ========================================================

    print()
    print("Shutting down...")

    virtual_camera.close()

    cap.release()

    cv2.destroyAllWindows()

    print("Done.")