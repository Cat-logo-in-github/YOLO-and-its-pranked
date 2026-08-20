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

CONFIDENCE = 0.4
SMOOTHING_ALPHA = 0.7

CAMERA_INDEX = 0
FPS = 30

WINDOW_NAME = "Subtle Prank Controller"

# How long the inversion takes to appear
EFFECT_DURATION = 15.0

# Maximum opacity of the inversion
#
# 1.0 = completely inverted
# 0.5 = halfway between normal and inverted
#
MAX_STRENGTH = 0.85


# Size of the inverted patch relative to
# the detected object's dimensions.
STAIN_MIN_RATIO = 0.12
STAIN_MAX_RATIO = 0.25


# ============================================================
# DROPDOWN
# ============================================================

DROPDOWN_X = 20
DROPDOWN_Y = 20

DROPDOWN_WIDTH = 280
DROPDOWN_HEIGHT = 45

OPTION_HEIGHT = 32
VISIBLE_OPTIONS = 10

dropdown_open = False
dropdown_scroll = 0


# ============================================================
# LOAD YOLO
# ============================================================

print("Loading YOLO model...")

model = YOLO(MODEL_PATH)

print("YOLO loaded.")


# ============================================================
# GET YOLO CLASSES
# ============================================================

class_names = model.names

if isinstance(class_names, dict):

    class_list = [
        class_names[i]
        for i in sorted(class_names.keys())
    ]

else:

    class_list = list(class_names)


# ============================================================
# DEFAULT TARGET
# ============================================================

selected_object = "tie"

if selected_object not in class_list:

    selected_object = class_list[0]


selected_class_id = class_list.index(
    selected_object
)


print(
    f"Default object: {selected_object}"
)


# ============================================================
# EFFECT STATE
# ============================================================

effect_active = False

effect_start_time = None

stain_seed = None

stain_center = None

stain_radius = None

stain_angle = None


# ============================================================
# TEMPORAL MASK
# ============================================================

prev_mask = None


# ============================================================
# CREATE STAIN PARAMETERS
# ============================================================

def create_stain_parameters(
    binary_mask
):

    global stain_seed
    global stain_center
    global stain_radius
    global stain_angle


    ys, xs = np.where(
        binary_mask > 0
    )


    if len(xs) < 50:

        stain_center = None
        stain_radius = None

        return


    x_min = xs.min()
    x_max = xs.max()

    y_min = ys.min()
    y_max = ys.max()


    target_w = x_max - x_min
    target_h = y_max - y_min


    if (
        target_w < 20
        or
        target_h < 20
    ):

        return


    # --------------------------------------------------------
    # Find a point safely inside the object
    # --------------------------------------------------------

    for _ in range(100):

        x = random.randint(
            x_min,
            x_max
        )

        y = random.randint(
            y_min,
            y_max
        )


        if binary_mask[y, x] != 1:

            continue


        # Check surrounding region.

        local_size = max(
            5,
            min(
                target_w,
                target_h
            ) // 8
        )


        xa = max(
            0,
            x - local_size
        )

        xb = min(
            binary_mask.shape[1],
            x + local_size
        )

        ya = max(
            0,
            y - local_size
        )

        yb = min(
            binary_mask.shape[0],
            y + local_size
        )


        local_region = binary_mask[
            ya:yb,
            xa:xb
        ]


        if np.mean(local_region) > 0.75:

            stain_center = (
                x,
                y
            )

            break


    # --------------------------------------------------------
    # Fallback to center of object
    # --------------------------------------------------------

    if stain_center is None:

        stain_center = (
            int(
                (x_min + x_max) / 2
            ),
            int(
                (y_min + y_max) / 2
            )
        )


    # --------------------------------------------------------
    # Random stain size
    # --------------------------------------------------------

    base_size = min(
        target_w,
        target_h
    )


    ratio = random.uniform(
        STAIN_MIN_RATIO,
        STAIN_MAX_RATIO
    )


    stain_radius = max(
        4,
        int(
            base_size * ratio
        )
    )


    # --------------------------------------------------------
    # Random orientation
    # --------------------------------------------------------

    stain_angle = random.uniform(
        0,
        np.pi
    )


    # --------------------------------------------------------
    # Random seed
    # --------------------------------------------------------

    stain_seed = random.randint(
        0,
        999999
    )


# ============================================================
# GENERATE IRREGULAR STAIN
# ============================================================

def generate_stain_mask(
    shape,
    center,
    radius,
    angle,
    seed
):

    h, w = shape

    rng = np.random.default_rng(
        seed
    )


    cx, cy = center


    # ========================================================
    # LOW RESOLUTION RANDOM FIELD
    # ========================================================

    noise_h = max(
        3,
        h // 45
    )

    noise_w = max(
        3,
        w // 45
    )


    noise = rng.random(
        (
            noise_h,
            noise_w
        )
    ).astype(
        np.float32
    )


    noise = cv2.resize(
        noise,
        (w, h),
        interpolation=cv2.INTER_CUBIC
    )


    noise = cv2.GaussianBlur(
        noise,
        (31, 31),
        0
    )


    noise = cv2.normalize(
        noise,
        None,
        0.72,
        1.28,
        cv2.NORM_MINMAX
    )


    # ========================================================
    # ROTATED ELLIPSE
    # ========================================================

    yy, xx = np.mgrid[
        0:h,
        0:w
    ]


    dx = xx - cx
    dy = yy - cy


    cos_a = np.cos(
        angle
    )

    sin_a = np.sin(
        angle
    )


    rotated_x = (
        dx * cos_a
        +
        dy * sin_a
    )

    rotated_y = (
        -dx * sin_a
        +
        dy * cos_a
    )


    rx = radius * 1.35
    ry = radius * 0.85


    distance = np.sqrt(
        (
            rotated_x / rx
        ) ** 2
        +
        (
            rotated_y / ry
        ) ** 2
    )


    # ========================================================
    # IRREGULAR EDGE
    # ========================================================

    effective_radius = (
        noise
    )


    raw_mask = (
        distance
        <
        effective_radius
    ).astype(
        np.float32
    )


    # ========================================================
    # SOFT EDGE
    # ========================================================

    blur_size = max(
        7,
        radius // 3
    )


    if blur_size % 2 == 0:

        blur_size += 1


    soft_mask = cv2.GaussianBlur(
        raw_mask,
        (
            blur_size,
            blur_size
        ),
        0
    )


    return np.clip(
        soft_mask,
        0.0,
        1.0
    )


# ============================================================
# MOUSE CALLBACK
# ============================================================

def mouse_callback(
    event,
    x,
    y,
    flags,
    param
):

    global dropdown_open
    global dropdown_scroll

    global selected_object
    global selected_class_id

    global prev_mask


    # ========================================================
    # LEFT CLICK
    # ========================================================

    if event == cv2.EVENT_LBUTTONDOWN:


        # ----------------------------------------------------
        # Main dropdown
        # ----------------------------------------------------

        if (
            DROPDOWN_X <= x <=
            DROPDOWN_X + DROPDOWN_WIDTH
            and
            DROPDOWN_Y <= y <=
            DROPDOWN_Y + DROPDOWN_HEIGHT
        ):

            dropdown_open = (
                not dropdown_open
            )

            return


        # ----------------------------------------------------
        # Select option
        # ----------------------------------------------------

        if dropdown_open:

            options_top = (
                DROPDOWN_Y
                +
                DROPDOWN_HEIGHT
            )


            option_index = (
                (
                    y
                    -
                    options_top
                )
                //
                OPTION_HEIGHT
                +
                dropdown_scroll
            )


            if (
                0 <= option_index
                <
                len(class_list)
                and
                DROPDOWN_X <= x <=
                DROPDOWN_X + DROPDOWN_WIDTH
            ):

                selected_object = (
                    class_list[
                        option_index
                    ]
                )


                selected_class_id = (
                    option_index
                )


                # Changing the object resets
                # the tracking state.

                prev_mask = None


                print()
                print(
                    f"Selected object: "
                    f"{selected_object}"
                )


                dropdown_open = False


                return


    # ========================================================
    # MOUSE WHEEL
    # ========================================================

    elif event == cv2.EVENT_MOUSEWHEEL:

        if dropdown_open:

            if flags > 0:

                dropdown_scroll -= 1

            else:

                dropdown_scroll += 1


            max_scroll = max(
                0,
                len(class_list)
                -
                VISIBLE_OPTIONS
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

print()
print("Opening webcam...")

cap = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_DSHOW
)


if not cap.isOpened():

    cap = cv2.VideoCapture(
        CAMERA_INDEX
    )


if not cap.isOpened():

    raise RuntimeError(
        "Could not open webcam."
    )


cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)


ret, frame = cap.read()


if not ret:

    cap.release()

    raise RuntimeError(
        "Could not read webcam."
    )


h, w = frame.shape[:2]


print(
    f"Webcam resolution: {w}x{h}"
)


# ============================================================
# OBS VIRTUAL CAMERA
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
    print(
        "Could not start OBS Virtual Camera."
    )

    print(e)

    raise


print(
    f"Virtual camera started: "
    f"{virtual_camera.device}"
)


# ============================================================
# MAIN LOOP
# ============================================================

try:

    while True:

        # ====================================================
        # READ CAMERA
        # ====================================================

        ret, frame = cap.read()


        if not ret:

            continue


        output_frame = frame.copy()


        # ====================================================
        # YOLO SEGMENTATION
        # ====================================================

        results = model(
            frame,
            conf=CONFIDENCE,
            verbose=False
        )[0]


        mask_full = np.zeros(
            (h, w),
            dtype=np.float32
        )


        if (
            results.boxes is not None
            and
            results.masks is not None
        ):

            for i in range(
                len(results.boxes)
            ):

                class_id = int(
                    results.boxes.cls[i]
                )


                if (
                    class_id
                    !=
                    selected_class_id
                ):

                    continue


                mask = (
                    results
                    .masks
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


        # ====================================================
        # TEMPORAL SMOOTHING
        # ====================================================

        if prev_mask is None:

            stable_mask = mask_full

        else:

            stable_mask = (
                SMOOTHING_ALPHA
                *
                prev_mask
                +
                (
                    1.0
                    -
                    SMOOTHING_ALPHA
                )
                *
                mask_full
            )


        prev_mask = stable_mask


        # ====================================================
        # BINARY OBJECT MASK
        # ====================================================

        binary_mask = (
            stable_mask > 0.5
        ).astype(
            np.uint8
        )


        # ====================================================
        # MORPHOLOGY CLEANUP
        # ====================================================

        kernel = np.ones(
            (5, 5),
            np.uint8
        )


        binary_mask = cv2.morphologyEx(
            binary_mask,
            cv2.MORPH_OPEN,
            kernel
        )


        binary_mask = cv2.morphologyEx(
            binary_mask,
            cv2.MORPH_CLOSE,
            kernel
        )


        # ====================================================
        # EFFECT
        # ====================================================

        if effect_active:


            # ------------------------------------------------
            # Create stain position once
            # ------------------------------------------------

            if stain_center is None:

                create_stain_parameters(
                    binary_mask
                )


            # ------------------------------------------------
            # Animation progress
            # ------------------------------------------------

            elapsed = (
                time.time()
                -
                effect_start_time
            )


            progress = np.clip(
                elapsed
                /
                EFFECT_DURATION,
                0.0,
                1.0
            )


            # Smoothstep
            #
            # Starts slowly,
            # becomes noticeable,
            # then settles.

            progress = (
                progress
                *
                progress
                *
                (
                    3.0
                    -
                    2.0
                    *
                    progress
                )
            )


            strength = (
                progress
                *
                MAX_STRENGTH
            )


            # ------------------------------------------------
            # Generate stain
            # ------------------------------------------------

            if (
                stain_center is not None
                and
                stain_radius is not None
            ):

                stain = generate_stain_mask(
                    (
                        h,
                        w
                    ),
                    stain_center,
                    stain_radius,
                    stain_angle,
                    stain_seed
                )


                # ------------------------------------------------
                # CRITICAL:
                #
                # Stain only exists inside detected object.
                # ------------------------------------------------

                stain *= (
                    binary_mask
                    .astype(
                        np.float32
                    )
                )


                # ------------------------------------------------
                # Additional feathering
                # ------------------------------------------------

                stain = cv2.GaussianBlur(
                    stain,
                    (9, 9),
                    0
                )


                # ------------------------------------------------
                # Final alpha
                # ------------------------------------------------

                alpha = (
                    stain
                    *
                    strength
                )


                alpha = np.clip(
                    alpha,
                    0.0,
                    1.0
                )


                alpha = (
                    alpha[..., None]
                )


                # =================================================
                # COLOR INVERSION
                # =================================================

                frame_float = (
                    output_frame
                    .astype(
                        np.float32
                    )
                )


                # 255 - pixel = inverted color

                inverted = (
                    255.0
                    -
                    frame_float
                )


                # Blend normal and inverted image.

                result = (
                    frame_float
                    *
                    (
                        1.0
                        -
                        alpha
                    )
                    +
                    inverted
                    *
                    alpha
                )


                output_frame = np.clip(
                    result,
                    0,
                    255
                ).astype(
                    np.uint8
                )


        # ====================================================
        # SEND CLEAN FRAME TO OBS
        # ====================================================

        virtual_camera.send(
            output_frame
        )

        virtual_camera.sleep_until_next_frame()


        # ====================================================
        # LOCAL CONTROLLER
        #
        # This window is NOT sent to OBS.
        # ====================================================

        controller = output_frame.copy()


        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

        cv2.putText(
            controller,
            "TARGET OBJECT",
            (
                DROPDOWN_X,
                DROPDOWN_Y - 5
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 220, 220),
            1,
            cv2.LINE_AA
        )


        # ----------------------------------------------------
        # Dropdown
        # ----------------------------------------------------

        cv2.rectangle(
            controller,
            (
                DROPDOWN_X,
                DROPDOWN_Y
            ),
            (
                DROPDOWN_X
                +
                DROPDOWN_WIDTH,
                DROPDOWN_Y
                +
                DROPDOWN_HEIGHT
            ),
            (40, 40, 40),
            -1
        )


        cv2.rectangle(
            controller,
            (
                DROPDOWN_X,
                DROPDOWN_Y
            ),
            (
                DROPDOWN_X
                +
                DROPDOWN_WIDTH,
                DROPDOWN_Y
                +
                DROPDOWN_HEIGHT
            ),
            (255, 255, 255),
            2
        )


        cv2.putText(
            controller,
            selected_object,
            (
                DROPDOWN_X + 12,
                DROPDOWN_Y + 30
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


        # ----------------------------------------------------
        # Arrow
        # ----------------------------------------------------

        arrow = (
            "▲"
            if dropdown_open
            else
            "▼"
        )


        cv2.putText(
            controller,
            arrow,
            (
                DROPDOWN_X
                +
                DROPDOWN_WIDTH
                -
                32,
                DROPDOWN_Y + 30
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


        # ====================================================
        # DROPDOWN OPTIONS
        # ====================================================

        if dropdown_open:

            for display_index in range(
                VISIBLE_OPTIONS
            ):

                actual_index = (
                    display_index
                    +
                    dropdown_scroll
                )


                if (
                    actual_index
                    >=
                    len(class_list)
                ):

                    break


                option_y = (
                    DROPDOWN_Y
                    +
                    DROPDOWN_HEIGHT
                    +
                    display_index
                    *
                    OPTION_HEIGHT
                )


                if (
                    actual_index
                    ==
                    selected_class_id
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
                        DROPDOWN_X,
                        option_y
                    ),
                    (
                        DROPDOWN_X
                        +
                        DROPDOWN_WIDTH,
                        option_y
                        +
                        OPTION_HEIGHT
                    ),
                    color,
                    -1
                )


                cv2.rectangle(
                    controller,
                    (
                        DROPDOWN_X,
                        option_y
                    ),
                    (
                        DROPDOWN_X
                        +
                        DROPDOWN_WIDTH,
                        option_y
                        +
                        OPTION_HEIGHT
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
                        DROPDOWN_X + 12,
                        option_y + 22
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )


        # ====================================================
        # STATUS
        # ====================================================

        if effect_active:

            status = "INVERSION: ON"

            status_color = (
                80,
                220,
                100
            )

        else:

            status = "INVERSION: OFF"

            status_color = (
                180,
                180,
                180
            )


        status_y = 100


        if dropdown_open:

            status_y = (
                DROPDOWN_Y
                +
                DROPDOWN_HEIGHT
                +
                VISIBLE_OPTIONS
                *
                OPTION_HEIGHT
                +
                35
            )


        cv2.putText(
            controller,
            status,
            (
                20,
                status_y
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            status_color,
            2,
            cv2.LINE_AA
        )


        cv2.putText(
            controller,
            "T = invert    R = reset    Q = quit",
            (
                20,
                status_y + 30
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 220, 220),
            1,
            cv2.LINE_AA
        )


        # ====================================================
        # DEBUG MASK
        # ====================================================

        cv2.imshow(
            "Object Mask",
            binary_mask * 255
        )


        # ====================================================
        # CONTROLLER WINDOW
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
            &
            0xFF
        )


        # ====================================================
        # T = TRIGGER
        # ====================================================

        if key == ord("t"):

            effect_active = True

            effect_start_time = (
                time.time()
            )


            # Reset stain parameters so
            # a new location is selected.

            stain_center = None
            stain_radius = None

            stain_angle = None

            stain_seed = random.randint(
                0,
                999999
            )


            print()
            print(
                "======================================"
            )

            print(
                "COLOR INVERSION STARTED"
            )

            print(
                f"Target: {selected_object}"
            )

            print(
                f"Duration: "
                f"{EFFECT_DURATION:.1f}s"
            )

            print(
                "======================================"
            )


        # ====================================================
        # R = RESET
        # ====================================================

        elif key == ord("r"):

            effect_active = False

            effect_start_time = None

            stain_center = None
            stain_radius = None
            stain_angle = None

            prev_mask = None


            print()
            print(
                "Inversion reset."
            )


        # ====================================================
        # Q = QUIT
        # ====================================================

        elif key == ord("q"):

            break


finally:

    print()
    print("Shutting down...")

    virtual_camera.close()

    cap.release()

    cv2.destroyAllWindows()

    print("Done.")