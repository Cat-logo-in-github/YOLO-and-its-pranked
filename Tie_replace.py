from ultralytics import YOLO
import cv2
import numpy as np
import pyvirtualcam


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "yolov8n-seg.pt"
SOURCE_IMAGE_PATH = "Man-holding-apple-Stock-Photo.jpg"

CONFIDENCE = 0.4
SMOOTHING_ALPHA = 0.7

TARGET_PADDING = 20

CAMERA_INDEX = 0
FPS = 30

WINDOW_NAME = "Object Replacement Controller"

VISIBLE_OPTIONS = 10
OPTION_HEIGHT = 32

DROPDOWN_WIDTH = 280
DROPDOWN_HEIGHT = 45

REPLACE_X = 20
REPLACE_Y = 20

WITH_X = 20
WITH_Y = 90


# ============================================================
# LOAD YOLO
# ============================================================

print("Loading YOLO model...")

model = YOLO(MODEL_PATH)

print("YOLO loaded.")


# ============================================================
# YOLO CLASS LIST
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
# LOAD SOURCE IMAGE
# ============================================================

print("Loading source image...")

source_img = cv2.imread(
    SOURCE_IMAGE_PATH
)

if source_img is None:

    raise ValueError(
        f"Could not load source image: "
        f"{SOURCE_IMAGE_PATH}"
    )

print("Source image loaded.")


# ============================================================
# DETECT ALL OBJECTS IN SOURCE IMAGE
# ============================================================

print("Scanning source image...")

source_results = model(
    source_img,
    conf=CONFIDENCE,
    verbose=False
)[0]


source_objects = {}


if (
    source_results.boxes is not None
    and
    source_results.masks is not None
):

    for i in range(
        len(source_results.boxes)
    ):

        class_id = int(
            source_results.boxes.cls[i]
        )

        class_name = model.names[
            class_id
        ]


        # ----------------------------------------------------
        # Get mask
        # ----------------------------------------------------

        mask = (
            source_results
            .masks
            .data[i]
            .cpu()
            .numpy()
        )


        mask = cv2.resize(
            mask,
            (
                source_img.shape[1],
                source_img.shape[0]
            ),
            interpolation=cv2.INTER_NEAREST
        )


        mask_full = (
            mask > 0.5
        ).astype(np.uint8)


        ys, xs = np.where(
            mask_full == 1
        )


        if (
            len(xs) == 0
            or
            len(ys) == 0
        ):

            continue


        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        x1 = xs.min()
        x2 = xs.max()

        y1 = ys.min()
        y2 = ys.max()


        x1 = max(
            0,
            x1 - 10
        )

        x2 = min(
            source_img.shape[1],
            x2 + 10
        )

        y1 = max(
            0,
            y1 - 10
        )

        y2 = min(
            source_img.shape[0],
            y2 + 10
        )


        # ----------------------------------------------------
        # Crop object
        # ----------------------------------------------------

        cropped = source_img[
            y1:y2,
            x1:x2
        ].copy()


        cropped_mask = mask_full[
            y1:y2,
            x1:x2
        ].copy()


        # Remove background

        cropped[
            cropped_mask == 0
        ] = 0


        # ----------------------------------------------------
        # Store source object
        #
        # If multiple instances of the same class exist,
        # keep the first one for now.
        # ----------------------------------------------------

        if class_name not in source_objects:

            source_objects[class_name] = {
                "class_id": class_id,
                "image": cropped,
                "mask": cropped_mask
            }


print(
    "Objects found in source image:"
)

for name in source_objects:

    print(
        f"  - {name}"
    )


# ============================================================
# SOURCE OBJECT LIST
# ============================================================

source_object_list = list(
    source_objects.keys()
)


if len(source_object_list) == 0:

    raise ValueError(
        "No segmented objects were found "
        "in the source image."
    )


# ============================================================
# DEFAULT SELECTIONS
# ============================================================

# Object to replace
selected_target = "tie"

if selected_target not in class_list:

    selected_target = class_list[0]


# Object to replace it WITH
#
# Prefer tie if available.
#

if "tie" in source_object_list:

    selected_replacement = "tie"

else:

    selected_replacement = (
        source_object_list[0]
    )


selected_target_id = class_list.index(
    selected_target
)


print()
print(
    f"Replace: {selected_target}"
)

print(
    f"With:    {selected_replacement}"
)


# ============================================================
# DROPDOWN STATE
# ============================================================

target_dropdown_open = False
replacement_dropdown_open = False

target_scroll = 0
replacement_scroll = 0


# ============================================================
# REPLACEMENT STATE
# ============================================================

replacement_active = False

prev_mask = None


# ============================================================
# WEBCAM
# ============================================================

print()
print("Opening webcam...")

cap = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_DSHOW
)


if not cap.isOpened():

    print(
        "DirectShow failed. "
        "Trying default webcam backend..."
    )

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
        "ERROR: Could not start "
        "OBS Virtual Camera."
    )
    print()
    print(e)

    raise


print(
    f"Virtual camera started: "
    f"{virtual_camera.device}"
)


# ============================================================
# HELPER: DRAW DROPDOWN
# ============================================================

def draw_dropdown(
    image,
    x,
    y,
    value,
    options,
    selected_index,
    is_open,
    scroll
):

    # --------------------------------------------------------
    # Main box
    # --------------------------------------------------------

    cv2.rectangle(
        image,
        (x, y),
        (
            x + DROPDOWN_WIDTH,
            y + DROPDOWN_HEIGHT
        ),
        (40, 40, 40),
        -1
    )


    cv2.rectangle(
        image,
        (x, y),
        (
            x + DROPDOWN_WIDTH,
            y + DROPDOWN_HEIGHT
        ),
        (255, 255, 255),
        2
    )


    # --------------------------------------------------------
    # Selected text
    # --------------------------------------------------------

    cv2.putText(
        image,
        value,
        (
            x + 12,
            y + 30
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


    # --------------------------------------------------------
    # Arrow
    # --------------------------------------------------------

    arrow = "▲" if is_open else "▼"

    cv2.putText(
        image,
        arrow,
        (
            x + DROPDOWN_WIDTH - 32,
            y + 30
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


    # --------------------------------------------------------
    # Options
    # --------------------------------------------------------

    if not is_open:

        return


    for display_index in range(
        VISIBLE_OPTIONS
    ):

        actual_index = (
            display_index
            +
            scroll
        )


        if (
            actual_index
            >=
            len(options)
        ):

            break


        option_y = (
            y
            +
            DROPDOWN_HEIGHT
            +
            display_index
            *
            OPTION_HEIGHT
        )


        if actual_index == selected_index:

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
            image,
            (
                x,
                option_y
            ),
            (
                x + DROPDOWN_WIDTH,
                option_y + OPTION_HEIGHT
            ),
            color,
            -1
        )


        cv2.rectangle(
            image,
            (
                x,
                option_y
            ),
            (
                x + DROPDOWN_WIDTH,
                option_y + OPTION_HEIGHT
            ),
            (100, 100, 100),
            1
        )


        cv2.putText(
            image,
            options[actual_index],
            (
                x + 12,
                option_y + 22
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA
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

    global target_dropdown_open
    global replacement_dropdown_open

    global target_scroll
    global replacement_scroll

    global selected_target
    global selected_target_id

    global selected_replacement

    global prev_mask


    if event == cv2.EVENT_LBUTTONDOWN:

        # ====================================================
        # TARGET DROPDOWN
        # ====================================================

        if (
            REPLACE_X <= x <=
            REPLACE_X + DROPDOWN_WIDTH
            and
            REPLACE_Y <= y <=
            REPLACE_Y + DROPDOWN_HEIGHT
        ):

            target_dropdown_open = (
                not target_dropdown_open
            )

            replacement_dropdown_open = False

            return


        # ====================================================
        # REPLACEMENT DROPDOWN
        # ====================================================

        if (
            WITH_X <= x <=
            WITH_X + DROPDOWN_WIDTH
            and
            WITH_Y <= y <=
            WITH_Y + DROPDOWN_HEIGHT
        ):

            replacement_dropdown_open = (
                not replacement_dropdown_open
            )

            target_dropdown_open = False

            return


        # ====================================================
        # TARGET OPTIONS
        # ====================================================

        if target_dropdown_open:

            options_top = (
                REPLACE_Y
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
                target_scroll
            )


            if (
                0 <= option_index
                <
                len(class_list)
                and
                REPLACE_X <= x <=
                REPLACE_X + DROPDOWN_WIDTH
            ):

                selected_target = (
                    class_list[
                        option_index
                    ]
                )


                selected_target_id = (
                    option_index
                )


                print()
                print(
                    f"Replace object: "
                    f"{selected_target}"
                )


                prev_mask = None

                target_dropdown_open = False

                return


        # ====================================================
        # REPLACEMENT OPTIONS
        # ====================================================

        if replacement_dropdown_open:

            options_top = (
                WITH_Y
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
                replacement_scroll
            )


            if (
                0 <= option_index
                <
                len(source_object_list)
                and
                WITH_X <= x <=
                WITH_X + DROPDOWN_WIDTH
            ):

                selected_replacement = (
                    source_object_list[
                        option_index
                    ]
                )


                print()
                print(
                    f"Replace with: "
                    f"{selected_replacement}"
                )


                replacement_dropdown_open = False

                return


    # ========================================================
    # MOUSE WHEEL
    # ========================================================

    elif event == cv2.EVENT_MOUSEWHEEL:

        if target_dropdown_open:

            if flags > 0:

                target_scroll -= 1

            else:

                target_scroll += 1


            max_scroll = max(
                0,
                len(class_list)
                -
                VISIBLE_OPTIONS
            )


            target_scroll = max(
                0,
                min(
                    target_scroll,
                    max_scroll
                )
            )


        elif replacement_dropdown_open:

            if flags > 0:

                replacement_scroll -= 1

            else:

                replacement_scroll += 1


            max_scroll = max(
                0,
                len(source_object_list)
                -
                VISIBLE_OPTIONS
            )


            replacement_scroll = max(
                0,
                min(
                    replacement_scroll,
                    max_scroll
                )
            )


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
# MAIN LOOP
# ============================================================

try:

    while True:

        # ----------------------------------------------------
        # Webcam
        # ----------------------------------------------------

        ret, frame = cap.read()


        if not ret:

            continue


        output_frame = frame.copy()


        # ====================================================
        # YOLO WEBCAM DETECTION
        # ====================================================

        results = model(
            frame,
            conf=CONFIDENCE,
            verbose=False
        )[0]


        target_mask = np.zeros(
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
                    selected_target_id
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


                target_mask = np.maximum(
                    target_mask,
                    mask
                )


        # ====================================================
        # TEMPORAL SMOOTHING
        # ====================================================

        if prev_mask is None:

            stable_mask = target_mask

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
                target_mask
            )


        prev_mask = stable_mask


        # ====================================================
        # BINARY MASK
        # ====================================================

        binary_mask = (
            stable_mask > 0.5
        ).astype(np.uint8)


        # ====================================================
        # MORPHOLOGY
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
        # APPLY REPLACEMENT
        # ====================================================

        if (
            replacement_active
            and
            selected_replacement
            in
            source_objects
        ):

            source_data = (
                source_objects[
                    selected_replacement
                ]
            )


            replacement_image = (
                source_data["image"]
            )

            replacement_mask = (
                source_data["mask"]
            )


            ys, xs = np.where(
                binary_mask == 1
            )


            if (
                len(xs) > 0
                and
                len(ys) > 0
            ):

                # ------------------------------------------------
                # Target bounds
                # ------------------------------------------------

                x1 = xs.min()
                x2 = xs.max()

                y1 = ys.min()
                y2 = ys.max()


                x1 = max(
                    0,
                    x1 - TARGET_PADDING
                )

                x2 = min(
                    w,
                    x2 + TARGET_PADDING
                )

                y1 = max(
                    0,
                    y1 - TARGET_PADDING
                )

                y2 = min(
                    h,
                    y2 + TARGET_PADDING
                )


                target_w = x2 - x1
                target_h = y2 - y1


                if (
                    target_w > 20
                    and
                    target_h > 20
                ):

                    # =================================================
                    # PRESERVE REPLACEMENT ASPECT RATIO
                    # =================================================

                    source_h, source_w = (
                        replacement_image.shape[:2]
                    )


                    scale = min(
                        target_w / source_w,
                        target_h / source_h
                    )


                    new_w = max(
                        1,
                        int(
                            source_w
                            * scale
                        )
                    )

                    new_h = max(
                        1,
                        int(
                            source_h
                            * scale
                        )
                    )


                    resized_image = cv2.resize(
                        replacement_image,
                        (
                            new_w,
                            new_h
                        ),
                        interpolation=cv2.INTER_LINEAR
                    )


                    resized_mask = cv2.resize(
                        replacement_mask,
                        (
                            new_w,
                            new_h
                        ),
                        interpolation=cv2.INTER_NEAREST
                    ).astype(
                        np.float32
                    )


                    # ------------------------------------------------
                    # Center replacement in target region
                    # ------------------------------------------------

                    offset_x = (
                        target_w
                        -
                        new_w
                    ) // 2


                    offset_y = (
                        target_h
                        -
                        new_h
                    ) // 2


                    paste_x1 = (
                        x1
                        +
                        offset_x
                    )

                    paste_y1 = (
                        y1
                        +
                        offset_y
                    )


                    paste_x2 = (
                        paste_x1
                        +
                        new_w
                    )

                    paste_y2 = (
                        paste_y1
                        +
                        new_h
                    )


                    # ------------------------------------------------
                    # Feather replacement mask
                    # ------------------------------------------------

                    resized_mask = cv2.GaussianBlur(
                        resized_mask,
                        (7, 7),
                        0
                    )


                    resized_mask = np.clip(
                        resized_mask,
                        0.0,
                        1.0
                    )


                    resized_mask = (
                        resized_mask[..., None]
                    )


                    # ------------------------------------------------
                    # Blend
                    # ------------------------------------------------

                    roi = output_frame[
                        paste_y1:paste_y2,
                        paste_x1:paste_x2
                    ].astype(
                        np.float32
                    )


                    resized_image = (
                        resized_image.astype(
                            np.float32
                        )
                    )


                    blended = (
                        roi
                        *
                        (
                            1.0
                            -
                            resized_mask
                        )
                        +
                        resized_image
                        *
                        resized_mask
                    )


                    output_frame[
                        paste_y1:paste_y2,
                        paste_x1:paste_x2
                    ] = np.clip(
                        blended,
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
        # IMPORTANT:
        # This is NOT sent to OBS.
        # ====================================================

        controller = output_frame.copy()


        # ----------------------------------------------------
        # Draw labels
        # ----------------------------------------------------

        cv2.putText(
            controller,
            "REPLACE",
            (
                REPLACE_X,
                REPLACE_Y - 5
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 220, 220),
            1,
            cv2.LINE_AA
        )


        cv2.putText(
            controller,
            "WITH",
            (
                WITH_X,
                WITH_Y - 5
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 220, 220),
            1,
            cv2.LINE_AA
        )


        # ----------------------------------------------------
        # Target dropdown
        # ----------------------------------------------------

        target_index = class_list.index(
            selected_target
        )


        draw_dropdown(
            controller,
            REPLACE_X,
            REPLACE_Y,
            selected_target,
            class_list,
            target_index,
            target_dropdown_open,
            target_scroll
        )


        # ----------------------------------------------------
        # Replacement dropdown
        # ----------------------------------------------------

        replacement_index = (
            source_object_list.index(
                selected_replacement
            )
        )


        draw_dropdown(
            controller,
            WITH_X,
            WITH_Y,
            selected_replacement,
            source_object_list,
            replacement_index,
            replacement_dropdown_open,
            replacement_scroll
        )


        # ----------------------------------------------------
        # Status
        # ----------------------------------------------------

        status_y = 160


        if replacement_active:

            status_text = (
                "REPLACEMENT: ON"
            )

            status_color = (
                80,
                220,
                100
            )

        else:

            status_text = (
                "REPLACEMENT: OFF"
            )

            status_color = (
                180,
                180,
                180
            )


        # Move status down if dropdown is open

        if target_dropdown_open:

            status_y = (
                REPLACE_Y
                +
                DROPDOWN_HEIGHT
                +
                VISIBLE_OPTIONS
                *
                OPTION_HEIGHT
                +
                35
            )

        elif replacement_dropdown_open:

            status_y = (
                WITH_Y
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
            status_text,
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
            "T = apply    R = reset    Q = quit",
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


        # ----------------------------------------------------
        # Show local mask
        # ----------------------------------------------------

        cv2.imshow(
            "Object Mask",
            binary_mask * 255
        )


        # ----------------------------------------------------
        # Show controller
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # T = APPLY
        # ----------------------------------------------------

        if key == ord("t"):

            if (
                selected_replacement
                not in source_objects
            ):

                print(
                    "Replacement object "
                    "is unavailable."
                )

            else:

                replacement_active = True

                prev_mask = None

                print()
                print(
                    "================================"
                )

                print(
                    f"REPLACE: "
                    f"{selected_target}"
                )

                print(
                    f"WITH:    "
                    f"{selected_replacement}"
                )

                print(
                    "================================"
                )


        # ----------------------------------------------------
        # R = RESET
        # ----------------------------------------------------

        elif key == ord("r"):

            replacement_active = False

            prev_mask = None

            print()
            print(
                "Replacement OFF."
            )


        # ----------------------------------------------------
        # Q = QUIT
        # ----------------------------------------------------

        elif key == ord("q"):

            break


finally:

    print()
    print("Shutting down...")

    virtual_camera.close()

    cap.release()

    cv2.destroyAllWindows()

    print("Done.")