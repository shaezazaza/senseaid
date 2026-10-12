"""
SenseAid - A Unified Computer Vision Assistant

Modes (switch with hand gestures, held ~1 s OUTSIDE the green box):
    1 finger  -> Currency mode (which note you're holding)
    2 fingers -> Person mode (how many people are ahead)
    3 fingers -> Repeat last message
Keyboard backup: 1 / 2 modes, r repeat, q quit.

Usage:
    python senseaid.py collect 100   # take webcam photos of a note (SPACE = save, q = quit)
    python senseaid.py train     # train the currency model
    python senseaid.py run       # start the live webcam app
    python senseaid.py export    # (optional) convert model to TFLite for the phone

Install:
    pip install tensorflow opencv-python mediapipe pyttsx3 scikit-learn matplotlib
    (If `mp.solutions` is missing in your MediaPipe version: pip install mediapipe==0.10.14)
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"      # hide TensorFlow info spam
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import sys
import json
import time
import queue
import threading

import cv2
import numpy as np

# ---- Change these paths for your laptop ----
BASE_DIR      = r"C:\Users\shaez\senseaid"
CURRENCY_DATA = os.path.join(BASE_DIR, "dataset", "currency")  # currency/10, 20, 50, 100, 200, 500
MODEL_DIR     = os.path.join(BASE_DIR, "models")

IMG_SIZE = 224
os.makedirs(MODEL_DIR, exist_ok=True)


# =====================================================================
# 1. Currency model training
# =====================================================================

def load_image_folder(data_dir, img_size=IMG_SIZE):
    """Loads data_dir/<class_name>/*.jpg -> X (0-255), y (ints), class_names."""
    class_names = sorted(d for d in os.listdir(data_dir)
                         if os.path.isdir(os.path.join(data_dir, d)))
    images, labels = [], []
    for idx, name in enumerate(class_names):
        folder = os.path.join(data_dir, name)
        files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if len(files) < 10:
            raise SystemExit(f"Only {len(files)} images in {folder} - add at least 10 (30+ is better).")
        for file in files:
            img = cv2.imread(os.path.join(folder, file))
            if img is None:              # skip non-image / broken files
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_size, img_size))
            images.append(img)
            labels.append(idx)
    X = np.array(images, dtype="float32")   # keep 0-255; the model rescales itself
    y = np.array(labels)
    print(f"Loaded {len(X)} images, {len(class_names)} classes: {class_names}")
    return X, y, class_names


def build_classifier(num_classes, img_size=IMG_SIZE):
    """MobileNetV2 (pretrained) + small new head. Much better than a CNN
    from scratch on ~100 images, and light enough for TFLite."""
    import tensorflow as tf

    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomBrightness(0.2),
        tf.keras.layers.RandomContrast(0.2),
    ], name="augment")

    base = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3), include_top=False, weights="imagenet")
    base.trainable = False   # freeze pretrained layers

    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    x = augment(inputs)
    x = tf.keras.layers.Rescaling(1 / 127.5, offset=-1)(x)   # 0-255 -> -1..1
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def train_and_save(data_dir, name, epochs=40):
    import tensorflow as tf
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split

    X, y, class_names = load_image_folder(data_dir)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)   # every class in both splits

    model = build_classifier(len(class_names))
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=8, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=16,
                        validation_data=(X_test, y_test), callbacks=[early_stop])

    _, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"{name}: test accuracy = {acc:.2%}")

    model.save(os.path.join(MODEL_DIR, f"{name}_model.keras"))
    with open(os.path.join(MODEL_DIR, f"{name}_classes.json"), "w") as f:
        json.dump(class_names, f)

    # Save the accuracy plot instead of blocking with plt.show()
    plt.figure()
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="val")
    plt.title(f"{name} accuracy"); plt.xlabel("epoch"); plt.legend()
    plot_path = os.path.join(MODEL_DIR, f"{name}_accuracy.png")
    plt.savefig(plot_path); plt.close()
    print("Saved plot:", plot_path)


# =====================================================================
# 1b. Collect webcam photos (fixes "real note looks different from web images")
# =====================================================================

X1, Y1, X2, Y2 = 150, 100, 450, 350      # green box (ROI) - same box used in run()


def collect(note):
    folder = os.path.join(CURRENCY_DATA, note)
    if not os.path.isdir(folder):
        raise SystemExit(f"No folder {folder}. Use one of: {sorted(os.listdir(CURRENCY_DATA))}")
    cap = cv2.VideoCapture(0)
    saved = 0
    print(f"Hold a {note} note inside the green box. SPACE = save photo, q = quit.")
    print("Tip: change angle, distance, front/back and lighting between photos.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)                 # same as run()
        roi = frame[Y1:Y2, X1:X2].copy()
        show = frame.copy()
        cv2.rectangle(show, (X1, Y1), (X2, Y2), (0, 255, 0), 2)
        cv2.putText(show, f"{note}: saved {saved}  (SPACE save, q quit)", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("SenseAid - collect", show)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            path = os.path.join(folder, f"cam_{int(time.time() * 1000)}.jpg")
            cv2.imwrite(path, roi)
            saved += 1
        elif key == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {saved} photos to {folder}")


# =====================================================================
# 2. Voice output (background thread, so the camera never freezes)
# =====================================================================

class Speaker:
    def __init__(self, rate=150):
        self.rate = rate
        self.q = queue.Queue()
        self.last_message = ""
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        import pyttsx3
        while True:
            text = self.q.get()
            engine = pyttsx3.init()          # fresh engine each time = no hangs on Windows
            engine.setProperty("rate", self.rate)
            engine.say(text)
            engine.runAndWait()
            engine.stop()

    def say(self, text):
        print("[voice]", text)
        self.last_message = text
        self.q.put(text)

    def repeat(self):
        if self.last_message:
            self.q.put(self.last_message)


# =====================================================================
# 3. Stability helpers (speak / switch only after a steady result)
# =====================================================================

class StableLabel:
    def __init__(self, hold_seconds=2.0, threshold=0.6):
        self.hold = hold_seconds
        self.threshold = threshold   # 0.35 was too low for 5 classes
        self.label, self.start, self.done = None, 0, False

    def update(self, label, confidence):
        """Returns the label ONCE after it has been stable for hold_seconds."""
        if confidence < self.threshold:
            self.label, self.done = None, False
            return None
        if label != self.label:
            self.label, self.start, self.done = label, time.time(), False
            return None
        if not self.done and time.time() - self.start >= self.hold:
            self.done = True
            return label
        return None

    def reset(self):
        self.label, self.done = None, False


class GestureControl:
    def __init__(self, hold_seconds=1.0):
        self.hold = hold_seconds
        self.current, self.start, self.fired = None, 0, False

    def update(self, fingers):
        """Returns the finger count ONCE after it's held steady."""
        if fingers != self.current:
            self.current, self.start, self.fired = fingers, time.time(), False
            return None
        if fingers is not None and not self.fired and time.time() - self.start >= self.hold:
            self.fired = True
            return fingers
        return None


def count_fingers(hand_landmarks):
    """Raised fingers (index to pinky): tip above the middle joint."""
    lm = hand_landmarks.landmark
    tips, pips = [8, 12, 16, 20], [6, 10, 14, 18]
    return sum(lm[t].y < lm[p].y for t, p in zip(tips, pips))


# =====================================================================
# 4. Live app
# =====================================================================

def load_model_and_classes(name):
    import tensorflow as tf
    model = tf.keras.models.load_model(os.path.join(MODEL_DIR, f"{name}_model.keras"))
    with open(os.path.join(MODEL_DIR, f"{name}_classes.json")) as f:
        classes = json.load(f)
    return model, classes


def classify_roi(model, classes, roi):
    img = cv2.cvtColor(cv2.resize(roi, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2RGB)
    probs = model.predict(np.expand_dims(img.astype("float32"), 0), verbose=0)[0]
    i = int(np.argmax(probs))
    return classes[i], float(probs[i])


def load_hands():
    """Gesture control is optional: if MediaPipe is missing or broken,
    the app still runs with keyboard controls."""
    try:
        import mediapipe as mp
        mp_hands = mp.solutions.hands
        return mp_hands, mp.solutions.drawing_utils, mp_hands.Hands(
            max_num_hands=1, min_detection_confidence=0.6)
    except Exception as e:
        print("Gestures OFF (MediaPipe problem):", e)
        print("Use the keyboard instead: 1 / 2 / r / q")
        return None, None, None


def run():
    currency_model, currency_classes = load_model_and_classes("currency")

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    mp_hands, mp_draw, hands = load_hands()

    speaker = Speaker()
    stable = StableLabel(hold_seconds=2.0, threshold=0.6)
    gesture = GestureControl(hold_seconds=1.0)

    MODES = {1: "currency", 2: "person"}
    state = {"mode": "currency", "last_people": -1}
    frame_no = 0
    label, conf = "", 0.0

    x1, y1, x2, y2 = X1, Y1, X2, Y2          # green box (ROI) for notes

    def switch_mode(new_mode):
        state["mode"] = new_mode
        state["last_people"] = -1
        stable.reset()
        speaker.say(f"{new_mode} mode")

    cap = cv2.VideoCapture(0)
    speaker.say("SenseAid started. Currency mode.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        frame_no += 1
        mode = state["mode"]

        # ---------- Gestures (only outside the green box) ----------
        fingers = None
        if hands is not None:
            try:
                result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if result.multi_hand_landmarks:
                    hand = result.multi_hand_landmarks[0]
                    cx, cy = hand.landmark[9].x * w, hand.landmark[9].y * h   # middle of palm
                    in_box = x1 < cx < x2 and y1 < cy < y2
                    if not in_box:
                        fingers = count_fingers(hand)
                        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            except Exception as e:
                print("Gestures OFF (MediaPipe error):", e)
                print("Use the keyboard instead: 1 / 2 / r / q")
                hands = None

        g = gesture.update(fingers)
        if g in MODES:
            switch_mode(MODES[g])
        elif g == 3:
            speaker.repeat()

        # ---------- Mode logic (models every 5th frame to stay smooth) ----------
        if mode == "currency":
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if frame_no % 5 == 0:
                roi = frame[y1:y2, x1:x2]
                label, conf = classify_roi(currency_model, currency_classes, roi)
                confirmed = stable.update(label, conf)
                if confirmed:
                    speaker.say(f"{confirmed} rupees")

            cv2.putText(frame, f"{label} ({conf:.0%})", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        elif mode == "person" and frame_no % 10 == 0:
            boxes, _ = hog.detectMultiScale(frame, winStride=(8, 8))
            for (x, y, bw, bh) in boxes:
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            n = len(boxes)
            if n != state["last_people"]:
                state["last_people"] = n
                speaker.say("No one ahead" if n == 0 else
                            f"{n} {'person' if n == 1 else 'people'} ahead")

        # ---------- Status bar ----------
        cv2.putText(frame, f"Mode: {mode.upper()}",
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow("SenseAid", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key in (ord("1"), ord("2")):
            switch_mode(MODES[int(chr(key))])
        elif key == ord("r"):
            speaker.repeat()

    cap.release()
    if hands is not None:
        hands.close()
    cv2.destroyAllWindows()


# =====================================================================
# 5. Optional: export to TensorFlow Lite for the phone app
# =====================================================================

def export_tflite():
    import tensorflow as tf
    for name in ["currency"]:
        model, _ = load_model_and_classes(name)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]   # smaller + faster
        tflite = converter.convert()
        path = os.path.join(MODEL_DIR, f"{name}_model.tflite")
        with open(path, "wb") as f:
            f.write(tflite)
        print(f"{name}: {len(tflite) / 1e6:.1f} MB -> {path}")


# =====================================================================
if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "run"

    if command == "train":
        train_and_save(CURRENCY_DATA, "currency")
    elif command == "collect" and len(sys.argv) > 2:
        collect(sys.argv[2])
    elif command == "run":
        run()
    elif command == "export":
        export_tflite()
    else:
        print(__doc__)
