import os
import tkinter as tk
from tkinter import PhotoImage, filedialog, messagebox
from PIL import Image, ImageTk, ImageOps
import cv2
import random
import psutil
import time
import numpy as np
import tensorflow as tf
import pickle
import dlib
from imgbeddings import imgbeddings
from PIL import Image

cap = None   # global camera object
cam_label_id = None
cam_imgtk = None
no_cam = None
no_cam_text = None
breed_frame = None

success_count = 0
fail_count = 0
frame_count = 0
last_check_time = 0
DETECTION_INTERVAL = 2.0
score = None
status_text = "Authenticating"
status_color = "saddle brown"
locked = False

MATCH_THRESHOLD = 0.865
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
ibed = imgbeddings()
db_faces = np.load("db_faces.npy")

# Load TFLite model and allocate tensors
dog_interpreter = tf.lite.Interpreter(model_path="dog_breed_model.tflite")
dog_interpreter.allocate_tensors()
dog_input_details = dog_interpreter.get_input_details()
dog_output_details = dog_interpreter.get_output_details()

# Load TFLite model and allocate tensors
cat_interpreter = tf.lite.Interpreter(model_path="cat_breed_model.tflite")
cat_interpreter.allocate_tensors()
cat_input_details = cat_interpreter.get_input_details()
cat_output_details = cat_interpreter.get_output_details()

# Load dog indices
with open("dog_class_indices.pkl", "rb") as f:
    dog_class_indices = pickle.load(f)
dog_inv_map = {v: k for k, v in dog_class_indices.items()}

with open("cat_class_indices.pkl", "rb") as f:
    cat_class_indices = pickle.load(f)
cat_inv_map = {v: k for k, v in cat_class_indices.items()}
# ========================================================
# Helpers
# ========================================================
def load_image(path):
    if not os.path.isfile(path):
        raise FileNotFoundError("File not found")
    img = Image.open(path).convert("RGB")
    img.thumbnail((360, 220))
    img = ImageOps.expand(img, border=1, fill="white")
    return img, ImageTk.PhotoImage(img)

def classify_image(path, mode="dog"):
    img = Image.open(path).convert("RGB")

    if mode.lower() == "dog":
        img = img.resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        dog_interpreter.set_tensor(dog_input_details[0]['index'], img_array)
        dog_interpreter.invoke()
        predictions = dog_interpreter.get_tensor(dog_output_details[0]['index'])[0]

        sorted_idx = np.argsort(predictions)[::-1][:3]
        results = [(dog_inv_map[i], float(predictions[i])) for i in sorted_idx]
        return results

    elif mode.lower() == "cat":
        img = img.resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        cat_interpreter.set_tensor(cat_input_details[0]['index'], img_array)
        cat_interpreter.invoke()
        predictions = cat_interpreter.get_tensor(cat_output_details[0]['index'])[0]

        sorted_idx = np.argsort(predictions)[::-1][:3]
        results = [(cat_inv_map[i], float(predictions[i])) for i in sorted_idx]
        return results

    else:
        raise ValueError("Mode must be 'dog' or 'cat'")

def round_rectangle(canvas, x1, y1, x2, y2, r = 50, **kwargs):    
    points = (x1+r, y1, x1+r, y1, x2-r, y1, x2-r, y1, x2, y1, x2, y1+r, x2, y1+r, x2, y2-r, x2, y2-r, x2, y2, x2-r, y2, x2-r, y2, x1+r, y2, x1+r, y2, x1, y2, x1, y2-r, x1, y2-r, x1, y1+r, x1, y1+r, x1, y1)
    return canvas.create_polygon(points, **kwargs, smooth=True)

def create_modern_button(canvas, x1, y1, x2, y2, r, text, command, state="enabled"):
    # Draw rounded rectangle
    btn_id = round_rectangle(canvas, x1, y1, x2, y2, r, fill="saddle brown", outline="")

    # Add text centered inside
    txt_id = canvas.create_text(
        (x1 + x2) // 2,
        (y1 + y2) // 2,
        text=text,
        fill="old lace" if state == "enabled" else "gray",
        font=("Verdana", 12, "bold")
    )

    # Store state in a dictionary
    button_state = {"value": state}

    # Bind both rectangle and text to same command
    def on_click(event):
        if button_state["value"] == "enabled":
            command()

    canvas.tag_bind(btn_id, "<Button-1>", on_click)
    canvas.tag_bind(txt_id, "<Button-1>", on_click)

    # Optional hover effect
    def on_enter(event):
        canvas.itemconfig(btn_id, fill="#8B5524")  # lighter brown
    def on_leave(event):
        canvas.itemconfig(btn_id, fill="#7B4316")  # original

    canvas.tag_bind(btn_id, "<Enter>", on_enter)
    canvas.tag_bind(txt_id, "<Enter>", on_enter)
    canvas.tag_bind(btn_id, "<Leave>", on_leave)
    canvas.tag_bind(txt_id, "<Leave>", on_leave)

    # Function to update state later
    def set_state(new_state):
        button_state["value"] = new_state
        if new_state == "enabled":
            canvas.itemconfig(txt_id, fill="old lace")
            canvas.itemconfig(btn_id, fill="saddle brown")
        else:
            canvas.itemconfig(txt_id, fill="gray")
            canvas.itemconfig(btn_id, fill="#5A3A1C")  # darker brown for disabled

    return btn_id, txt_id, set_state

def update_time():
    now = time.strftime("%H : %M")
    canvas.itemconfig(time_text_id, text=now)
    window.after(1000, update_time)  # update every 1s

def close_camera():
    global cap
    if cap is not None and cap.isOpened():
        cap.release()
        cap = None
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def enhance_face(pil_img):

    pil_img = pil_img.convert("RGB")
    # Increase sharpness
    enhancer = ImageEnhance.Sharpness(pil_img)
    pil_img = enhancer.enhance(1)  # 1.0 = original, >1 = sharper

    # Increase contrast
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(1.1)  # 1.0 = original, >1 = more contrast

    # Optional: Slight brightness adjustment
    enhancer = ImageEnhance.Brightness(pil_img)
    pil_img = enhancer.enhance(1.2)

    # Optional: Slight Gaussian blur to reduce noise (use small radius)
    pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=1))

    return pil_img

def safe_crop(img, box):
    h, w = img.shape[:2]
    x1, y1 = max(0, box.left()), max(0, box.top())
    x2, y2 = min(w, box.right()), min(h, box.bottom())
    if x2 <= x1 or y2 <= y1:
        return None, None
    return img[y1:y2, x1:x2], (x1, y1, x2, y2)

def align_face(img, shape):
    left_eye = (shape.part(36).x, shape.part(36).y)
    right_eye = (shape.part(45).x, shape.part(45).y)

    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))

    eyes_center = ((left_eye[0] + right_eye[0]) // 2,
                   (left_eye[1] + right_eye[1]) // 2)

    M = cv2.getRotationMatrix2D(eyes_center, angle, 1)
    aligned = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    return aligned

def load_db_embeddings(db_folder, detector, predictor, ibed):
    if not os.path.exists(db_folder):
        raise FileNotFoundError(f"Authorized faces folder not found: {db_folder}")

    embeddings = []
    for file in os.listdir(db_folder):
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        path = os.path.join(db_folder, file)
        db_img = cv2.imread(path)
        db_rgb = cv2.cvtColor(db_img, cv2.COLOR_BGR2RGB)

        detections = detector(db_rgb, 1)
        if not detections:
            print(f"[WARN] No face detected in {file}, skipping.")
            continue

        f = detections[0]
        shape = predictor(db_rgb, f)
        aligned = align_face(db_rgb, shape)

        face_crop, _ = safe_crop(aligned, f)
        if face_crop is None:
            print(f"[WARN] Could not crop {file}, skipping.")
            continue

        pil_face = Image.fromarray(face_crop).resize((160, 160))
        pil_face = enhance_face(pil_face)
        emb = ibed.to_embeddings([pil_face])[0]  # FIXED
        embeddings.append(emb)

        print(f"[INFO] Added DB embedding from {file}")

    if not embeddings:
        raise ValueError("No valid faces found in authorized folder.")

    print(f"[INFO] Loaded {len(embeddings)} authorized face embeddings.")
    return embeddings
# ========================================================
# Pages
# ========================================================
def show_face_auth():
    global cap, cam_label_id, no_cam_text, no_cam, dog_logo, cat_logo, success_text_id, fail_text_id, start_btn, start_btn_text, start_btn_state
    global time_text_id, battery_text_id, paw_photo, paw_label_id
    canvas.delete("all")

    canvas.create_rectangle(0, 0, 800, 800, fill="sienna3", outline = "")
    #Tab Bar
    canvas.create_rectangle(0, 0, 800, 35, fill="sienna4", outline="")
    round_rectangle(canvas, 55, 180, 545, 735, r=30, fill="peach puff", outline="dark orange")
    time_text_id = canvas.create_text(260, 20, text="--:--", fill="old lace", font=("Courier New", 14, "bold"), anchor="w")
    update_time()
    paw_photo = tk.PhotoImage(file = "paw_logo.png")
    paw_label_id = canvas.create_image(17, 18, image = paw_photo)

    canvas.create_text(300, 75, text="WELCOME BACK!", fill = "old lace", font=("Verdana", 19, "bold"))
    status_id = canvas.create_text(300, 550, text="Idle", fill = "saddle brown", font=("Verdana", 17, "bold"))

    # Draw a rectangle for camera feed    
    canvas.create_rectangle(90, 208, 510, 515, fill = "AntiqueWhite2", outline="")

    cam_label_id = canvas.create_image(300, 270)

    dog_logo = tk.PhotoImage(file = "Dog_logo.png")
    canvas.create_image(300, 148, image = dog_logo)

    no_cam = tk.PhotoImage(file = "No_camera.png")
    cam_label_id = canvas.create_image(300, 350, image = no_cam)
    #Indicators
    success_text_id = canvas.create_text(200, 585, text="Success: 0/5", fill="green", font=("Verdana", 14))
    fail_text_id    = canvas.create_text(400, 585, text="Fail: 0/5", fill="red", font=("Verdana", 14))

    def start_auth():
        global cap, cam_label_id, stop_btn
        canvas.itemconfig(status_id, text="Opening camera...", fill="saddle brown")
        window.after(1000, open_camera, status_id)
   
    cat_logo = tk.PhotoImage(file = "Cat_logo.png")
    canvas.create_image(300, 637, image = cat_logo)
            
    start_btn, start_btn_text, start_btn_state = create_modern_button(canvas, 130, 665, 470, 715, 20, "Start Authentication", start_auth)

def open_camera(status_id):
    global cap
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    if not cap.isOpened():
        canvas.itemconfig(status_id, text="Camera not found!", fill="red")
        return
    update_frame(status_id)

def update_frame(status_id):
    global cap, cam_label_id, cam_imgtk
    global success_count, fail_count, start_btn_text
    global last_check_time, score, locked
    global last_status_text, last_status_color

    if cap is None or not cap.isOpened():
        return

    ret, frame = cap.read()
    if not ret:
        window.after(10, lambda: update_frame(status_id))
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = detector(gray, 0)

    score = None

    if detections:
        f = detections[0]
        x, y, w, h = f.left(), f.top(), f.width(), f.height()
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        now = time.time()
        if now - last_check_time > DETECTION_INTERVAL:
            last_check_time = now
            face_crop = frame[y:y+h, x:x+w]

            if face_crop.size > 0:
                pil_face = Image.fromarray(
                    cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                ).resize((160, 160))

                try:
                    query_emb = ibed.to_embeddings([pil_face])[0]
                    score = float(np.max([
                        np.dot(query_emb, db) /
                        (np.linalg.norm(query_emb) * np.linalg.norm(db) + 1e-9)
                        for db in db_faces
                    ]))

                    if not locked:
                        if score > MATCH_THRESHOLD:
                            success_count += 1
                            canvas.itemconfig(success_text_id, text=f"Success: {success_count}/5")
                            last_status_text = f"Matching Score: {score:.2f}"
                            last_status_color = "green"
                            if success_count >= 5:
                                locked = False
                                last_status_text = "Authentication Success :)"
                                last_status_color = "green"
                                window.after(2000, show_selection)
                        else:
                            fail_count += 1
                            canvas.itemconfig(fail_text_id, text=f"Fail: {fail_count}/5")
                            last_status_text = f"Matching Score: {score:.2f}"
                            last_status_color = "red"

                            if fail_count >= 5 and success_count <= 4:
                                last_status_text = "Authentication Fail :("
                                last_status_color = "red"
                                canvas.itemconfig(start_btn_text, text="Restart")
                                def restart_auth():
                                    global fail_count, success_count, locked
                                    close_camera()
                                    show_face_auth()
                                    fail_count = 0
                                    success_count = 0
                                    locked = False
                                canvas.tag_bind(start_btn, "<Button-1>", lambda e: restart_auth())
                                canvas.tag_bind(start_btn_text, "<Button-1>", lambda e: restart_auth())
                                locked = True

                except Exception as e:
                    last_status_text = f"Embedding error: {e}"
                    last_status_color = "red"
    else:
        last_status_text = "No face detected"
        last_status_color = "red"

    # --- always display the last status ---
    canvas.itemconfig(status_id, text=last_status_text, fill=last_status_color)

    # show frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    img.thumbnail((420, 525))
    cam_imgtk = ImageTk.PhotoImage(img)
    canvas.coords(cam_label_id, 300, 365)
    canvas.itemconfig(cam_label_id, image=cam_imgtk)

    window.after(1, lambda: update_frame(status_id))

def show_selection():
    global time_text_id, dog_btn, dog_btn_text, dog_btn_state, cat_btn, cat_btn_text, cat_btn_state, select_dog, select_cat, paw_photo, paw_label_id
    canvas.delete("all")
    canvas.create_rectangle(0, 0, 800, 600, fill="old lace", outline="")

    canvas.create_rectangle(0, 0, 800, 800, fill="sienna3", outline = "")
    #Tab Bar
    canvas.create_rectangle(0, 0, 800, 35, fill="sienna4", outline="")
    time_text_id = canvas.create_text(260, 20, text="--:--", fill="old lace", font=("Courier New", 14, "bold"), anchor="w")
    update_time()
    paw_photo = tk.PhotoImage(file = "paw_logo.png")
    paw_label_id = canvas.create_image(17, 18, image = paw_photo)

    canvas.create_text(300, 80, text="Choose Category", fill = "old lace", font=("Verdana", 28, "bold"))

    select_dog = tk.PhotoImage(file = "Select_dog.png")
    canvas.create_image(190, 215, image = select_dog)

    select_cat = tk.PhotoImage(file = "Select_cat.png")
    canvas.create_image(400, 215, image = select_cat)

    dog_btn, dog_btn_text, dog_btn_state = create_modern_button(canvas, 130, 310, 260, 350, 20, "Dog", show_dog_page)
    cat_btn, cat_btn_text, cat_btn_state = create_modern_button(canvas, 330, 310, 460, 350, 20, "Cat", show_cat_page)

def show_dog_page():
    show_classification_page("Dog Breed Identification", "Dog", show_selection)


def show_cat_page():
    show_classification_page("Cat Breed Identification", "Cat", show_selection)


def show_classification_page(title, mode, back_callback):
    global time_text_id, choose_btn, choose_btn_text, choose_btn_state, clear_btn, clear_btn_text, clear_btn_state, classify_btn, classify_btn_text, classify_btn_state
    global no_photo, photo_label_id, paw_photo, paw_label_id, camera_btn, camera_btn_text, camera_btn_state, capture_btn, capture_btn_text, capture_btn_state
    canvas.delete("all")
    canvas.create_rectangle(0, 0, 800, 600, fill="old lace", outline="")
    canvas.create_rectangle(0, 0, 800, 800, fill="sienna3", outline = "")
    # Tab Bar
    canvas.create_rectangle(0, 0, 800, 35, fill="sienna4", outline="")
    # Title top-left
    canvas.create_text(30, 65, text = title, fill = "old lace", font=("Verdana", 20, "bold"), anchor="w")

    time_text_id = canvas.create_text(260, 20, text="--:--", fill="old lace", font=("Courier New", 14, "bold"), anchor="w")
    update_time()

    # Back button top-right
    back_btn = tk.Button(window, text="Back", bd = 0, bg ="sienna3", font=("Verdana", 12, "underline"), fg = "old lace", command=lambda: [close_camera(), back_callback()])
    canvas.create_window(560, 65, window=back_btn, anchor="e")

    # Left column buttons
    choose_btn, choose_btn_text, choose_btn_state= create_modern_button(canvas, 30, 105, 190, 145, 20, "Choose Image", lambda: choose_file(mode))
    camera_btn, camera_btn_text, camera_btn_state = create_modern_button(canvas, 30, 155, 190, 195, 20, "Open Camera", lambda: start_camera_classify(mode))
    capture_btn, capture_btn_text, capture_btn_state = create_modern_button(canvas, 30, 205, 190, 245, 20, "Capture Photo", lambda: capture_photo(mode), state="disabled")
    classify_btn, classify_btn_text, classify_btn_state= create_modern_button(canvas, 30, 255, 190, 295, 20, "Start Classify", lambda: classify(mode), state="disabled")    
    clear_btn, clear_btn_text, clear_btn_state = create_modern_button(canvas, 30, 305, 190, 345, 20, "Clear", lambda: clear_preview(mode))


    # Right preview box
    preview_box_id = canvas.create_rectangle(220, 100, 555, 340, outline="old lace")
    no_photo = tk.PhotoImage(file = "No_image.png")
    photo_label_id = canvas.create_image(390, 200, image = no_photo)

    paw_photo = tk.PhotoImage(file = "paw_logo.png")
    paw_label_id = canvas.create_image(17, 18, image = paw_photo)

    preview_text_id = canvas.create_text(395, 270, text="No image selected.", fill = "old lace", font=("Verdana", 14, "bold"))

    preview_states[mode] = {
        "box_id" : preview_box_id,
        "text_id": preview_text_id,
        "img": None,
        "path": None,
        "classify_btn_state": classify_btn_state,
        "capture_btn_state": capture_btn_state,
        "camera_btn_state": camera_btn_state,
        "results": []
    }

    # Results section
    canvas.create_text(300, 405, text="Top 3 Results:", fill = "old lace", font=("Verdana", 16, "bold", "underline"))

def start_camera_classify(mode):
    global cap, is_camera_active
    
    # Close any existing camera connection
    close_camera()
    
    cap = cv2.VideoCapture(0)
    state = preview_states[mode]
    canvas.delete(state["box_id"])
    canvas.delete(state["text_id"])
    if not cap.isOpened():
        messagebox.showerror("Error", "Cannot open camera!")
        return
    
    is_camera_active = True
    
    # Enable capture button, disable camera button
    state["capture_btn_state"]("enabled")
    state["camera_btn_state"]("disabled")
        
    # Start camera feed
    update_camera_feed(mode)

def update_camera_feed(mode):
    global cap, cam_imgtk, is_camera_active
    
    if not is_camera_active or cap is None or not cap.isOpened():
        return
    
    ret, frame = cap.read()
    if ret:
        # Convert and resize frame for display
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img.thumbnail((330, 235))  # Fit preview box
        cam_imgtk = ImageTk.PhotoImage(img)
        
        # Update preview image
        canvas.coords(photo_label_id, 390, 220)
        canvas.itemconfig(photo_label_id, image=cam_imgtk)
    
    # Continue updating if camera is active
    if is_camera_active:
        window.after(30, lambda: update_camera_feed(mode))

def capture_photo(mode):
    global cap
    
    if cap is None or not cap.isOpened():
        messagebox.showerror("Error", "Camera not available!")
        return
    
    ret, frame = cap.read()
    if ret:
        # Save captured frame
        temp_path = "temp_capture.jpg"
        cv2.imwrite(temp_path, frame)
        
        # Stop camera feed
        stop_camera_feed(mode)
        
        # Load captured image into preview
        try:
            pil_img, tk_img = load_image(temp_path)
            state = preview_states[mode]
            state["img"] = tk_img
            state["img_id"] = canvas.create_image(390, 225, image=tk_img)
            state["path"] = temp_path
            
            canvas.delete(state["box_id"])
            
            # Enable classify button
            state["classify_btn_state"]("enabled")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load captured image: {str(e)}")

def stop_camera_feed(mode):
    global is_camera_active
    
    is_camera_active = False
    close_camera()
    
    state = preview_states[mode]
    
    # Reset button states
    state["capture_btn_state"]("disabled")
    state["camera_btn_state"]("enabled")

def choose_file(mode):
    global classify_btn_state
    path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if not path:
        return
    try:
        pil_img, tk_img = load_image(path)
    except Exception as e:
        messagebox.showerror("Error", str(e))
        return

    state = preview_states[mode]
    state["img"] = tk_img
    state["img_id"] = canvas.create_image(390, 225, image=tk_img)
    state["path"] = path

    canvas.delete(state["box_id"])
    canvas.delete(state["text_id"])
    classify_btn_state("enabled")


def classify(mode):
    global score 
    score = 0
    state = preview_states[mode]

    # clear previous result items and image refs
    for rid in state.get("results", []):
        canvas.delete(rid)
    state["results"].clear()
    # clear stored image PhotoImage refs to avoid memory growth
    state["img_refs"] = []

    # run model (top-3) - keep your existing classify_image
    results = classify_image(state["path"], mode)

    # layout constants (tweak these x/y to fit your UI)
    start_y = 480
    spacing_y = 100          
    img_x = 50              
    label_x = 160            
    score_x = 520

    y = start_y
    for i, (label, score) in enumerate(results, 1):
        # label cleanup for display
        clean_label = label.replace("_", " ").title()

        # --- Reference image (left) ---
        ref_path = os.path.join("Image", f"{label}.png")
        if os.path.exists(ref_path):
            ref_img = Image.open(ref_path).convert("RGB")
            ref_img.thumbnail((90, 90))                # thumbnail size
            ref_tk = ImageTk.PhotoImage(ref_img)
            img_id = canvas.create_image(img_x, y, image=ref_tk, anchor="w")
            # keep PhotoImage alive
            state.setdefault("img_refs", []).append(ref_tk)
            state["results"].append(img_id)
        else:
            # placeholder rectangle if no reference image
            rect_id = canvas.create_rectangle(img_x, y-40, img_x+80, y+40, outline="old lace")
            state["results"].append(rect_id)
        
        # --- Breed name (left column, next to image) ---
        breed_text = f"{i}. {clean_label}"
        breed_id = canvas.create_text(
            label_x, y,
            text=breed_text,
            font=("Verdana", 14),
            fill="old lace",
            anchor="w"
        )
        state["results"].append(breed_id)

        # --- Score (right column, aligned with anchor='e') ---
        score_text = f"{score*100:.2f}%"
        score_id = canvas.create_text(
            score_x, y,
            text=score_text,
            font=("Courier New", 16, "bold"),
            fill="old lace",
            anchor="e"
        )
        state["results"].append(score_id)

        y += spacing_y

    top_label, top_score = results[0]
    clean_top = top_label.replace("_", " ").title()
    confidence = top_score * 100

    if confidence >= 90:
        confidence = 100
        display_text = f"{confidence:.2f}% comfirm this is {clean_top}"
    elif confidence < 50:
        display_text = "This breed is not in training list"
    else:
        display_text = f"{confidence:.2f}% confidence likely is {clean_top}"
        
    text_id = canvas.create_text(
        380, 345,  
        text=display_text,
        font=("Verdana", 14),
        fill="light goldenrod",
        anchor="n"
    )
    state["results"].append(text_id)           
        
def clear_preview(mode):
    stop_camera_feed(mode)
    
    state = preview_states[mode]

    # Remove image (if any)
    if "img_id" in state and state["img_id"]:
        canvas.delete(state["img_id"])
        state["img_id"] = None

    # Remove camera frame from preview (if present)
    global photo_label_id
    if photo_label_id:
        canvas.delete(photo_label_id)
        photo_label_id = None

    # Remove results
    for rid in state["results"]:
        canvas.delete(rid)
    state["results"].clear()

    # Restore placeholder box + text
    state["box_id"] = canvas.create_rectangle(220, 100, 555, 340, outline="old lace")
    state["text_id"] = canvas.create_text(395, 270, text="No image selected.", fill="old lace", font=("Verdana", 16, "bold"))

    # Re-create preview image holder for camera/image
    no_photo = tk.PhotoImage(file="No_image.png")
    photo_label_id = canvas.create_image(390, 200, image=no_photo)
    # Keep reference so image doesn't disappear
    state["img"] = no_photo
    state["img_id"] = photo_label_id

    # Reset state
    state["path"] = None
    state["classify_btn_state"]("disabled")

# ========================================================
# Main
# ========================================================
window = tk.Tk()
window.title("Breed Identification")
window.geometry("600x750")

canvas = tk.Canvas(window, width=800, height=700, bg="old lace")
canvas.pack(fill="both", expand=True)

preview_states = {}  # holds preview & result state for dog and cat

# Start at face auth
show_face_auth()

window.mainloop()
