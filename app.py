from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for, session
from pathlib import Path
from ultralytics import YOLO
from deepface import DeepFace
from PIL import Image
import uuid
import numpy as np
import cv2
from datetime import datetime
from deepface.modules.exceptions import FaceNotDetected
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

from database import (
    save_reference_face, get_all_reference_faces, test_connection, 
    save_history, delete_reference_face, get_user_by_email, create_user,
    get_references_with_details
)

app = Flask(__name__)
app.secret_key = "visionx_secret_2025_osama_karachi"  # production pe change karna
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)
REFERENCE_FOLDER = BASE_DIR / "known_faces"
REFERENCE_FOLDER.mkdir(exist_ok=True)

REFERENCE_IMAGE = REFERENCE_FOLDER / "Person.jpg"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "jfif"}
model = YOLO("yolo11n.pt")

# ========== AUTH DECORATOR ==========
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_email' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

# ========== FACE HELPERS (Same as yours) ==========
def enhance_for_distant(frame_path):
    img = cv2.imread(frame_path)
    if img is None:
        return frame_path
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l,a,b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    enhanced = cv2.addWeighted(enhanced, 1.3, cv2.GaussianBlur(enhanced, (0,0), 3), -0.3, 0)
    cv2.imwrite(frame_path, enhanced)
    return frame_path

def get_face_embedding_hybrid(img_path, for_live=False):
    backends = ["yolov8", "retinaface", "opencv"] if for_live else ["retinaface", "yolov8", "opencv"]
    for backend in backends:
        try:
            result = DeepFace.represent(
                img_path=img_path, model_name="Facenet512",
                detector_backend=backend, enforce_detection=True, align=True
            )
            if result and len(result) > 0:
                print(f"✅ Face found with: {backend}")
                return result
        except Exception:
            continue
    return None

def extract_faces_hybrid(img_path, for_live=False):
    backends = ["yolov8", "retinaface", "opencv"] if for_live else ["retinaface", "yolov8", "opencv"]
    for backend in backends:
        try:
            faces = DeepFace.extract_faces(
                img_path=img_path, detector_backend=backend,
                enforce_detection=True, align=True
            )
            if faces:
                return faces
        except:
            continue
    return None

test_connection()

# ========== AUTH ROUTES ==========
@app.route("/login", methods=["GET", "POST"])
def login():
    if 'user_email' in session:
        return redirect(url_for('home'))
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        user = get_user_by_email(email)
        if user and check_password_hash(user["password"], password):
            session['user_email'] = user["email"]
            session['user_name'] = user["name"]
            return redirect(url_for('home'))
        return render_template("login.html", error="Invalid email or password")
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if 'user_email' in session:
        return redirect(url_for('home'))
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        if not name or not email or not password:
            return render_template("register.html", error="All fields required")
        if get_user_by_email(email):
            return render_template("register.html", error="Email already exists")
        hashed = generate_password_hash(password)
        create_user(name, email, hashed)
        return redirect(url_for('login'))
    return render_template("register.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for('login'))

# ========== MAIN ROUTES ==========
@app.route("/")
@login_required
def home():
    return render_template("index.html", user_name=session.get('user_name'), user_email=session.get('user_email'))

@app.route("/detect", methods=["POST"])
@login_required
def detect():
    if "image" not in request.files:
        return jsonify({"error": "Please select an image."}), 400
    image = request.files["image"]
    if not image.filename or "." not in image.filename:
        return jsonify({"error": "Invalid filename."}), 400
    ext = image.filename.rsplit(".", 1)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Use JPG, JPEG, PNG, JFIF, or WEBP."}), 400
    
    image_path = UPLOAD_FOLDER / f"{uuid.uuid4().hex}.{ext}"
    result_path = UPLOAD_FOLDER / f"{uuid.uuid4().hex}.jpg"
    
    try:
        image.save(image_path)
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            im.save(image_path, format="JPEG")
        results = model.predict(source=str(image_path), conf=0.25, imgsz=640, iou=0.45, verbose=False)
        result = results[0]
        objects = []
        for box in result.boxes:
            cid = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            objects.append({
                "name": result.names[cid],
                "confidence": round(conf * 100, 2),
                "box": [round(float(v), 1) for v in box.xyxy[0].tolist()]
            })
        result.save(filename=str(result_path))
        try:
            save_history("image", result_path.name, objects)
        except:
            pass
        return jsonify({"count": len(objects), "objects": objects, "image_url": f"/uploads/{result_path.name}"})
    except Exception as e:
        app.logger.exception("Object detection failed")
        return jsonify({"error": f"Could not process image: {str(e)}"}), 400
    finally:
        image_path.unlink(missing_ok=True)

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/known_faces/<path:filename>")
def known_faces_file(filename):
    return send_from_directory(REFERENCE_FOLDER, filename)

@app.route("/my-references", methods=["GET"])
@login_required
def my_references():
    try:
        user_email = session.get('user_email')
        refs = get_all_reference_faces(user_email)
        data = []
        for r in refs:
            data.append({
                "name": r.get("name", "Unknown"),
                "reason": r.get("reason", ""),
                "image_path": r.get("image_path", ""),
                "created_at": str(r.get("created_at", "")),
                "id": str(r.get("_id", ""))
            })
        return jsonify({"references": data, "count": len(data)})
    except Exception as e:
        print(f"Error in my-references: {e}")
        return jsonify({"references": [], "count": 0, "error": str(e)})

@app.route("/delete-reference/<path:name>", methods=["GET", "DELETE", "POST"])
@login_required
def delete_ref(name):
    try:
        from urllib.parse import unquote
        name = unquote(name).strip()
        user_email = session.get('user_email')
        print(f"🗑 DELETE REQUEST FOR: '{name}' by {user_email}")
        result = delete_reference_face(name, user_email)
        print(f"Deleted count: {result.deleted_count}")
        return jsonify({"message": f"{name} deleted!", "deleted": result.deleted_count})
    except Exception as e:
        app.logger.exception("Delete failed")
        return jsonify({"error": str(e)}), 500

@app.route("/upload-reference", methods=["POST"])
@login_required
def upload_reference():
    if "image" not in request.files:
        return jsonify({"error": "Please select a photo."}), 400
    image = request.files["image"]
    name = request.form.get("name", "").strip()
    reason = request.form.get("reason", "").strip()
    user_email = session.get('user_email')

    if not image.filename:
        return jsonify({"error": "Please select a photo."}), 400
    if not name:
        return jsonify({"error": "Please enter person name."}), 400

    temp_path = UPLOAD_FOLDER / f"{uuid.uuid4().hex}.jpg"
    try:
        image.save(temp_path)
        with Image.open(temp_path) as im:
            im.verify()
        faces = extract_faces_hybrid(str(temp_path), for_live=False)
        if not faces:
            return jsonify({"error": "No face found. Try clearer photo."}), 400
        
        safe_name = "".join(c if c.isalnum() else "_" for c in name)[:30]
        final_ref_path = REFERENCE_FOLDER / f"{safe_name}_{uuid.uuid4().hex[:4]}.jpg"
        with Image.open(temp_path) as im:
            im.convert("RGB").save(final_ref_path, format="JPEG")
            im.convert("RGB").save(REFERENCE_IMAGE, format="JPEG")

        embedding_result = get_face_embedding_hybrid(str(final_ref_path), for_live=False)
        if not embedding_result:
            return jsonify({"error": "Could not create face embedding."}), 400

        embedding = embedding_result[0]["embedding"]
        save_reference_face(name=name, image_path=str(final_ref_path), face_encoding=embedding, reason=reason, user_email=user_email)

        print(f"✅ Saved to DB: {name} | User: {user_email} | Reason: {reason}")
        return jsonify({"message": f"Reference saved for {name}!"})
    except Exception as e:
        app.logger.exception("Reference upload failed")
        return jsonify({"error": f"Could not use this photo: {str(e)}"}), 400
    finally:
        temp_path.unlink(missing_ok=True)

@app.route("/detect-frame", methods=["POST"])
@login_required
def detect_frame():
    if "frame" not in request.files:
        return jsonify({"error": "Camera frame missing."}), 400
    frame = request.files["frame"]
    image_path = UPLOAD_FOLDER / f"{uuid.uuid4().hex}.jpg"
    user_email = session.get('user_email')
    try:
        frame.save(image_path)
        enhance_for_distant(str(image_path))
        results = model.predict(source=str(image_path), conf=0.25, imgsz=1280, verbose=False)
        result = results[0]
        objects = []
        for box in result.boxes:
            cid = int(box.cls[0].item())
            objects.append({
                "name": result.names[cid],
                "confidence": round(float(box.conf[0].item()) * 100, 1),
                "box": [round(float(v), 1) for v in box.xyxy[0].tolist()]
            })

        matched_faces = []
        # USER KE REFERENCES HI LOAD HONGE
        all_refs = get_all_reference_faces(user_email)
        if all_refs:
            try:
                representations = None
                for backend in ["yolov8", "retinaface", "opencv"]:
                    try:
                        representations = DeepFace.represent(
                            img_path=str(image_path), model_name="Facenet512",
                            detector_backend=backend, enforce_detection=False, align=True
                        )
                        if representations:
                            break
                    except:
                        continue

                if representations:
                    for ref in all_refs:
                        ref_emb = np.asarray(ref["face_encoding"], dtype=np.float32)
                        ref_norm = np.linalg.norm(ref_emb)
                        if ref_norm == 0:
                            continue
                        for rep in representations:
                            area = rep.get("facial_area", {})
                            if area.get("w", 0) < 15:
                                continue
                            curr_emb = np.asarray(rep["embedding"], dtype=np.float32)
                            curr_norm = np.linalg.norm(curr_emb)
                            if curr_norm == 0:
                                continue
                            similarity = float(np.dot(ref_emb, curr_emb) / (ref_norm * curr_norm))
                            distance = 1.0 - similarity
                            if distance <= 0.45:
                                if not all(k in area for k in ("x","y","w","h")):
                                    continue
                                matched_faces.append({
                                    "x": int(area["x"]), "y": int(area["y"]),
                                    "w": int(area["w"]), "h": int(area["h"]),
                                    "similarity": round(similarity * 100, 2),
                                    "name": ref.get("name", "Unknown"),
                                    "reason": ref.get("reason", ""),
                                    "distance": round(distance, 3)
                                })
            except FaceNotDetected:
                pass
            except Exception:
                app.logger.exception("Face matching error")

        return jsonify({
            "width": int(result.orig_shape[1]),
            "height": int(result.orig_shape[0]),
            "objects": objects,
            "matched_faces": matched_faces,
            "has_reference": len(all_refs) > 0
        })
    except Exception:
        app.logger.exception("Camera detection failed")
        return jsonify({"error": "Could not process camera frame."}), 500
    finally:
        image_path.unlink(missing_ok=True)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)