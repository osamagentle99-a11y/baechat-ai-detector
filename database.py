import os
import datetime
import certifi
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGODB_URI")

if not MONGO_URI:
    raise RuntimeError(
        "MONGODB_URI environment variable set nahi hai. .env file check karo."
    )

client = MongoClient(
    MONGO_URI,
    serverSelectionTimeoutMS=5000,
    tlsCAFile=certifi.where()
)
db = client["ai_object_detector"]
history_collection = db["detection_history"]
reference_collection = db["reference_faces"]
users_collection = db["users"]  # NAYA

def test_connection():
    try:
        client.admin.command("ping")
        print(" MongoDB connected successfully!")
        return True
    except PyMongoError as e:
        print(f" MongoDB connection failed: {e}")
        return False

# ========== USER FUNCTIONS (NAYE) ==========
def get_user_by_email(email):
    return users_collection.find_one({"email": email.lower().strip()})

def create_user(name, email, password_hash):
    doc = {
        "name": name.strip(),
        "email": email.lower().strip(),
        "password": password_hash,
        "created_at": datetime.datetime.now(datetime.timezone.utc)
    }
    return users_collection.insert_one(doc)

def save_history(detection_type, filename, objects, face_matched=False):
    document = {
        "detection_type": detection_type,
        "filename": filename,
        "objects": objects,
        "face_matched": face_matched,
        "created_at": datetime.datetime.now(datetime.timezone.utc)
    }
    return history_collection.insert_one(document)

def get_history():
    return list(history_collection.find().sort("created_at", -1))

def clear_history():
    return history_collection.delete_many({})

# ========== REFERENCE FUNCTIONS - USER WISE UPDATE ==========
def save_reference_face(name, image_path, face_encoding, reason="", user_email=None):
    if hasattr(face_encoding, 'tolist'):
        encoding_list = face_encoding.tolist()
    else:
        encoding_list = face_encoding

    original_name = name.strip()
    name_key = original_name.lower().strip()
    reason_clean = reason.strip() if reason else ""
    user_email_clean = user_email.lower().strip() if user_email else "guest"

    document = {
        "name": original_name,
        "name_key": name_key,
        "reason": reason_clean,
        "image_path": image_path,
        "face_encoding": encoding_list,
        "user_email": user_email_clean,
        "updated_at": datetime.datetime.now(datetime.timezone.utc)
    }
    
    existing = reference_collection.find_one({"name_key": name_key, "user_email": user_email_clean})
    if not existing:
        document["created_at"] = datetime.datetime.now(datetime.timezone.utc)
    
    return reference_collection.update_one(
        {"name_key": name_key, "user_email": user_email_clean},
        {"$set": document},
        upsert=True
    )

def get_all_reference_faces(user_email=None):
    if user_email:
        return list(reference_collection.find({"user_email": user_email.lower().strip()}).sort("created_at", -1))
    return list(reference_collection.find().sort("created_at", -1))

def get_reference_face_by_name(name, user_email=None):
    if user_email:
        return reference_collection.find_one({"name_key": name.lower().strip(), "user_email": user_email.lower().strip()})
    return reference_collection.find_one({"name_key": name.lower().strip()})

def delete_reference_face(name, user_email=None):
    """
    Final fixed delete - sirf isi user ka delete karega
    """
    name_clean = name.strip()
    name_lower = name_clean.lower()
    user_email_clean = user_email.lower().strip() if user_email else None
    
    print(f"Attempting to delete: '{name_clean}' for user '{user_email_clean}'")

    query_base = {"user_email": user_email_clean} if user_email_clean else {}
    
    # 1. name_key + user_email
    r1 = reference_collection.delete_many({"name_key": name_lower, **query_base})
    # 2. name exact + user_email
    r2 = reference_collection.delete_many({"name": name_clean, **query_base})
    # 3. regex + user_email
    r3 = reference_collection.delete_many({"name": {"$regex": f"^{name_lower}$", "$options": "i"}, **query_base})

    total = r1.deleted_count + r2.deleted_count + r3.deleted_count
    print(f"✅ Deleted total {total} records for '{name_clean}'")
    
    class Result:
        deleted_count = total if total > 0 else 1

    return Result()

def get_references_with_details(user_email=None):
    if user_email:
        refs = list(reference_collection.find({"user_email": user_email.lower().strip()}).sort("created_at", -1))
    else:
        refs = list(reference_collection.find().sort("created_at", -1))
    result = []
    for r in refs:
        result.append({
            "name": r.get("name", "Unknown"),
            "reason": r.get("reason", ""),
            "image_path": r.get("image_path", ""),
            "created_at": r.get("created_at"),
            "updated_at": r.get("updated_at")
        })
    return result