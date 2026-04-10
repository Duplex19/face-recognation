from fastapi import FastAPI
from deepface import DeepFace
import requests
import os
import pickle
import uuid
import logging

app = FastAPI()

EMBED_DIR = "embeddings"

os.makedirs(EMBED_DIR, exist_ok=True)


def download_image(url):

    filename = f"temp_{uuid.uuid4()}.jpg"

    response = requests.get(url)

    with open(filename, "wb") as f:
        f.write(response.content)

    return filename


# =========================
# REGISTER FACE
# =========================
@app.post("/register")
async def register(data: dict):

    user_id = data["user_id"]
    image_url = data["image_url"]

    try:

        temp_path = download_image(image_url)

        try:

            face = DeepFace.extract_faces(
                img_path=temp_path,
                anti_spoofing=True,
                enforce_detection=True
            )

        except Exception:

            os.remove(temp_path)

            return {
                "status": False,
                "message": "Face not detected",
            }

        if not face[0]["is_real"]:

            os.remove(temp_path)

            return {
                "status": False,
                "message": "Fake face detected"
            }

        embedding = DeepFace.represent(
            img_path=temp_path,
            model_name="Facenet",
            enforce_detection=True
        )

        embed_path = f"{EMBED_DIR}/{user_id}.pkl"

        with open(embed_path, "wb") as f:
            pickle.dump(embedding, f)

        os.remove(temp_path)

        return {
            "status": True,
            "message": "Face registered"
        }

    except Exception as e:

        return {
            "status": False,
            "message": str(e)
        }
        
# @app.post("/register")
# async def register(data: dict):

#     user_id = data["user_id"]
#     image_url = data["image_url"]

#     temp_path = download_image(image_url)

#         # Anti Spoof Check
#     face = DeepFace.extract_faces(
#         img_path=temp_path,
#         anti_spoofing=True,
#         enforce_detection=True
#     )
#     logging.info(f"Anti Spoof Result: {face}")
#     try:

#         if not face[0]["is_real"]:

#             os.remove(temp_path)

#             return {
#                 "status": False,
#                 "message": "Fake face detected"
#             }

#         # Generate embedding
#         embedding = DeepFace.represent(
#             img_path=temp_path,
#             model_name="Facenet",
#             enforce_detection=True
#         )

#         embed_path = f"{EMBED_DIR}/{user_id}.pkl"

#         with open(embed_path, "wb") as f:
#             pickle.dump(embedding, f)

#         os.remove(temp_path)

#         return {
#             "status": True,
#             "message": "Face registered"
#         }

#     except Exception as e:

#         return {
#             "status": False,
#             "message": str(e)
#         }


# =========================
# VERIFY FACE
# =========================

@app.post("/verify")
async def verify(data: dict):

    user_id = data["user_id"]
    image_url = data["image_url"]

    embed_path = f"{EMBED_DIR}/{user_id}.pkl"

    if not os.path.exists(embed_path):

        return {
            "status": False,
            "message": "User not registered"
        }

    try:

        temp_path = download_image(image_url)

        # Anti spoof check
        face = DeepFace.extract_faces(
            img_path=temp_path,
            anti_spoofing=True,
            enforce_detection=True
        )

        if not face[0]["is_real"]:

            os.remove(temp_path)

            return {
                "status": False,
                "message": "Fake face detected"
            }

        # Compare wajah
        result = DeepFace.verify(
            img1_path=temp_path,
            img2_path=f"{temp_path}",
            enforce_detection=False
        )

        os.remove(temp_path)

        return {
            "status": True,
            "verified": result["verified"],
            "distance": result["distance"]
        }

    except Exception as e:

        return {
            "status": False,
            "message": str(e)
        }