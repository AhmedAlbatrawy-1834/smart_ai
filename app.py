from fastapi import FastAPI, File, UploadFile, HTTPException
import shutil
import os
import torch
from src import preprocessing
from src import load_model
from src import prediction

app = FastAPI(title="Plant Disease Detection API")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['Apple___alternaria_leaf_spot',
                'Apple___black_rot',
                'Apple___brown_spot',
                'Apple___gray_spot',
                'Apple___healthy',
                'Apple___rust',
                'Apple___scab',
                'Bell_pepper___bacterial_spot',
                'Bell_pepper___healthy',
                'Blueberry___healthy',
                'Cassava___bacterial_blight',
                'Cassava___brown_streak_disease',
                'Cassava___green_mottle',
                'Cassava___healthy',
                'Cassava___mosaic_disease',
                'Cherry___healthy',
                'Cherry___powdery_mildew',
                'Coffee___healthy',
                'Coffee___red_spider_mite',
                'Coffee___rust',
                'Corn___common_rust',
                'Corn___gray_leaf_spot',
                'Corn___healthy',
                'Corn___northern_leaf_blight',
                'Grape___Leaf_blight',
                'Grape___black_measles',
                'Grape___black_rot',
                'Grape___healthy',
                'Grape___leaf_blight',
                'Orange___citrus_greening',
                'Peach___bacterial_spot',
                'Peach___healthy',
                'Potato___bacterial_wilt',
                'Potato___early_blight',
                'Potato___healthy',
                'Potato___late_blight',
                'Potato___leafroll_virus',
                'Potato___mosaic_virus',
                'Potato___nematode',
                'Potato___pests',
                'Potato___phytophthora',
                'Raspberry___healthy',
                'Rice___bacterial_blight',
                'Rice___blast',
                'Rice___brown_spot',
                'Rice___tungro',
                'Rose___healthy',
                'Rose___rust',
                'Rose___slug_sawfly',
                'Soybean___healthy',
                'Squash___powdery_mildew',
                'Strawberry___healthy',
                'Strawberry___leaf_scorch',
                'Sugercane___healthy',
                'Sugercane___mosaic',
                'Sugercane___red_rot',
                'Sugercane___rust',
                'Sugercane___yellow_leaf',
                'Tomato___bacterial_spot',
                'Tomato___early_blight',
                'Tomato___healthy',
                'Tomato___late_blight',
                'Tomato___leaf_curl',
                'Tomato___leaf_mold',
                'Tomato___mosaic_virus',
                'Tomato___septoria_leaf_spot',
                'Tomato___spider_mites',
                'Tomato___target_spot',
                'Watermelon___anthracnose',
                'Watermelon___downy_mildew',
                'Watermelon___healthy',
                'Watermelon___mosaic_virus']


MODEL_PATH = r"models\plant_disease_classifier_state_dict.pth"
model = load_model.train_model(MODEL_PATH, class_names, device)

@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="ERROR not image")

    temp_path = f"img/temp_{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        img_tensor = preprocessing.preprocess(temp_path, device)
        result = prediction.predict(img_tensor, model, class_names)

        return {
            "status": "success",
            "prediction": result
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)