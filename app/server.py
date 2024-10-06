# app.py

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import io
import uvicorn

app = FastAPI(title="YOLOv8 Inference API", description="API for YOLOv8 object detection", version="1.0")

# Load the YOLOv8 model at startup
# Ensure 'models/best.pt' is the correct path to your trained model
model = YOLO('app/best.pt')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint to perform object detection on an uploaded image.
    """
    # Validate the uploaded file
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    try:
        # Read the image content
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")  # Ensure image is in RGB format

        # Perform inference using YOLOv8
        results = model(image)

        # Parse results
        detections = results[0].boxes.data.tolist()  # List of detections

        # Prepare response data
        response = []
        for det in detections:
            xmin, ymin, xmax, ymax, confidence, class_id = det
            response.append({
                "bbox": {
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax
                },
                "confidence": confidence,
                "class_id": int(class_id),
                "class_name": model.names[int(class_id)] if model.names else "unknown"
            })

        return JSONResponse(content={"detections": response})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Welcome to the YOLOv8 Inference API. Use the /predict/ endpoint to perform object detection."}

# if __name__ == "__main__":
#     # Run the server with uvicorn
#     uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=True)