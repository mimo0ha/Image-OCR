import base64
import mimetypes
import json
from mistralai import Mistral, ImageURLChunk
from PIL import Image
import io

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

API_KEY = "RJIqm5OvwoMvLeWrFdv5JBx26tLsSSK7"

client = Mistral(api_key=API_KEY)

app = FastAPI(title="Mistral OCR API")

def process_image(image: Image.Image):
    try:
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        encoded = base64.b64encode(buffered.getvalue()).decode()
        mime_type = "image/jpeg"
        data_url = f"data:{mime_type};base64,{encoded}"

        response = client.ocr.process(
            document=ImageURLChunk(image_url=data_url),
            model="mistral-ocr-latest"
        )

        response_dict = json.loads(response.model_dump_json())
        return response_dict

    except Exception as e:
        return {"error": str(e)}

@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        result = process_image(image)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
