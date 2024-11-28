from typing import Annotated
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.exceptions import HTTPException
from starlette.responses import FileResponse
import json
import os
import yaml
import time

app = FastAPI()
"""
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
timestr = time.strftime("%Y%m%d-%H%M%S")

@app.post("/file/upload")
def upload_file(file: UploadFile):
    if file.content_type != "application/json":
        raise HTTPException(400, detail="Invalid document type")
    else:
        data = json.loads(file.file.read())
    return {"content": data, "filename": file.filename}


@app.post("/file/uploadndownload")
def upload_n_downloadfile(file: UploadFile):

    if file.content_type != "application/json":
        raise HTTPException(400, detail="Invalid document type")
    else:
        json_data = json.loads(file.file.read())
        new_filename = "{}_{}.yaml".format(os.path.splitext(file.filename)[0], timestr)
        # Write the data to a file
        # Store the saved file
        SAVE_FILE_PATH = os.path.join(UPLOAD_DIR, new_filename)
        with open(SAVE_FILE_PATH, "w") as f:
            yaml.dump(json_data, f)

        # Return as a download
        return FileResponse(
            path=SAVE_FILE_PATH,
            media_type="application/octet-stream",
            filename=new_filename,
        )
"""

app = FastAPI()
@app.post("/upload/")
async def uploadfile(files: list[UploadFile]):
    try:
        for file in files:
            file_path = f"C:/Users/skills/Desktop/4/{file.filename}"
            with open(file_path, "wb") as f:
                f.write(file.file.read())
            return {"message": "File saved successfully"}
    except Exception as e:
        return {"message": e.args}

@app.get("/")
async def main():
    return FileResponse('pages/form.html')

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)