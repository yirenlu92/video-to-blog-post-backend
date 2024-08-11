from modal.functions import FunctionCall
from modal import App, Function, Image, Secret
from pydantic import BaseModel
from modal import asgi_app

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .r2_utils import upload_file_to_r2_streaming


class YouTubeRequest(BaseModel):
    youtube_url: str


api_image = Image.debian_slim().pip_install("boto3")
api_app = App("api", image=api_image)

web_app = FastAPI()

origins = [
    "http://localhost:5173",
    "https://video-to-blog-post-frontend.onrender.com",
    # Add other origins you want to allow
]

web_app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # List of allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)


@api_app.function(secrets=[Secret.from_name("r2-secret")])
@asgi_app()
def fastapi_app():
    return web_app


@web_app.post("/accept")
def accept_create_video_to_post_job(video: UploadFile = File(...)):
    # upload the file to r2
    video_file_public_url = upload_file_to_r2_streaming(video)

    create_video_to_post_f = Function.lookup(
        "video-to-blog-post", "create_video_to_post"
    )

    # Spawn the function with the path to the temporary video file
    call = create_video_to_post_f.spawn(video_file_public_url)
    return {"call_id": call.object_id}


@web_app.get("/result/{call_id}")
async def poll_results(call_id: str):
    function_call = FunctionCall.from_id(call_id)
    try:
        # return as { "result": { "markdown": function_call.get(timeout=0)} }
        result = function_call.get(timeout=0)
        print("result:", result)
        return {"markdown": result}
    except TimeoutError:
        return JSONResponse({"status": "processing"}, status_code=202)
