from modal import App
from .api import api_app
from .main import app as main_app

app = App("video-to-blog-post")

app.include(api_app)
app.include(main_app)
