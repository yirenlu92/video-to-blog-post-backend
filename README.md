All the source code is in `src`

`src/deploy.py`: deploy files

`src/api.py`: API routes

`src/main.py`: primary functions deployed on Modal

`src/helper.py`: helper functions for `src/main.py`

`src/r2_utils.py`: utility functions for uploading/downloading from R2

`src/assemblyai_utils.py`: utility functions for calling the AssemblyAI API


To deploy on Modal:

```
cd video-to-blog-post-backend
modal deploy src.deploy
```

To run individual Modal functions, use `modal run`, e.g.:

```
cd video-to-blog-post-backend
modal run src.main::create_video_to_post --video-r2-url https://pub-f1ee73dd9450494a95fae11b75fb5a42.r2.dev/1hr_talk_intro_to_large_language_models.mp4
```
