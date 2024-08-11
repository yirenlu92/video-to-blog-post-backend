All the source code is in `src`

`src/deploy.py`: deploy files
`src/api.py`: API routes
`src/main.py`: primary functions deployed on Modal
`src/helper.py`: helper functions for `src/main.py`
`src/r2_utils.py`: utility functions for uploading/downloading from R2
`src/assemblyai_utils.py`: utility functions for calling the AssemblyAI API


To deploy on Modal:

```
modal deploy src.deploy
```
