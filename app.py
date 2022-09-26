import uvicorn
from fastapi import FastAPI
from controller import gender_age_controller
from config.constants import HOST_API
from config.constants import PORT_API
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title='Gender Age Service', redoc_url=None)
origins = ["*"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(gender_age_controller.router)

if __name__ == "__main__":
    uvicorn.run(app, host=HOST_API, port=PORT_API)
