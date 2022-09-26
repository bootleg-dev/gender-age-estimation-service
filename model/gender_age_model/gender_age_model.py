from pydantic import BaseModel
import pydantic


class GenderAgeResponse(BaseModel):
    gender:str  = pydantic.Field(default=None, example='Male', description='Predicted gender')
    age: int  = pydantic.Field(default=None, example=25, description='Predicted age')
    face_base64: str = pydantic.Field(default=None, example=None, description='Cropped face in base64 format')

class GenderAgeRequest(BaseModel):
    image_base64: str  = pydantic.Field(default=None, example=None, description='Image in base64 format')
