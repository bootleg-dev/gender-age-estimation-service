from fastapi import APIRouter
from model.gender_age_model.gender_age_model import GenderAgeRequest, GenderAgeResponse
from service.gender_age_service import GenderAgeService
from service.utils.logger import Logger


router = APIRouter()
ga_service = GenderAgeService()
logger = Logger()


@router.post("/gender-age", response_model=GenderAgeResponse)
async def predict_gender_age(request_model: GenderAgeRequest):
    logger.info("[REQUEST] /gender-age")
    converted_image = ga_service.img_proc_utils.convert_base64_to_image(file=request_model.image_base64)
    result = ga_service.detect_gender_age(img=converted_image)
    logger.info("[RESPONSE] /gender-age")
    return result
