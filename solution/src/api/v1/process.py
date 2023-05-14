from http import HTTPStatus

from fastapi import APIRouter, Body, Depends, HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse

from services.get_result import MLModelProcess, get_model_service

router = APIRouter()
empty_response=[]

@router.post(
    path="/",
    summary="Get Result from Models",
    description="Get Result from Models",
)
async def process_data(
    request: Request,
    model_service: MLModelProcess = Depends(
        get_model_service,
    ),
):
    payload = await request.body()
    payload = payload.decode("utf-8")
    #cache empty payload
    if payload == "":
        global empty_response
        if len(empty_response)>0:
            return empty_response
        else:
            try:
                response=await model_service.process_data(payload)
                empty_response=response
                return response
            except Exception as e:
                raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail=str(e))

    try:
        return await model_service.process_data(payload)
    except Exception as e:
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail=str(e))
