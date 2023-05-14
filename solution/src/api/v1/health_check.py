from http import HTTPStatus

from fastapi import APIRouter
from starlette.responses import JSONResponse

router = APIRouter()


@router.get(
    path="/",
    summary="Create Notification",
    description="Create notification for user id",
)
async def health_check():
    return JSONResponse(
        status_code=HTTPStatus.OK,
        content={"message": "Health Check"},
    )
