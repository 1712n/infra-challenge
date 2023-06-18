from pydantic import BaseModel


class ModelRequest(BaseModel):
    text: str

    class Config:
        schema_extra = {
            "example": {
                "text": "Tis is how true happiness looks like",
            }
        }
