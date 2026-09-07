from pydantic import BaseModel, ConfigDict

class BaseSchema(BaseModel):
    name: str
    description: str | None = None
    user_id: int | None = None