from pydantic import BaseModel, ConfigDict

class BaseSchema(BaseModel):
    name: str
    is_active: bool = True
    