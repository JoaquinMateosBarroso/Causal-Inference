from pydantic import BaseModel
from typing import List

class BasicPC_parameters(BaseModel):
    defaultColumns: List[str]
    exogeneousColumns: List[str]
    endogeneousColumns: List[str]