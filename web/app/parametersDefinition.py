from pydantic import BaseModel
from typing import List

class BasicPC_parameters(BaseModel):
    defaultFeatures: List[str]
    exogeneousFeatures: List[str]
    endogeneousFeatures: List[str]