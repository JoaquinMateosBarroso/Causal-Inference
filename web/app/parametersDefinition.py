from pydantic import BaseModel
from typing import List

class Parameters_CausalDiscoveryBase(BaseModel):
    defaultFeatures: List[str]
    exogeneousFeatures: List[str]
    endogeneousFeatures: List[str]

