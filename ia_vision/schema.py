from typing import List, Optional
from pydantic import BaseModel, Field

# Define your desired data structure.
class Table(BaseModel):
    titulo: str = Field(description="Titulo da tabela")
    cenario: List[str] = Field(description="Nome dos cenários")
    cobertura: List[Optional[int]] = Field(description="Quantidade representando a Cobertura")
    percentual_cobertura: List[Optional[float]] = Field(description="Percentual de Cobertura %")