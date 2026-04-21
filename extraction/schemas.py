from typing import List, Optional
from pydantic import BaseModel, Field


class Entity(BaseModel):
    

    id: str = Field(..., description="Unique identifier for the entity")
    name: str = Field(..., description="Entity name")
    type: str = Field(..., description="Type of entity (e.g., Person, Company, Location)")
    description: str = Field(..., description="Brief description of the entity")

    class Config:
        """Pydantic model config."""
        json_schema_extra = {
            "example": {
                "id": "entity_1",
                "name": "Albert Einstein",
                "type": "Person",
                "description": "Theoretical physicist, developed theory of relativity",
            }
        }


class Relation(BaseModel):
   

    source: str = Field(..., description="Source entity name")
    relation: str = Field(..., description="Type of relation (e.g., knows, works_at, founded)")
    target: str = Field(..., description="Target entity name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0-1")

    class Config:
        
        json_schema_extra = {
            "example": {
                "source": "Albert Einstein",
                "relation": "won_prize",
                "target": "Nobel Prize in Physics",
                "confidence": 0.95,
            }
        }


class ExtractionResult(BaseModel):
   

    entities: List[Entity] = Field(..., description="List of extracted entities")
    relations: List[Relation] = Field(..., description="List of extracted relations")

    class Config:
        """Pydantic model config."""
        json_schema_extra = {
            "example": {
                "entities": [
                    {
                        "id": "entity_1",
                        "name": "Albert Einstein",
                        "type": "Person",
                        "description": "Theoretical physicist",
                    }
                ],
                "relations": [
                    {
                        "source": "Albert Einstein",
                        "relation": "won_prize",
                        "target": "Nobel Prize in Physics",
                        "confidence": 0.95,
                    }
                ],
            }
        }


# Backward compatibility
class KnowledgeGraph(BaseModel):
    

    entities: List[Entity] = Field(..., description="List of extracted entities")
    relations: List[Relation] = Field(..., description="List of extracted relations")
    source_text: Optional[str] = Field(None, description="Source text that was analyzed")

    class Config:
        
        json_schema_extra = {
            "example": {
                "entities": [
                    {
                        "id": "entity_1",
                        "name": "Albert Einstein",
                        "type": "Person",
                        "description": "Theoretical physicist",
                    }
                ],
                "relations": [
                    {
                        "source": "Albert Einstein",
                        "relation": "won_prize",
                        "target": "Nobel Prize in Physics",
                        "confidence": 0.95,
                    }
                ],
            }
        }
