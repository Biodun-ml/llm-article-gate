from typing import Literal

from pydantic import BaseModel, Field

ArticleType = Literal[
    "destination_guide", "itinerary", "news", "review", "recipe", "personal", "other"
]
PlaceType = Literal[
    "country", "region", "city", "neighbourhood", "landmark", "nature", "venue", "other"
]


class Classification(BaseModel):
    is_bookable: bool
    article_type: ArticleType
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(max_length=300)


class Place(BaseModel):
    name: str = Field(min_length=1)
    type: PlaceType
    country: str | None = None
    evidence: str = Field(min_length=1)


class Activity(BaseModel):
    name: str = Field(min_length=1)
    place: str | None = None
    evidence: str = Field(min_length=1)


class Extraction(BaseModel):
    places: list[Place] = []
    activities: list[Activity] = []
