# schemas.py

from pydantic import BaseModel, Field, AliasChoices
from typing import List

class ResumeAnalysisSchema(BaseModel):
    match_score: int = Field(
        validation_alias=AliasChoices("match_score", "match_percentage"),
        description="Match percentage between 0 and 100 based on job requirements"
    )
    found_skills: List[str] = Field(
        validation_alias=AliasChoices("found_skills", "matching_skills"),
        description="Skills present in both candidate CV and Job Description"
    )
    missing_skills: List[str] = Field(
        description="Critical skills present in Job Description but missing from CV"
    )
    recommendations: List[str] = Field(
        validation_alias=AliasChoices("recommendations", "relevant_experience", "suggestions"),
        default_factory=list,
        description="3-4 actionable tips on how to improve the CV for this specific role"
    )