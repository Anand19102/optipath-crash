from typing import List, Optional
from pydantic import BaseModel, Field, model_validator

class Task(BaseModel):
    id: str = Field(..., description="Unique task identifier")
    description: str = Field(default="", description="Task explanation")
    predecessors: List[str] = Field(default_factory=list, description="List of immediate predecessor task IDs")
    normal_time: float = Field(..., gt=0, description="Standard activity duration")
    crash_time: float = Field(..., gt=0, description="Minimum expedited duration")
    crash_cost_per_unit: float = Field(default=0.0, ge=0, description="Marginal cost slope per unit time compressed")

    @model_validator(mode="after")
    def validate_durations(self):
        if self.crash_time > self.normal_time:
            raise ValueError(f"Crash time ({self.crash_time}) cannot exceed normal time ({self.normal_time}) for task {self.id}")
        return self

    @property
    def max_crash(self) -> float:
        return self.normal_time - self.crash_time

class ProjectSchedule(BaseModel):
    project_name: str = "Project"
    time_unit: str = "weeks"
    indirect_cost_per_unit_time: float = 0.0
    tasks: List[Task]
