from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, validator


ScheduleSlot = Tuple[str, str, str]


class Course(BaseModel):
    course_code: str = Field(..., description="The unique code for the course, e.g. 'CS101'.")
    title: str
    credits: int
    schedule: List[ScheduleSlot] = Field(default_factory=list)


class CatalogCourse(Course):
    description: str = ""
    prerequisites: List[str] = Field(default_factory=list)
    semesters_offered: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)


class Semester(BaseModel):
    semester_name: str
    courses: List[Course] = Field(default_factory=list)
    max_credits: int = 18

    @validator("courses")
    def check_credit_limit(cls, courses: List[Course], values: Dict[str, int]) -> List[Course]:
        total_credits = sum(course.credits for course in courses)
        if total_credits > values.get("max_credits", 18):
            raise ValueError(
                f"Total credits ({total_credits}) exceed the limit of {values.get('max_credits', 18)}."
            )
        return courses

    @validator("courses")
    def check_schedule_conflicts(cls, courses: List[Course]) -> List[Course]:
        def to_minutes(value: str) -> int:
            hours, minutes = map(int, value.split(":"))
            return hours * 60 + minutes

        for left_index in range(len(courses)):
            left_course = courses[left_index]
            for right_index in range(left_index + 1, len(courses)):
                right_course = courses[right_index]
                for left_day, left_start, left_end in left_course.schedule:
                    left_start_minutes = to_minutes(left_start)
                    left_end_minutes = to_minutes(left_end)
                    for right_day, right_start, right_end in right_course.schedule:
                        if left_day != right_day:
                            continue

                        right_start_minutes = to_minutes(right_start)
                        right_end_minutes = to_minutes(right_end)
                        if left_start_minutes < right_end_minutes and right_start_minutes < left_end_minutes:
                            raise ValueError(
                                f"Schedule conflict between {left_course.course_code} and {right_course.course_code}"
                            )

        return courses


class DegreePlan(BaseModel):
    student_id: str
    completed_courses: List[Course]
    planned_semesters: List[Semester] = Field(default_factory=list)


class ParsedTranscript(BaseModel):
    student_id: str
    completed_courses: List[Course]


class ProgramRequirements(BaseModel):
    program_name: str
    required_courses: List[str]
    elective_count: int
    elective_options: List[str]
    max_credits_per_semester: int = 9


class GoalInterpretation(BaseModel):
    planning_horizon_semesters: int = 2
    preferred_start_term: Optional[str] = None
    target_program: str = "AI minor"
    requested_course_codes: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    reasoning: str = ""


class AgentAction(BaseModel):
    type: Literal["tool", "final"]
    reasoning: str
    tool_name: Optional[str] = None
    tool_args: Dict[str, object] = Field(default_factory=dict)
    final_response: Optional[str] = None
