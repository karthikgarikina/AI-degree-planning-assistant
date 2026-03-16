from pydantic import BaseModel, Field, validator
from typing import List, Tuple


class Course(BaseModel):

    course_code: str
    title: str
    credits: int
    schedule: List[Tuple[str, str, str]]


class Semester(BaseModel):

    semester_name: str
    courses: List[Course] = []
    max_credits: int = 18

    @validator("courses")
    def check_credit_limit(cls, courses, values):

        total = sum(c.credits for c in courses)

        if total > values.get("max_credits", 18):
            raise ValueError("Credit limit exceeded")

        return courses

    @validator("courses")
    def check_schedule_conflicts(cls, courses):

        def to_minutes(t):
            h, m = map(int, t.split(":"))
            return h * 60 + m

        for i in range(len(courses)):
            for j in range(i + 1, len(courses)):

                c1 = courses[i]
                c2 = courses[j]

                for d1, s1, e1 in c1.schedule:
                    for d2, s2, e2 in c2.schedule:

                        if d1 != d2:
                            continue

                        start1 = to_minutes(s1)
                        end1 = to_minutes(e1)

                        start2 = to_minutes(s2)
                        end2 = to_minutes(e2)

                        if start1 < end2 and start2 < end1:
                            raise ValueError(
                                f"Schedule conflict between {c1.course_code} and {c2.course_code}"
                            )

        return courses


class DegreePlan(BaseModel):

    student_id: str
    completed_courses: List[Course]
    planned_semesters: List[Semester] = []