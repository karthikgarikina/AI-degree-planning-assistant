from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Dict, List, Literal, Optional, TypedDict

import requests
from langgraph.graph import END, StateGraph
from loguru import logger
from pydantic import ValidationError

from config import settings
from models import AgentAction, Course, DegreePlan, GoalInterpretation, ProgramRequirements, Semester
from tools import check_prerequisites, client, get_course_details, get_program_requirements, search_courses


class GraphState(TypedDict):
    goal: str
    degree_plan: DegreePlan
    messages: List[dict]
    step_count: int
    target_terms: List[str]
    target_program: str
    requested_course_codes: List[str]
    requirements: Optional[dict]
    remaining_targets: List[str]
    course_details: Dict[str, dict]
    prerequisite_checks: Dict[str, dict]
    search_results: Dict[str, List[dict]]
    next_action: Optional[dict]
    final_response: str
    finished: bool
    used_fallback: bool
    planning_notes: List[str]


TERM_ORDER = {"Spring": 1, "Summer": 2, "Fall": 3}


def _parse_term(term_name: str) -> tuple[int, int]:
    season, year = term_name.split()
    return int(year), TERM_ORDER.get(season, 99)


def _extract_requested_course_codes(goal: str) -> List[str]:
    return sorted(set(re.findall(r"\b[A-Z]{2,4}\d{3,4}\b", goal.upper())))


def _infer_target_program(goal: str) -> str:
    lowered_goal = goal.lower()
    if "ai minor" in lowered_goal or "artificial intelligence minor" in lowered_goal:
        return "AI minor"
    return "AI minor"


def _sorted_catalog_terms() -> List[str]:
    terms = set()
    for course in client.load_catalog():
        terms.update(course.semesters_offered)
    return sorted(terms, key=_parse_term)


def _next_primary_term(reference: datetime) -> str:
    year = reference.year
    if reference.month <= 5:
        return f"Fall {year}"
    return f"Spring {year + 1}"


def _next_named_term(season: str, reference: datetime) -> str:
    season = season.capitalize()
    year = reference.year
    if season == "Spring":
        return f"Spring {year + 1}"
    if season == "Fall":
        return f"Fall {year}" if reference.month <= 5 else f"Fall {year + 1}"
    return _next_primary_term(reference)


def _infer_goal_target_terms(goal: str) -> List[str]:
    normalized = re.sub(r"\s+", " ", goal.lower()).strip()
    now = datetime.now()

    if "next spring" in normalized:
        return [_next_named_term("Spring", now)]
    if "next fall" in normalized:
        return [_next_named_term("Fall", now)]
    if "next semester" in normalized and "two semesters" not in normalized and "2 semesters" not in normalized:
        first_term = _next_primary_term(now)
        return [first_term]

    if "next two semesters" in normalized or "next 2 semesters" in normalized:
        first_term = _next_primary_term(now)
        year, order = _parse_term(first_term)
        if order == TERM_ORDER["Fall"]:
            second_term = f"Spring {year + 1}"
        else:
            second_term = f"Fall {year}"
        return [first_term, second_term]

    first_term = _next_primary_term(now)
    year, order = _parse_term(first_term)
    second_term = f"Spring {year + 1}" if order == TERM_ORDER["Fall"] else f"Fall {year}"
    return [first_term, second_term]


def _slice_terms_from_start(start_term: Optional[str], count: int) -> List[str]:
    available_terms = _sorted_catalog_terms()
    if not available_terms:
        return []

    count = max(1, min(count, len(available_terms)))
    if start_term and start_term in available_terms:
        start_index = available_terms.index(start_term)
    else:
        start_index = 0
        if start_term:
            requested_year, requested_order = _parse_term(start_term)
            for index, term in enumerate(available_terms):
                year, order = _parse_term(term)
                if (year, order) >= (requested_year, requested_order):
                    start_index = index
                    break

    return available_terms[start_index : start_index + count]


def _build_goal_interpretation_prompt(goal: str) -> str:
    current_date = datetime.now().strftime("%Y-%m-%d")
    return (
        "You are interpreting a student's planning goal for a university degree planner.\n"
        "Return valid JSON with keys: planning_horizon_semesters, preferred_start_term, target_program, "
        "requested_course_codes, constraints, reasoning.\n"
        "Rules:\n"
        "- Infer the number of semesters requested from the goal.\n"
        "- Use semester names exactly like 'Fall 2026' or 'Spring 2027' when the goal implies a starting term.\n"
        "- If a specific program is mentioned, set target_program.\n"
        "- Extract any explicit course codes.\n"
        f"- Today's date is {current_date}.\n"
        f"- Available planning terms are: {json.dumps(_sorted_catalog_terms())}.\n"
        f"Goal: {goal}"
    )


def _call_ollama_for_goal_interpretation(goal: str) -> Optional[GoalInterpretation]:
    try:
        response = requests.post(
            f"{settings.ollama_base_url.rstrip('/')}/api/chat",
            json={
                "model": settings.ollama_model,
                "stream": False,
                "format": "json",
                "messages": [{"role": "system", "content": _build_goal_interpretation_prompt(goal)}],
            },
            timeout=settings.ollama_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        interpretation = GoalInterpretation(**json.loads(payload["message"]["content"]))
        logger.bind(
            planning_horizon_semesters=interpretation.planning_horizon_semesters,
            preferred_start_term=interpretation.preferred_start_term,
            target_program=interpretation.target_program,
            requested_course_codes=interpretation.requested_course_codes,
            constraints=interpretation.constraints,
        ).info("Goal interpreted by LLM")
        return interpretation
    except Exception as exc:
        logger.bind(error=str(exc), provider=settings.llm_provider).warning("LLM goal interpretation failed")
        return None


def _fallback_goal_interpretation(goal: str) -> GoalInterpretation:
    normalized = re.sub(r"\s+", " ", goal.lower()).strip()
    planning_horizon_semesters = 2

    number_match = re.search(r"next\s+(\d+)\s+semesters?", normalized)
    if number_match:
        planning_horizon_semesters = max(1, int(number_match.group(1)))
    elif "next spring" in normalized or "next fall" in normalized or "next semester" in normalized:
        planning_horizon_semesters = 1

    preferred_start_term = None
    if "next spring" in normalized:
        preferred_start_term = _next_named_term("Spring", datetime.now())
    elif "next fall" in normalized:
        preferred_start_term = _next_named_term("Fall", datetime.now())
    elif "next semester" in normalized or "next" in normalized:
        preferred_start_term = _next_primary_term(datetime.now())

    return GoalInterpretation(
        planning_horizon_semesters=planning_horizon_semesters,
        preferred_start_term=preferred_start_term,
        target_program=_infer_target_program(goal),
        requested_course_codes=_extract_requested_course_codes(goal),
        reasoning="Fallback goal interpretation was used because the LLM was unavailable.",
    )


def _interpret_goal(goal: str) -> GoalInterpretation:
    if settings.llm_provider == "ollama":
        interpretation = _call_ollama_for_goal_interpretation(goal)
        if interpretation:
            if not interpretation.requested_course_codes:
                interpretation.requested_course_codes = _extract_requested_course_codes(goal)
            if not interpretation.target_program:
                interpretation.target_program = _infer_target_program(goal)
            return interpretation

    interpretation = _fallback_goal_interpretation(goal)
    logger.bind(
        planning_horizon_semesters=interpretation.planning_horizon_semesters,
        preferred_start_term=interpretation.preferred_start_term,
        target_program=interpretation.target_program,
        requested_course_codes=interpretation.requested_course_codes,
    ).info("Goal interpreted by fallback")
    return interpretation


def _select_target_terms(cached_course_details: Dict[str, dict]) -> List[str]:
    if not cached_course_details:
        return []

    semesters = set()
    for course in cached_course_details.values():
        semesters.update(course.get("semesters_offered", []))
    return sorted(semesters, key=_parse_term)


def _course_priority(course_code: str, requirements: ProgramRequirements) -> tuple[int, int, str]:
    if course_code in requirements.required_courses:
        return (0, requirements.required_courses.index(course_code), course_code)
    return (1, requirements.elective_options.index(course_code), course_code)


def _build_llm_prompt(state: GraphState) -> str:
    requirements = state["requirements"] or {}
    completed_codes = [course.course_code for course in state["degree_plan"].completed_courses]
    summary = {
        "goal": state["goal"],
        "step_count": state["step_count"],
        "completed_courses": completed_codes,
        "requested_course_codes": state["requested_course_codes"],
        "requirements_loaded": bool(requirements),
        "remaining_targets": state["remaining_targets"],
        "known_course_details": sorted(state["course_details"].keys()),
        "checked_prerequisites": sorted(state["prerequisite_checks"].keys()),
        "searched": sorted(state["search_results"].keys()),
        "target_terms": state["target_terms"],
    }
    return (
        "You are a degree-planning agent. Decide the next single action.\n"
        "Available tools: get_program_requirements, get_course_details, check_prerequisites, search_courses.\n"
        "Return valid JSON with keys: type, reasoning, tool_name, tool_args, final_response.\n"
        "Use type='tool' to call one tool. Use type='final' only when enough information exists to build a plan.\n"
        f"State: {json.dumps(summary)}"
    )


def _call_ollama_for_action(state: GraphState) -> Optional[AgentAction]:
    try:
        response = requests.post(
            f"{settings.ollama_base_url.rstrip('/')}/api/chat",
            json={
                "model": settings.ollama_model,
                "stream": False,
                "format": "json",
                "messages": [{"role": "system", "content": _build_llm_prompt(state)}],
            },
            timeout=settings.ollama_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        return AgentAction(**json.loads(payload["message"]["content"]))
    except Exception as exc:
        logger.bind(error=str(exc), provider=settings.llm_provider).warning("LLM action selection failed")
        return None


def _derive_remaining_targets(state: GraphState) -> List[str]:
    requirements_payload = state["requirements"] or {}
    requirements = ProgramRequirements(**requirements_payload) if requirements_payload else None
    if not requirements:
        return []

    completed_codes = {course.course_code for course in state["degree_plan"].completed_courses}
    remaining = [code for code in requirements.required_courses if code not in completed_codes]

    completed_and_required = completed_codes | set(remaining)
    elective_candidates = [code for code in requirements.elective_options if code not in completed_and_required]
    requested = [code for code in state["requested_course_codes"] if code in requirements.elective_options]
    if requested:
        elective_candidates = requested + [code for code in elective_candidates if code not in requested]

    if requirements.elective_count > 0:
        remaining.extend(elective_candidates[: requirements.elective_count])

    return sorted(set(remaining), key=lambda code: _course_priority(code, requirements))


def _rule_based_action(state: GraphState) -> AgentAction:
    if not state["requirements"]:
        return AgentAction(
            type="tool",
            reasoning="Program requirements have not been loaded yet.",
            tool_name="get_program_requirements",
            tool_args={"program_name": state["target_program"]},
        )

    if not state["remaining_targets"]:
        return AgentAction(
            type="final",
            reasoning="All required planning inputs have been collected.",
            final_response="Collected program requirements and course data; ready to build the final degree plan.",
        )

    for course_code in state["remaining_targets"]:
        if course_code not in state["course_details"]:
            for semester in state["target_terms"]:
                cache_key = f"{course_code}:{semester}"
                if cache_key not in state["search_results"]:
                    return AgentAction(
                        type="tool",
                        reasoning=f"Need semester-specific availability for {course_code} in {semester}.",
                        tool_name="search_courses",
                        tool_args={"query": course_code, "semester": semester},
                    )
            return AgentAction(
                type="tool",
                reasoning=f"Need direct catalog details for {course_code} after empty search results.",
                tool_name="get_course_details",
                tool_args={"course_code": course_code},
            )

    completed_codes = [course.course_code for course in state["degree_plan"].completed_courses]
    for course_code in state["remaining_targets"]:
        if course_code not in state["prerequisite_checks"]:
            return AgentAction(
                type="tool",
                reasoning=f"Need to verify prerequisites for {course_code}.",
                tool_name="check_prerequisites",
                tool_args={"course_code": course_code, "completed_courses": completed_codes},
            )

    return AgentAction(
        type="final",
        reasoning="Course details, prerequisites, and search results are available.",
        final_response="Sufficient tool observations collected; proceeding to construct the degree plan.",
    )


def _resolve_next_action(state: GraphState) -> AgentAction:
    if settings.llm_provider == "ollama" and not state["used_fallback"]:
        llm_action = _call_ollama_for_action(state)
        if llm_action:
            return llm_action
        state["used_fallback"] = True
    return _rule_based_action(state)


def _course_from_payload(payload: dict) -> Course:
    return Course(
        course_code=payload["course_code"],
        title=payload["title"],
        credits=payload["credits"],
        schedule=payload.get("schedule", []),
    )


def _semester_has_course(semester: Semester, course_code: str) -> bool:
    return any(course.course_code == course_code for course in semester.courses)


def _attempt_add_course(semester: Semester, catalog_course: dict) -> Literal["added", "conflict", "credit_limit"]:
    candidate = _course_from_payload(catalog_course)
    trial_courses = [*semester.courses, candidate]
    try:
        validated = Semester(
            semester_name=semester.semester_name,
            max_credits=semester.max_credits,
            courses=trial_courses,
        )
        semester.courses = validated.courses
        return "added"
    except ValidationError as exc:
        error_text = str(exc)
        if "Schedule conflict" in error_text:
            return "conflict"
        return "credit_limit"


def _prerequisites_met_for_term(course_code: str, completed_codes: set[str], course_details: Dict[str, dict]) -> bool:
    prerequisites = course_details[course_code].get("prerequisites", [])
    return all(prerequisite in completed_codes for prerequisite in prerequisites)


def _construct_degree_plan(state: GraphState) -> DegreePlan:
    requirements = ProgramRequirements(**state["requirements"])
    semesters = [
        Semester(semester_name=term, max_credits=requirements.max_credits_per_semester, courses=[])
        for term in state["target_terms"]
    ]
    completed_codes = {course.course_code for course in state["degree_plan"].completed_courses}
    scheduled_codes = set()

    desired_courses = sorted(state["remaining_targets"], key=lambda code: _course_priority(code, requirements))
    for course_code in desired_courses:
        course_details = state["course_details"].get(course_code)
        if not course_details:
            state["planning_notes"].append(f"Skipped {course_code} because no catalog details were available.")
            continue

        for semester in semesters:
            if semester.semester_name not in course_details.get("semesters_offered", []):
                continue
            if not _prerequisites_met_for_term(course_code, completed_codes, state["course_details"]):
                continue
            if _semester_has_course(semester, course_code):
                scheduled_codes.add(course_code)
                break

            add_result = _attempt_add_course(semester, course_details)
            if add_result == "added":
                scheduled_codes.add(course_code)
                break
            if add_result == "conflict":
                logger.bind(
                    semester=semester.semester_name,
                    course_code=course_code,
                    existing_courses=[course.course_code for course in semester.courses],
                ).warning("Potential schedule conflict detected")
            if add_result == "credit_limit":
                logger.bind(
                    semester=semester.semester_name,
                    course_code=course_code,
                    existing_credits=sum(course.credits for course in semester.courses),
                ).warning("Semester credit limit prevented course assignment")

        if course_code in scheduled_codes:
            completed_codes.add(course_code)
        else:
            state["planning_notes"].append(f"Could not place {course_code} in the requested planning window.")

    return DegreePlan(
        student_id=state["degree_plan"].student_id,
        completed_courses=state["degree_plan"].completed_courses,
        planned_semesters=[semester for semester in semesters if semester.courses],
    )


def agent_node(state: GraphState) -> GraphState:
    logger.bind(step=state["step_count"], goal=state["goal"]).info("Executing agent step")

    if state["step_count"] >= settings.agent_max_steps:
        logger.warning("Agent step limit reached")
        state["finished"] = True
        state["final_response"] = "Agent stopped because the configured step limit was reached."
        return state

    if state["requirements"] and not state["remaining_targets"]:
        state["remaining_targets"] = _derive_remaining_targets(state)

    action = _resolve_next_action(state)
    state["step_count"] += 1
    state["next_action"] = action.dict()
    state["messages"].append({"role": "assistant", "content": action.reasoning})

    if action.type == "final":
        state["degree_plan"] = _construct_degree_plan(state)
        state["final_response"] = action.final_response or "Degree plan generated."
        state["finished"] = True

    return state


def tool_executor_node(state: GraphState) -> GraphState:
    action_payload = state.get("next_action") or {}
    action = AgentAction(**action_payload)
    if action.type != "tool" or not action.tool_name:
        return state

    try:
        if action.tool_name == "get_program_requirements":
            result = get_program_requirements(**action.tool_args)
            state["requirements"] = result
        elif action.tool_name == "get_course_details":
            result = get_course_details(**action.tool_args)
            if result:
                state["course_details"][result["course_code"]] = result
            else:
                result = {"error": f"Course not found: {action.tool_args['course_code']}"}
        elif action.tool_name == "check_prerequisites":
            result = check_prerequisites(**action.tool_args)
            state["prerequisite_checks"][result["course_code"]] = result
        elif action.tool_name == "search_courses":
            result = search_courses(**action.tool_args)
            key = f"{action.tool_args['query']}:{action.tool_args['semester']}"
            state["search_results"][key] = result
            exact_match = next((item for item in result if item["course_code"] == action.tool_args["query"]), None)
            if exact_match:
                state["course_details"][exact_match["course_code"]] = exact_match
        else:
            result = {"error": f"Unsupported tool: {action.tool_name}"}
    except Exception as exc:
        result = {"error": str(exc)}
        logger.bind(tool_name=action.tool_name, error=str(exc)).warning("Tool execution failed")

    state["messages"].append({"role": "tool", "name": action.tool_name, "content": json.dumps(result)})
    state["next_action"] = None

    if state["requirements"] and not state["remaining_targets"]:
        state["remaining_targets"] = _derive_remaining_targets(state)

    return state


def should_continue(state: GraphState) -> str:
    if state["finished"]:
        return END
    return "tool_executor"


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("agent", agent_node)
    graph.add_node("tool_executor", tool_executor_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tool_executor": "tool_executor", END: END})
    graph.add_edge("tool_executor", "agent")
    return graph.compile()


def build_initial_state(goal: str, degree_plan: DegreePlan) -> GraphState:
    interpretation = _interpret_goal(goal)
    target_terms = _slice_terms_from_start(
        interpretation.preferred_start_term or _next_primary_term(datetime.now()),
        interpretation.planning_horizon_semesters,
    )

    return GraphState(
        goal=goal,
        degree_plan=degree_plan,
        messages=[],
        step_count=0,
        target_terms=target_terms or _infer_goal_target_terms(goal),
        target_program=interpretation.target_program,
        requested_course_codes=interpretation.requested_course_codes,
        requirements=None,
        remaining_targets=[],
        course_details={},
        prerequisite_checks={},
        search_results={},
        next_action=None,
        final_response="",
        finished=False,
        used_fallback=False,
        planning_notes=[],
    )
