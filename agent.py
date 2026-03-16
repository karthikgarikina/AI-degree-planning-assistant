from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from models import DegreePlan
from tools import search_courses
from config import settings
from loguru import logger


class GraphState(TypedDict):

    goal: str
    plan: DegreePlan
    step: int


def agent_node(state: GraphState):

    logger.info("Agent reasoning step", step=state["step"])

    if state["step"] > settings.AGENT_MAX_STEPS:

        logger.warning(
            "Agent step limit reached",
            event="Agent step limit reached"
        )

        return state

    courses = search_courses("AI", "Fall")

    if courses:

        state["plan"].planned_semesters = []

    state["step"] += 1

    return state


def should_continue(state: GraphState):

    if state["step"] >= settings.AGENT_MAX_STEPS:
        return END

    return "agent"


def build_graph():

    graph = StateGraph(GraphState)

    graph.add_node("agent", agent_node)

    graph.set_entry_point("agent")

    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"agent": "agent", END: END},
    )

    return graph.compile()