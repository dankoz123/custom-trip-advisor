from datetime import datetime, timedelta
from crewai import Crew, Process, LLM
from core.config import MODEL
from core.agents import make_researcher, make_optimizer, make_planner
from core.tasks import make_research_task, make_optimization_task, make_planning_task
from core.models import Itinerary


def run_crew(destination: str, start_dt: datetime, end_dt: datetime,
             explore: int, hours_per_day: float) -> tuple:
    """Run the CrewAI pipeline. Returns (itinerary, token_usage)."""
    START_DATE = start_dt.strftime("%Y-%m-%d")
    END_DATE   = end_dt.strftime("%Y-%m-%d")
    date_range = [(start_dt + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(explore)]

    llm               = LLM(model=MODEL, temperature=0.3, max_tokens=16000)
    researcher        = make_researcher(destination, llm)
    optimizer         = make_optimizer(destination, explore, hours_per_day, llm)
    planner           = make_planner(destination, explore, START_DATE, hours_per_day, llm)
    research_task     = make_research_task(researcher, destination, explore, date_range, hours_per_day)
    optimization_task = make_optimization_task(optimizer, research_task, destination, explore, hours_per_day)
    planning_task     = make_planning_task(
        planner, optimization_task, destination,
        START_DATE, END_DATE, explore, date_range, hours_per_day,
    )
    crew   = Crew(
        agents=[researcher, optimizer, planner],
        tasks=[research_task, optimization_task, planning_task],
        process=Process.sequential,
        verbose=False,
    )
    result = crew.kickoff()
    return result.pydantic, result.token_usage


def trim_to_visit_budget(itinerary: Itinerary, hours_per_day: float) -> Itinerary:
    """Remove trailing attractions from any day whose visit time exceeds the budget."""
    max_min = int(hours_per_day * 60)
    for day in itinerary.days:
        while len(day.attractions) > 1:
            if sum(a.duration_min for a in day.attractions) <= max_min:
                break
            day.attractions.pop()
    return itinerary
