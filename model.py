from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple
from ortools.sat.python import cp_model
from entities import Event, Room, Day


@dataclass
class Vars:
    start: Dict[str, cp_model.IntVar] # event start time
    end: Dict[str, cp_model.IntVar] # event end time
    assign: Dict[Tuple[str, str], cp_model.BoolVar]  # (event_id, room_id)
    track: Dict[Tuple[str, int], cp_model.BoolVar]  # (event_id, track_id)
    spans_lunch: Dict[str, cp_model.BoolVar]
    lunch_start: cp_model.IntVar | None = None
    lunch_end: cp_model.IntVar | None = None

@dataclass
class Ctx:
    model: cp_model.CpModel
    vars: Vars
    events: Dict[str, Event]
    rooms: Dict[str, Room]
    day: Day
    horizon: int  # day.end - day.start, in minutes from day.start (0..horizon)


def create_lunch_vars(model: cp_model.CpModel, day: Day) -> tuple[cp_model.IntVar, cp_model.IntVar]:
    earliest_start = (
        day.lunch_earliest_start if day.lunch_earliest_start is not None else day.lunch_start
    ) - day.start
    latest_start = (
        day.lunch_latest_start if day.lunch_latest_start is not None else day.lunch_start
    ) - day.start
    min_duration = (
        day.lunch_min_duration
        if day.lunch_min_duration is not None
        else (day.lunch_end - day.lunch_start)
    )
    max_duration = day.lunch_max_duration
    latest_end = (
        day.lunch_latest_end if day.lunch_latest_end is not None else day.end
    ) - day.start

    if earliest_start > latest_start:
        raise ValueError("Lunch earliest start must be <= lunch latest start.")
    if min_duration <= 0:
        raise ValueError("Lunch minimum duration must be positive.")
    if max_duration is not None and max_duration < min_duration:
        raise ValueError("Lunch maximum duration must be >= lunch minimum duration.")
    if earliest_start + min_duration > latest_end:
        raise ValueError("Lunch window does not leave enough space for the minimum duration.")

    lunch_start = model.NewIntVar(earliest_start, latest_start, "lunch_start")
    lunch_end_upper = latest_end if max_duration is None else min(latest_end, latest_start + max_duration)
    lunch_end = model.NewIntVar(earliest_start + min_duration, lunch_end_upper, "lunch_end")

    return lunch_start, lunch_end


PenaltyFn = Callable[[Ctx], List[cp_model.LinearExpr]]
