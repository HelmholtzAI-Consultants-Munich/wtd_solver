from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional

class DayHalf(Enum):
    ANY = "any"
    AM = "am"
    PM = "pm"

@dataclass(frozen=True)
class Room:
    id: str
    capacity: int
    zone: str = "confcenter"

@dataclass(frozen=True)
class Day:
    start: int # in minutes from 0:00, e.g. 9*60 for 9:00
    end: int # in minutes from 0:00, e.g. 17*60 for 17:00
    lunch_start: int # fixed lunch start, or default start when flexible lunch vars are used
    lunch_end: int # fixed lunch end, or default end when flexible lunch vars are used
    lunch_earliest_start: Optional[int] = None # optional earliest flexible lunch start
    lunch_latest_start: Optional[int] = None # optional latest flexible lunch start
    lunch_min_duration: Optional[int] = None # optional minimum flexible lunch duration
    lunch_max_duration: Optional[int] = None # optional maximum flexible lunch duration
    lunch_latest_end: Optional[int] = None # optional latest flexible lunch end

class SkillType(Enum):
    SOFT = "soft"
    HARD = "hard"

@dataclass(frozen=True)
class Event:
    id: str
    title: str
    duration_min: int # including the break
    min_capacity: int = 0 # minimum room size (for us the maximum audience size)
    half: DayHalf = DayHalf.ANY       # AM/PM preference/requirement
    is_workshop: bool = False         # workshop sessions may be eligible to span lunch
    must_be_room: Optional[str] = None  # hard: force room id
    must_be_zone: Optional[str] = None  # hard: force room zone
    prefer_zone: Optional[str] = None   # soft: prefer a room zone when possible
    prefer_near_lunch: bool = False     # 3 hour sessions should coincide with lunch
    skill_type: Optional[SkillType] = None  # e.g. SOFT / HARD
    after: Optional[str] = None  # hard ordering - this event must happen right after 
    # the listed event or potentially with a lunch or break inbetween
    # if it is set, the events should also happen in the same room
    after_hidden: bool = False  # hide the visual "after" arrow while keeping the solver rule
