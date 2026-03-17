from __future__ import annotations

"""
Define your events here
"""

from entities import Day, DayHalf, Event, Room, SkillType

TIME_QUANTUM_MIN = 5
MIN_BREAK_MIN = 15


def _hours(value: float) -> int:
    return int(round(value * 60))


def build_day() -> Day:
    return Day(
        start=9 * 60,
        end=18 * 60 + 30,
        lunch_start=12 * 60,
        lunch_end=13 * 60,
        lunch_earliest_start=11 * 60 + 30,
        lunch_latest_start=13 * 60+60,
        lunch_min_duration=60,
        lunch_max_duration=60,
        lunch_latest_end=15 * 60,
    )


def build_rooms() -> dict[str, Room]:
    return {
        "3522 Seminar 1": Room(id="3522 Seminar 1", capacity=30, zone="confcenter"),
        "3522 Seminar 2": Room(id="3522 Seminar 2", capacity=30, zone="confcenter"),
        "3522 Seminar 3": Room(id="3522 Seminar 3", capacity=30, zone="confcenter"),
        "3522 Seminar 4": Room(id="3522 Seminar 4", capacity=30, zone="confcenter"),
        "3522 Seminar 5": Room(id="3522 Seminar 5", capacity=30, zone="confcenter"),
        "3522 Seminar 6": Room(id="3522 Seminar 6", capacity=30, zone="confcenter"),
        "NHB33 106": Room(id="NHB33 106", capacity=117, zone="entrance_campus"),
        "NHB33 004": Room(id="NHB33 004", capacity=40, zone="entrance_campus"),
        "NHB31 003": Room(id="NHB31 003", capacity=34, zone="entrance_campus"),
        "3630 384": Room(id="3630 384", capacity=65, zone="hpc"),
        "3630 302": Room(id="3630 302", capacity=34, zone="hpc"),
        "3630 002": Room(id="3630 002", capacity=71, zone="hpc"),
        # "56 160 a/b": Room(id="56 160 a/b", capacity=50, zone="hai"),
        # "NHB25 001": Room(id="NHB25 001", capacity=36, zone="close_from_hdc_hpc"),
    }


def build_single_track_event_groups() -> list[list[str]]:
    # Each inner list must stay on one track.
    return [
        ["event_1", "event_2"],
    ]


def build_events() -> dict[str, Event]:
    return {
        "sample_event": Event(
            id="sample_event_id",
            title="Sample event title",
            duration_min=_hours(2),
            min_capacity=20,
            prefer_zone="confcenter",
            skill_type=SkillType.HARD,
            after="after_some_event_id",
        ),

    }
