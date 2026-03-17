from __future__ import annotations

import argparse
import importlib.util
import json
from itertools import combinations
from pathlib import Path

from ortools.sat.python import cp_model

from constraints import (
    MIN_OPTIONAL_LUNCH_SPAN_DURATION_MIN,
    after_events_must_be_adjacent,
    assign_events_to_rooms_and_tracks,
    at_most_one_event_per_room_at_a_time,
    at_most_one_event_per_track_at_a_time,
    events_must_match_duration,
    events_must_respect_halfday,
    lunch_must_fit_window,
    make_events_on_a_single_track,
    no_events_during_lunch,
    penalize_spanning_lunch,
    penalize_concurrent_soft_skills,
    penalize_track_compactness,
    penalize_track_usage,
    prefer_single_event_tracks_in_am,
    pin_schedule_to_grid,
    reward_cross_track_attendance_options,
    prefer_requested_zones,
    prefer_soft_sessions_to_overlap_hard_sessions,
    prefer_event_around_lunch,
)
from entities import Day
from model import Ctx, Vars, create_lunch_vars

TRACK_USAGE_WEIGHT = 1_000_000
TRACK_COMPACTNESS_WEIGHT = 1_000
SINGLE_EVENT_TRACK_PM_PENALTY_WEIGHT = 1_000
LUNCH_SPAN_PENALTY_WEIGHT = TRACK_USAGE_WEIGHT // 3
PREFERRED_ZONE_WEIGHT = 5_000
CONCURRENT_SOFT_SKILLS_WEIGHT = 130_000
SOFT_HARD_ALTERNATIVE_WEIGHT = 5_000
SOLVER_TIME_LIMIT_SECONDS = 300.0
ATTENDANCE_OPTION_BONUS_WEIGHT = 25_000
NUM_CPUS = 8

SOFT_CONSTRAINT_WEIGHTS = [
    {
        "name": "Minimize used tracks",
        "weight": TRACK_USAGE_WEIGHT,
        "kind": "penalty",
        "unit": "per used track",
    },
    {
        "name": "Track compactness",
        "weight": TRACK_COMPACTNESS_WEIGHT,
        "kind": "penalty",
        "unit": "per extra empty minute inside a used track",
    },
    {
        "name": "Prefer AM for single-event tracks",
        "weight": SINGLE_EVENT_TRACK_PM_PENALTY_WEIGHT,
        "kind": "penalty",
        "unit": "per single-event track that starts after lunch",
    },
    {
        "name": "Avoid lunch-spanning events",
        "weight": LUNCH_SPAN_PENALTY_WEIGHT,
        "kind": "penalty",
        "unit": "per lunch-spanning event",
    },
    {
        "name": "Prefer requested zones",
        "weight": PREFERRED_ZONE_WEIGHT,
        "kind": "penalty",
        "unit": "per event outside its preferred zone",
    },
    {
        "name": "Avoid concurrent soft skills",
        "weight": CONCURRENT_SOFT_SKILLS_WEIGHT,
        "kind": "penalty",
        "unit": "per overlapping soft-skill pair",
    },
    {
        "name": "Give soft sessions a hard alternative",
        "weight": SOFT_HARD_ALTERNATIVE_WEIGHT,
        "kind": "penalty",
        "unit": "per soft session without an overlapping hard session",
    },
    {
        "name": "Reward cross-track attendance options",
        "weight": ATTENDANCE_OPTION_BONUS_WEIGHT,
        "kind": "bonus",
        "unit": "per attendable cross-track transition",
    },
]


def _load_definitions_module():
    definitions_path = Path(__file__).with_name("def.py")
    spec = importlib.util.spec_from_file_location("agenda_definitions", definitions_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load agenda definitions from {definitions_path}.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AGENDA_DEFINITIONS = _load_definitions_module()
TIME_QUANTUM_MIN = AGENDA_DEFINITIONS.TIME_QUANTUM_MIN
MIN_BREAK_MIN = AGENDA_DEFINITIONS.MIN_BREAK_MIN


def _format_time(total_minutes: int) -> str:
    hours = total_minutes // 60
    minutes = total_minutes % 60
    return f"{hours:02d}:{minutes:02d}"


def _absolute_time(day: Day, relative_minutes: int) -> int:
    return day.start + relative_minutes


def _configured_single_track_event_groups() -> list[list[str]]:
    groups = AGENDA_DEFINITIONS.build_single_track_event_groups()
    return [
        list(dict.fromkeys(str(event_id) for event_id in group))
        for group in groups
    ]


def _lunch_window_bounds(day: Day) -> tuple[int, int, int, int | None, int]:
    earliest_start = day.lunch_earliest_start if day.lunch_earliest_start is not None else day.lunch_start
    latest_start = day.lunch_latest_start if day.lunch_latest_start is not None else day.lunch_start
    min_duration = (
        day.lunch_min_duration
        if day.lunch_min_duration is not None
        else (day.lunch_end - day.lunch_start)
    )
    max_duration = day.lunch_max_duration
    latest_end = day.lunch_latest_end if day.lunch_latest_end is not None else day.end
    return earliest_start, latest_start, min_duration, max_duration, latest_end


def _max_simultaneous_sessions(schedule_events: list[dict[str, object]]) -> int:
    boundaries: list[tuple[int, int]] = []
    for event in schedule_events:
        start = int(event["start_min"])
        end = int(event["end_min"])
        boundaries.append((start, 1))
        boundaries.append((end, -1))

    active = 0
    peak = 0
    for _, delta in sorted(boundaries, key=lambda item: (item[0], item[1])):
        active += delta
        peak = max(peak, active)

    return peak


def _renumber_tracks_by_first_start(scheduled_events: list[dict[str, object]]) -> int:
    first_start_by_track: dict[int, int] = {}
    for event in scheduled_events:
        track_id = int(event["track"])
        start_min = int(event["start_min"])
        if track_id not in first_start_by_track or start_min < first_start_by_track[track_id]:
            first_start_by_track[track_id] = start_min

    ordered_tracks = sorted(first_start_by_track, key=lambda track_id: (first_start_by_track[track_id], track_id))
    track_mapping = {
        old_track_id: new_track_id
        for new_track_id, old_track_id in enumerate(ordered_tracks, start=1)
    }

    for event in scheduled_events:
        event["track"] = track_mapping[int(event["track"])]

    return len(ordered_tracks)


def _events_overlap(left: dict[str, object], right: dict[str, object]) -> bool:
    return int(left["end_min"]) > int(right["start_min"]) and int(right["end_min"]) > int(left["start_min"])


def _distance_to_lunch_window(
    event: dict[str, object],
    lunch_start_abs: int,
    lunch_end_abs: int,
) -> int:
    start_min = int(event["start_min"])
    end_min = int(event["end_min"])

    if end_min <= lunch_start_abs:
        return lunch_start_abs - end_min
    if start_min >= lunch_end_abs:
        return start_min - lunch_end_abs
    return 0


def _track_compactness_gap_minutes(
    scheduled_events: list[dict[str, object]],
    lunch_start_abs: int,
    lunch_end_abs: int,
) -> int:
    lunch_duration = lunch_end_abs - lunch_start_abs
    events_by_track: dict[int, list[dict[str, object]]] = {}
    total_gap = 0

    for event in scheduled_events:
        events_by_track.setdefault(int(event["track"]), []).append(event)

    for track_events in events_by_track.values():
        ordered_events = sorted(track_events, key=lambda item: (int(item["start_min"]), str(item["title"])))
        all_before_lunch = all(int(event["end_min"]) <= lunch_start_abs for event in ordered_events)
        all_after_lunch = all(int(event["start_min"]) >= lunch_end_abs for event in ordered_events)

        if all_before_lunch:
            total_gap += lunch_start_abs - int(ordered_events[-1]["end_min"])
        elif all_after_lunch:
            total_gap += int(ordered_events[0]["start_min"]) - lunch_end_abs

        for left, right in zip(ordered_events, ordered_events[1:]):
            if int(left["end_min"]) <= lunch_start_abs and int(right["start_min"]) >= lunch_end_abs:
                total_gap += max(0, int(right["start_min"]) - int(left["end_min"]) - lunch_duration)
            else:
                total_gap += max(0, int(right["start_min"]) - int(left["end_min"]) - MIN_BREAK_MIN)

    return total_gap


def _count_cross_track_attendance_options(
    scheduled_events: list[dict[str, object]],
    lunch_start_abs: int,
    lunch_end_abs: int,
) -> int:
    options = 0

    for left in scheduled_events:
        for right in scheduled_events:
            if left["id"] == right["id"] or int(left["track"]) == int(right["track"]):
                continue

            can_follow = int(left["end_min"]) + TIME_QUANTUM_MIN <= int(right["start_min"])
            both_morning = int(left["end_min"]) <= lunch_start_abs and int(right["end_min"]) <= lunch_start_abs
            both_afternoon = int(left["start_min"]) >= lunch_end_abs and int(right["start_min"]) >= lunch_end_abs
            if can_follow and (both_morning or both_afternoon):
                options += 1

    return options


def _count_single_event_tracks_after_lunch(
    scheduled_events: list[dict[str, object]],
    lunch_end_abs: int,
) -> int:
    events_by_track: dict[int, list[dict[str, object]]] = {}
    for event in scheduled_events:
        events_by_track.setdefault(int(event["track"]), []).append(event)

    return sum(
        1
        for track_events in events_by_track.values()
        if len(track_events) == 1 and int(track_events[0]["start_min"]) >= lunch_end_abs
    )


def _build_constraint_report(
    ctx: Ctx,
    scheduled_events: list[dict[str, object]],
    lunch_start_abs: int,
    lunch_end_abs: int,
    *,
    status_name: str,
) -> dict[str, object]:
    by_id = {str(event["id"]): event for event in scheduled_events}
    rooms_by_id = ctx.rooms
    hard_constraints: list[dict[str, object]] = []

    def add_hard_check(name: str, satisfied: bool, details: str) -> None:
        hard_constraints.append(
            {
                "name": name,
                "status": "satisfied" if satisfied else "violated",
                "satisfied": satisfied,
                "details": details,
            }
        )

    scheduled_ids = [str(event["id"]) for event in scheduled_events]
    all_event_ids = list(ctx.events.keys())
    add_hard_check(
        "Every accepted event is scheduled exactly once",
        sorted(scheduled_ids) == sorted(all_event_ids),
        f"{len(scheduled_events)}/{len(ctx.events)} events appear in the final schedule",
    )
    add_hard_check(
        "Every event has exactly one room assignment",
        all(str(event["room_id"]) in ctx.rooms for event in scheduled_events),
        f"{sum(1 for event in scheduled_events if str(event['room_id']) in ctx.rooms)}/{len(scheduled_events)} scheduled events have a valid room",
    )
    add_hard_check(
        "Every event has exactly one plotted track assignment",
        all(int(event["track"]) >= 1 for event in scheduled_events),
        f"{sum(1 for event in scheduled_events if int(event['track']) >= 1)}/{len(scheduled_events)} scheduled events have a track",
    )

    room_requirement_failures: list[str] = []
    for event_id, event in ctx.events.items():
        scheduled = by_id[event_id]
        room = rooms_by_id[str(scheduled["room_id"])]
        if room.capacity < event.min_capacity:
            room_requirement_failures.append(event_id)
        if event.must_be_room is not None and str(scheduled["room_id"]) != event.must_be_room:
            room_requirement_failures.append(event_id)
        if event.must_be_zone is not None and room.zone != event.must_be_zone:
            room_requirement_failures.append(event_id)
    add_hard_check(
        "Room capacity and explicit room or zone requirements are respected",
        not room_requirement_failures,
        f"{len(ctx.events) - len(set(room_requirement_failures))}/{len(ctx.events)} events match capacity and hard room or zone constraints",
    )

    grid_ok = all(
        int(event["start_min"]) % TIME_QUANTUM_MIN == 0 and int(event["end_min"]) % TIME_QUANTUM_MIN == 0
        for event in scheduled_events
    ) and lunch_start_abs % TIME_QUANTUM_MIN == 0 and lunch_end_abs % TIME_QUANTUM_MIN == 0
    add_hard_check(
        "Schedule and lunch are aligned to the time grid",
        grid_ok,
        f"Time quantum {TIME_QUANTUM_MIN} minutes",
    )

    earliest_start, latest_start, min_duration, max_duration, latest_end = _lunch_window_bounds(ctx.day)
    lunch_duration = lunch_end_abs - lunch_start_abs
    lunch_ok = (
        earliest_start <= lunch_start_abs <= latest_start
        and lunch_duration >= min_duration
        and (max_duration is None or lunch_duration <= max_duration)
        and lunch_end_abs <= latest_end
    )
    add_hard_check(
        "Lunch stays inside the allowed window",
        lunch_ok,
        f"Lunch {_format_time(lunch_start_abs)} - {_format_time(lunch_end_abs)} ({lunch_duration} min)",
    )

    duration_failures: list[str] = []
    for event_id, event in ctx.events.items():
        scheduled = by_id[event_id]
        actual_duration = int(scheduled["end_min"]) - int(scheduled["start_min"])
        spans_lunch = bool(scheduled["spans_lunch"])
        if not spans_lunch:
            if actual_duration != event.duration_min:
                duration_failures.append(event_id)
            continue

        if (
            not event.is_workshop
            or event.duration_min <= MIN_OPTIONAL_LUNCH_SPAN_DURATION_MIN
            or event.half.value != "any"
            or actual_duration != event.duration_min + lunch_duration
            or int(scheduled["start_min"]) > lunch_start_abs
            or int(scheduled["end_min"]) < lunch_end_abs
        ):
            duration_failures.append(event_id)
    add_hard_check(
        "Event durations and lunch-spanning workshop rules are respected",
        not duration_failures,
        f"{len(ctx.events) - len(set(duration_failures))}/{len(ctx.events)} events satisfy duration rules",
    )

    room_gap_violations = 0
    for left, right in combinations(scheduled_events, 2):
        if str(left["room_id"]) != str(right["room_id"]):
            continue
        left_id = str(left["id"])
        right_id = str(right["id"])
        setup_gap = 0 if (
            ctx.events[left_id].after == right_id or ctx.events[right_id].after == left_id
        ) else MIN_BREAK_MIN
        if not (
            int(left["end_min"]) + setup_gap <= int(right["start_min"])
            or int(right["end_min"]) + setup_gap <= int(left["start_min"])
        ):
            room_gap_violations += 1
    add_hard_check(
        "No two events overlap in the same room and room setup gaps are respected",
        room_gap_violations == 0,
        f"{room_gap_violations} violating room pair(s)",
    )

    track_gap_violations = 0
    for left, right in combinations(scheduled_events, 2):
        if int(left["track"]) != int(right["track"]):
            continue
        ordered_ok = (
            int(left["end_min"]) + MIN_BREAK_MIN <= int(right["start_min"])
            or int(right["end_min"]) + MIN_BREAK_MIN <= int(left["start_min"])
            or (int(left["end_min"]) <= lunch_start_abs and int(right["start_min"]) >= lunch_end_abs)
            or (int(right["end_min"]) <= lunch_start_abs and int(left["start_min"]) >= lunch_end_abs)
        )
        if not ordered_ok:
            track_gap_violations += 1
    add_hard_check(
        "No two events overlap on the same track and minimum track breaks are respected",
        track_gap_violations == 0,
        f"{track_gap_violations} violating track pair(s)",
    )

    lunch_overlap_failures = [
        str(event["id"])
        for event in scheduled_events
        if not bool(event["spans_lunch"])
        and not (
            int(event["end_min"]) <= lunch_start_abs or int(event["start_min"]) >= lunch_end_abs
        )
    ]
    add_hard_check(
        "No event takes place during lunch unless it is modeled as lunch-spanning",
        not lunch_overlap_failures,
        f"{len(scheduled_events) - len(lunch_overlap_failures)}/{len(scheduled_events)} events respect lunch",
    )

    halfday_failures: list[str] = []
    for event_id, event in ctx.events.items():
        scheduled = by_id[event_id]
        if event.half.value == "am" and int(scheduled["end_min"]) > lunch_start_abs:
            halfday_failures.append(event_id)
        if event.half.value == "pm" and int(scheduled["start_min"]) < lunch_end_abs:
            halfday_failures.append(event_id)
    add_hard_check(
        "Halfday requirements are respected",
        not halfday_failures,
        f"{len(ctx.events) - len(set(halfday_failures))}/{len(ctx.events)} events match their AM or PM requirement",
    )

    after_failures: list[str] = []
    for event_id, event in ctx.events.items():
        if event.after is None:
            continue

        current = by_id[event_id]
        previous = by_id[event.after]
        same_room = str(current["room_id"]) == str(previous["room_id"])
        same_track = int(current["track"]) == int(previous["track"])

        immediate = (
            int(current["start_min"]) >= int(previous["end_min"])
            and int(current["start_min"]) <= int(previous["end_min"]) + MIN_BREAK_MIN
        )
        separated_by_lunch = (
            int(previous["end_min"]) <= lunch_start_abs
            and int(current["start_min"]) >= lunch_end_abs
        )

        inserted_immediate = any(
            (
                str(other["room_id"]) == str(previous["room_id"])
                or int(other["track"]) == int(previous["track"])
            )
            and str(other["id"]) not in {event_id, event.after}
            and int(other["start_min"]) >= int(previous["end_min"])
            and int(other["end_min"]) <= int(current["start_min"])
            for other in scheduled_events
        )
        inserted_via_lunch = any(
            (
                str(other["room_id"]) == str(previous["room_id"])
                or int(other["track"]) == int(previous["track"])
            )
            and str(other["id"]) not in {event_id, event.after}
            and (
                (int(other["start_min"]) >= int(previous["end_min"]) and int(other["start_min"]) < lunch_start_abs)
                or (int(other["end_min"]) > lunch_end_abs and int(other["end_min"]) <= int(current["start_min"]))
            )
            for other in scheduled_events
        )

        valid = same_room and same_track and (
            (immediate and not inserted_immediate) or (separated_by_lunch and not inserted_via_lunch)
        )
        if not valid:
            after_failures.append(event_id)

    configured_after = sum(1 for event in ctx.events.values() if event.after is not None)
    add_hard_check(
        "Explicit after dependencies are respected",
        not after_failures,
        f"{configured_after - len(after_failures)}/{configured_after} configured dependency chains are satisfied" if configured_after else "No after dependencies configured",
    )

    lunch_anchor_failures: list[str] = []
    lunch_anchor_count = 0
    for event_id, event in ctx.events.items():
        if not event.prefer_near_lunch:
            continue

        lunch_anchor_count += 1
        scheduled = by_id[event_id]
        same_track_events = [
            other
            for other in scheduled_events
            if str(other["id"]) != event_id
            and int(other["track"]) == int(scheduled["track"])
        ]
        last_before_lunch = int(scheduled["end_min"]) <= lunch_start_abs and all(
            int(other["start_min"]) <= int(scheduled["end_min"]) or int(other["start_min"]) >= lunch_start_abs
            for other in same_track_events
        )
        first_after_lunch = int(scheduled["start_min"]) >= lunch_end_abs and all(
            int(other["end_min"]) <= lunch_end_abs or int(other["end_min"]) >= int(scheduled["start_min"])
            for other in same_track_events
        )
        if not (last_before_lunch or first_after_lunch):
            lunch_anchor_failures.append(event_id)
    add_hard_check(
        "Hard lunch-anchor preferences are respected",
        not lunch_anchor_failures,
        f"{lunch_anchor_count - len(lunch_anchor_failures)}/{lunch_anchor_count} configured lunch-anchor events are satisfied" if lunch_anchor_count else "No hard lunch-anchor events configured",
    )

    single_track_event_groups = _configured_single_track_event_groups()
    satisfied_single_track_groups = sum(
        1
        for group in single_track_event_groups
        if len({int(by_id[event_id]["track"]) for event_id in group}) <= 1
    )
    add_hard_check(
        "Configured single-track event groups share one track",
        satisfied_single_track_groups == len(single_track_event_groups),
        (
            f"{satisfied_single_track_groups}/{len(single_track_event_groups)} configured group(s) share one track"
            if single_track_event_groups
            else "No configured single-track event groups"
        ),
    )

    soft_events = [event for event in scheduled_events if event["skill_type"] == "soft"]
    hard_events = [event for event in scheduled_events if event["skill_type"] == "hard"]
    used_tracks = len({int(event["track"]) for event in scheduled_events})
    soft_overlap_pairs = sum(
        1
        for left, right in combinations(soft_events, 2)
        if _events_overlap(left, right)
    )
    soft_without_hard_alternative = sum(
        1
        for soft_event in soft_events
        if not any(_events_overlap(soft_event, hard_event) for hard_event in hard_events)
    )
    preferred_zone_misses = sum(
        1
        for event_id, event in ctx.events.items()
        if event.prefer_zone is not None and rooms_by_id[str(by_id[event_id]["room_id"])].zone != event.prefer_zone
    )
    spanning_lunch_count = sum(1 for event in scheduled_events if bool(event["spans_lunch"]))
    track_compactness_gap = _track_compactness_gap_minutes(
        scheduled_events,
        lunch_start_abs,
        lunch_end_abs,
    )
    attendance_option_count = _count_cross_track_attendance_options(
        scheduled_events,
        lunch_start_abs,
        lunch_end_abs,
    )
    single_event_tracks_after_lunch = _count_single_event_tracks_after_lunch(
        scheduled_events,
        lunch_end_abs,
    )
    soft_constraints = [
        {
            "name": "Minimize used tracks",
            "kind": "penalty",
            "weight": TRACK_USAGE_WEIGHT,
            "observed_value": used_tracks,
            "details": f"{used_tracks} used track(s)",
        },
        {
            "name": "Track compactness",
            "kind": "penalty",
            "weight": TRACK_COMPACTNESS_WEIGHT,
            "observed_value": track_compactness_gap,
            "details": f"{track_compactness_gap} extra empty minute(s) inside used tracks",
        },
        {
            "name": "Prefer AM for single-event tracks",
            "kind": "penalty",
            "weight": SINGLE_EVENT_TRACK_PM_PENALTY_WEIGHT,
            "observed_value": single_event_tracks_after_lunch,
            "details": f"{single_event_tracks_after_lunch} single-event track(s) start after lunch",
        },
        {
            "name": "Avoid lunch-spanning events",
            "kind": "penalty",
            "weight": LUNCH_SPAN_PENALTY_WEIGHT,
            "observed_value": spanning_lunch_count,
            "details": f"{spanning_lunch_count} lunch-spanning event(s)",
        },
        {
            "name": "Prefer requested zones",
            "kind": "penalty",
            "weight": PREFERRED_ZONE_WEIGHT,
            "observed_value": preferred_zone_misses,
            "details": f"{preferred_zone_misses} preferred-zone miss(es)",
        },
        {
            "name": "Avoid concurrent soft skills",
            "kind": "penalty",
            "weight": CONCURRENT_SOFT_SKILLS_WEIGHT,
            "observed_value": soft_overlap_pairs,
            "details": f"{soft_overlap_pairs} overlapping soft-skill pair(s)",
        },
        {
            "name": "Give soft sessions a hard alternative",
            "kind": "penalty",
            "weight": SOFT_HARD_ALTERNATIVE_WEIGHT,
            "observed_value": soft_without_hard_alternative,
            "details": f"{soft_without_hard_alternative} soft session(s) without an overlapping hard alternative",
        },
        {
            "name": "Reward cross-track attendance options",
            "kind": "bonus",
            "weight": ATTENDANCE_OPTION_BONUS_WEIGHT,
            "observed_value": attendance_option_count,
            "details": f"{attendance_option_count} attendable cross-track transition(s)",
        },
    ]

    return {
        "solver_status": status_name,
        "all_hard_constraints_satisfied": all(item["satisfied"] for item in hard_constraints),
        "hard_constraints": hard_constraints,
        "soft_constraint_weights": SOFT_CONSTRAINT_WEIGHTS,
        "soft_constraints": soft_constraints,
    }


def _render_constraint_report_text(report: dict[str, object]) -> str:
    lines = [
        f"Solver status: {report['solver_status']}",
        f"All hard constraints satisfied: {'yes' if report['all_hard_constraints_satisfied'] else 'no'}",
        "",
        "Hard constraints",
    ]

    for item in report["hard_constraints"]:
        lines.append(f"- {item['name']}: {item['status']} ({item['details']})")

    lines.append("")
    lines.append("Soft constraint weights")
    for item in report["soft_constraint_weights"]:
        lines.append(f"- {item['name']}: {item['weight']} ({item['kind']}, {item['unit']})")

    lines.append("")
    lines.append("Soft constraints")
    for item in report["soft_constraints"]:
        lines.append(
            f"- {item['name']}: {item['observed_value']} observed, weight {item['weight']} ({item['kind']}; {item['details']})"
        )

    lines.append("")
    return "\n".join(lines)


def _build_context() -> Ctx:
    day = AGENDA_DEFINITIONS.build_day()
    events = AGENDA_DEFINITIONS.build_events()
    rooms = AGENDA_DEFINITIONS.build_rooms()
    model = cp_model.CpModel()
    horizon = day.end - day.start

    start_vars = {
        event_id: model.NewIntVar(0, horizon, f"{event_id}_start")
        for event_id in events
    }
    end_vars = {
        event_id: model.NewIntVar(0, horizon, f"{event_id}_end")
        for event_id in events
    }
    assign_vars = {
        (event_id, room_id): model.NewBoolVar(f"{event_id}_in_{room_id}")
        for event_id in events
        for room_id in rooms
    }
    track_ids = range(1, len(rooms) + 1)
    track_vars = {
        (event_id, track_id): model.NewBoolVar(f"{event_id}_on_track_{track_id}")
        for event_id in events
        for track_id in track_ids
    }
    spans_lunch_vars = {
        event_id: model.NewBoolVar(f"{event_id}_spans_lunch")
        for event_id in events
    }
    lunch_start, lunch_end = create_lunch_vars(model, day)

    return Ctx(
        model=model,
        vars=Vars(
            start=start_vars,
            end=end_vars,
            assign=assign_vars,
            track=track_vars,
            spans_lunch=spans_lunch_vars,
            lunch_start=lunch_start,
            lunch_end=lunch_end,
        ),
        events=events,
        rooms=rooms,
        day=day,
        horizon=horizon,
    )


def _apply_constraints(ctx: Ctx) -> list[cp_model.LinearExpr]:
    penalties: list[cp_model.LinearExpr] = []
    single_track_event_groups = _configured_single_track_event_groups()
    for constraint in [
        assign_events_to_rooms_and_tracks(),
        pin_schedule_to_grid(step_min=TIME_QUANTUM_MIN),
        lunch_must_fit_window(),
        events_must_match_duration(),
        at_most_one_event_per_room_at_a_time(setup_gap_min=MIN_BREAK_MIN),
        at_most_one_event_per_track_at_a_time(min_gap_min=MIN_BREAK_MIN),
        no_events_during_lunch(),
        events_must_respect_halfday(),
        after_events_must_be_adjacent(max_gap_min=MIN_BREAK_MIN),
    ]:
        penalties.extend(constraint(ctx))

    for group in single_track_event_groups:
        penalties.extend(make_events_on_a_single_track(group)(ctx))

    for event_id, event in ctx.events.items():
        if event.prefer_near_lunch:
            penalties.extend(prefer_event_around_lunch(event_id)(ctx))

    penalties.extend(penalize_concurrent_soft_skills(weight=CONCURRENT_SOFT_SKILLS_WEIGHT)(ctx))
    penalties.extend(prefer_soft_sessions_to_overlap_hard_sessions(weight=SOFT_HARD_ALTERNATIVE_WEIGHT)(ctx))
    penalties.extend(
        reward_cross_track_attendance_options(
            weight=ATTENDANCE_OPTION_BONUS_WEIGHT,
            transition_gap_min=TIME_QUANTUM_MIN,
        )(ctx)
    )
    penalties.extend(prefer_requested_zones(weight=PREFERRED_ZONE_WEIGHT)(ctx))
    penalties.extend(penalize_spanning_lunch(weight=LUNCH_SPAN_PENALTY_WEIGHT)(ctx))
    penalties.extend(penalize_track_usage(weight=TRACK_USAGE_WEIGHT)(ctx))
    penalties.extend(penalize_track_compactness(weight=TRACK_COMPACTNESS_WEIGHT, min_break_min=MIN_BREAK_MIN)(ctx))
    penalties.extend(
        prefer_single_event_tracks_in_am(weight=SINGLE_EVENT_TRACK_PM_PENALTY_WEIGHT)(ctx)
    )
    return penalties


def solve_agenda() -> dict[str, object]:
    ctx = _build_context()
    penalties = _apply_constraints(ctx)
    if penalties:
        ctx.model.Minimize(sum(penalties))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = SOLVER_TIME_LIMIT_SECONDS
    solver.parameters.num_search_workers = NUM_CPUS

    status = solver.Solve(ctx.model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"Solver did not find a feasible schedule. Status: {solver.StatusName(status)}")

    lunch_start_rel = solver.Value(ctx.vars.lunch_start) if ctx.vars.lunch_start is not None else ctx.day.lunch_start - ctx.day.start
    lunch_end_rel = solver.Value(ctx.vars.lunch_end) if ctx.vars.lunch_end is not None else ctx.day.lunch_end - ctx.day.start
    lunch_start_abs = _absolute_time(ctx.day, lunch_start_rel)
    lunch_end_abs = _absolute_time(ctx.day, lunch_end_rel)

    scheduled_events: list[dict[str, object]] = []
    for event_id, event in ctx.events.items():
        room_id = next(
            room_id
            for room_id in ctx.rooms
            if solver.Value(ctx.vars.assign[event_id, room_id])
        )
        track_id = next(
            track_id
            for track_id in range(1, len(ctx.rooms) + 1)
            if solver.Value(ctx.vars.track[event_id, track_id])
        )
        start_rel = solver.Value(ctx.vars.start[event_id])
        end_rel = solver.Value(ctx.vars.end[event_id])
        start_abs = _absolute_time(ctx.day, start_rel)
        end_abs = _absolute_time(ctx.day, end_rel)
        spans_lunch = bool(solver.Value(ctx.vars.spans_lunch[event_id]))

        scheduled_events.append(
            {
                "id": event_id,
                "title": event.title,
                "room_id": room_id,
                "room_capacity": ctx.rooms[room_id].capacity,
                "start_min": start_abs,
                "end_min": end_abs,
                "start": _format_time(start_abs),
                "end": _format_time(end_abs),
                "duration_min": event.duration_min,
                "skill_type": event.skill_type.value if event.skill_type is not None else None,
                "half": event.half.value,
                "is_workshop": event.is_workshop,
                "spans_lunch": spans_lunch,
                "prefer_near_lunch": event.prefer_near_lunch,
                "after": event.after,
                "after_hidden": event.after_hidden,
                "track": track_id,
            }
        )

    num_tracks = _renumber_tracks_by_first_start(scheduled_events)
    scheduled_events.sort(key=lambda item: (item["track"], item["start_min"], item["title"]))
    status_name = solver.StatusName(status)
    constraint_report = _build_constraint_report(
        ctx,
        scheduled_events,
        lunch_start_abs,
        lunch_end_abs,
        status_name=status_name,
    )

    return {
        "meta": {
            "status": status_name,
            "objective_value": solver.ObjectiveValue(),
            "day_start": _format_time(ctx.day.start),
            "day_end": _format_time(ctx.day.end),
            "time_quantum_min": TIME_QUANTUM_MIN,
            "num_tracks": num_tracks,
            "peak_simultaneous_sessions": _max_simultaneous_sessions(scheduled_events),
        },
        "lunch": {
            "start_min": lunch_start_abs,
            "end_min": lunch_end_abs,
            "start": _format_time(lunch_start_abs),
            "end": _format_time(lunch_end_abs),
            "duration_min": lunch_end_abs - lunch_start_abs,
        },
        "rooms": [
            {
                "id": room.id,
                "capacity": room.capacity,
                "zone": room.zone,
            }
            for room in ctx.rooms.values()
        ],
        "events": scheduled_events,
        "constraint_report": constraint_report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve the conference agenda and write it to JSON.")
    parser.add_argument(
        "--output",
        default="agenda.json",
        help="Path to the output JSON file.",
    )
    parser.add_argument(
        "--report-output",
        default=None,
        help="Optional path to the human-readable constraint report. Defaults to <output>_report.txt.",
    )
    args = parser.parse_args()

    schedule = solve_agenda()
    output_path = Path(args.output)
    output_path.write_text(json.dumps(schedule, indent=2), encoding="utf-8")
    report_path = (
        Path(args.report_output)
        if args.report_output is not None
        else output_path.with_name(f"{output_path.stem}_report.txt")
    )
    report_path.write_text(
        _render_constraint_report_text(schedule["constraint_report"]),
        encoding="utf-8",
    )

    print(f"Wrote {output_path}")
    print(f"Wrote {report_path}")
    print(f"Lunch: {schedule['lunch']['start']} - {schedule['lunch']['end']}")
    print(f"Peak simultaneous sessions: {schedule['meta']['peak_simultaneous_sessions']}")


if __name__ == "__main__":
    main()
