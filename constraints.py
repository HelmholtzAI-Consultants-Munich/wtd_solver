from __future__ import annotations

from itertools import combinations
from typing import Iterable

from ortools.sat.python import cp_model

from entities import DayHalf, SkillType
from model import PenaltyFn

MIN_OPTIONAL_LUNCH_SPAN_DURATION_MIN = 240


def _track_ids(ctx) -> list[int]:
    return sorted({track_id for _, track_id in ctx.vars.track})


# Model the constraint that an event happens at the conference center.
def happens_at_conference_center(event_id: str, *, hard=True, weight: int = 100):
    def _constraint(ctx):
        conf_rooms = [r for r in ctx.rooms.values() if r.zone == "confcenter"]
        
        in_conf = sum(ctx.vars.assign[event_id, r.id] for r in conf_rooms)

        if hard:
            ctx.model.Add(in_conf == 1)
            return []
        else:
            # penalize if the event is not in the conference center
            return [weight * (1 - in_conf)]
    return _constraint


def lunch_must_fit_window() -> PenaltyFn:
    def _constraint(ctx):
        if ctx.vars.lunch_start is None or ctx.vars.lunch_end is None:
            return []

        earliest_start = (
            ctx.day.lunch_earliest_start
            if ctx.day.lunch_earliest_start is not None
            else ctx.day.lunch_start
        ) - ctx.day.start
        latest_start = (
            ctx.day.lunch_latest_start
            if ctx.day.lunch_latest_start is not None
            else ctx.day.lunch_start
        ) - ctx.day.start
        min_duration = (
            ctx.day.lunch_min_duration
            if ctx.day.lunch_min_duration is not None
            else (ctx.day.lunch_end - ctx.day.lunch_start)
        )
        max_duration = ctx.day.lunch_max_duration
        latest_end = (
            ctx.day.lunch_latest_end
            if ctx.day.lunch_latest_end is not None
            else ctx.day.end
        ) - ctx.day.start

        ctx.model.Add(ctx.vars.lunch_start >= earliest_start)
        ctx.model.Add(ctx.vars.lunch_start <= latest_start)
        ctx.model.Add(ctx.vars.lunch_end >= ctx.vars.lunch_start + min_duration)
        if max_duration is not None:
            ctx.model.Add(ctx.vars.lunch_end <= ctx.vars.lunch_start + max_duration)
        ctx.model.Add(ctx.vars.lunch_end <= latest_end)

        return []

    return _constraint


def assign_events_to_rooms_and_tracks() -> PenaltyFn:
    def _constraint(ctx):
        track_ids = _track_ids(ctx)

        for event_id, event in ctx.events.items():
            ctx.model.AddExactlyOne(ctx.vars.assign[event_id, room_id] for room_id in ctx.rooms)
            ctx.model.AddExactlyOne(ctx.vars.track[event_id, track_id] for track_id in track_ids)

            if event.must_be_zone is not None and not any(
                room.zone == event.must_be_zone for room in ctx.rooms.values()
            ):
                raise ValueError(f"Unknown room zone {event.must_be_zone!r} for event {event_id!r}.")
            if event.prefer_zone is not None and not any(
                room.zone == event.prefer_zone for room in ctx.rooms.values()
            ):
                raise ValueError(f"Unknown preferred room zone {event.prefer_zone!r} for event {event_id!r}.")

            for room_id, room in ctx.rooms.items():
                if room.capacity < event.min_capacity:
                    ctx.model.Add(ctx.vars.assign[event_id, room_id] == 0)

                if event.must_be_room is not None and room_id != event.must_be_room:
                    ctx.model.Add(ctx.vars.assign[event_id, room_id] == 0)

                if event.must_be_zone is not None and room.zone != event.must_be_zone:
                    ctx.model.Add(ctx.vars.assign[event_id, room_id] == 0)

            if event.must_be_room is not None:
                if event.must_be_room not in ctx.rooms:
                    raise ValueError(f"Unknown room {event.must_be_room!r} for event {event_id!r}.")

                if (
                    event.must_be_zone is not None
                    and ctx.rooms[event.must_be_room].zone != event.must_be_zone
                ):
                    raise ValueError(
                        f"Room {event.must_be_room!r} is not in zone {event.must_be_zone!r} for event {event_id!r}."
                    )

                ctx.model.Add(ctx.vars.assign[event_id, event.must_be_room] == 1)

        return []

    return _constraint


def pin_schedule_to_grid(*, step_min: int = 5) -> PenaltyFn:
    def _constraint(ctx):
        def pin_to_grid(var: cp_model.IntVar, name: str) -> None:
            remainder = ctx.model.NewIntVar(0, step_min - 1, f"{name}_mod_{step_min}")
            ctx.model.AddModuloEquality(remainder, var, step_min)
            ctx.model.Add(remainder == 0)

        for event_id in ctx.events:
            pin_to_grid(ctx.vars.start[event_id], f"{event_id}_start")
            pin_to_grid(ctx.vars.end[event_id], f"{event_id}_end")

        if ctx.vars.lunch_start is not None:
            pin_to_grid(ctx.vars.lunch_start, "lunch_start")
        if ctx.vars.lunch_end is not None:
            pin_to_grid(ctx.vars.lunch_end, "lunch_end")

        return []

    return _constraint


def prefer_requested_zones(*, weight: int = 1_000) -> PenaltyFn:
    def _constraint(ctx):
        penalties: list[cp_model.LinearExpr] = []

        for event_id, event in ctx.events.items():
            if event.prefer_zone is None:
                continue

            in_preferred_zone = sum(
                ctx.vars.assign[event_id, room_id]
                for room_id, room in ctx.rooms.items()
                if room.zone == event.prefer_zone
            )
            penalties.append(weight * (1 - in_preferred_zone))

        return penalties

    return _constraint


def penalize_track_usage(*, weight: int = 1_000_000) -> PenaltyFn:
    def _constraint(ctx):
        penalties: list[cp_model.LinearExpr] = []
        track_ids = _track_ids(ctx)
        track_used: dict[int, cp_model.BoolVar] = {}

        for track_id in track_ids:
            used = ctx.model.NewBoolVar(f"track_{track_id}_used")
            ctx.model.AddMaxEquality(
                used,
                [ctx.vars.track[event_id, track_id] for event_id in ctx.events],
            )
            penalties.append(weight * used)
            track_used[track_id] = used

        for track_id in track_ids[1:]:
            ctx.model.Add(track_used[track_id] <= track_used[track_id - 1])

        return penalties

    return _constraint


def penalize_track_compactness(*, weight: int = 1_000, min_break_min: int = 15) -> PenaltyFn:
    def _constraint(ctx):
        penalties: list[cp_model.LinearExpr] = []
        track_ids = _track_ids(ctx)
        lunch_start, lunch_end = _lunch_window_relative(ctx)
        lunch_duration = lunch_end - lunch_start
        event_ids = list(ctx.events.keys())

        ends_before_lunch: dict[str, cp_model.BoolVar] = {}
        starts_after_lunch: dict[str, cp_model.BoolVar] = {}
        for event_id in event_ids:
            ends_before = ctx.model.NewBoolVar(f"{event_id}_ends_before_lunch_for_track_gaps")
            starts_after = ctx.model.NewBoolVar(f"{event_id}_starts_after_lunch_for_track_gaps")
            ctx.model.Add(ctx.vars.end[event_id] <= lunch_start).OnlyEnforceIf(ends_before)
            ctx.model.Add(ctx.vars.end[event_id] > lunch_start).OnlyEnforceIf(ends_before.Not())
            ctx.model.Add(ctx.vars.start[event_id] >= lunch_end).OnlyEnforceIf(starts_after)
            ctx.model.Add(ctx.vars.start[event_id] < lunch_end).OnlyEnforceIf(starts_after.Not())
            ends_before_lunch[event_id] = ends_before
            starts_after_lunch[event_id] = starts_after

        for track_id in track_ids:
            track_used = ctx.model.NewBoolVar(f"track_{track_id}_used_for_compactness")
            ctx.model.AddMaxEquality(
                track_used,
                [ctx.vars.track[event_id, track_id] for event_id in event_ids],
            )

            starts_track: dict[str, cp_model.BoolVar] = {
                event_id: ctx.model.NewBoolVar(f"{event_id}_starts_track_{track_id}")
                for event_id in event_ids
            }
            ends_track: dict[str, cp_model.BoolVar] = {
                event_id: ctx.model.NewBoolVar(f"{event_id}_ends_track_{track_id}")
                for event_id in event_ids
            }
            successors: dict[tuple[str, str], cp_model.BoolVar] = {
                (left_id, right_id): ctx.model.NewBoolVar(f"{left_id}_before_{right_id}_on_track_{track_id}")
                for left_id in event_ids
                for right_id in event_ids
                if left_id != right_id
            }

            for event_id in event_ids:
                incoming = [starts_track[event_id]] + [
                    successors[left_id, event_id]
                    for left_id in event_ids
                    if left_id != event_id
                ]
                outgoing = [ends_track[event_id]] + [
                    successors[event_id, right_id]
                    for right_id in event_ids
                    if right_id != event_id
                ]

                ctx.model.Add(sum(incoming) == ctx.vars.track[event_id, track_id])
                ctx.model.Add(sum(outgoing) == ctx.vars.track[event_id, track_id])

            ctx.model.Add(sum(starts_track.values()) == track_used)
            ctx.model.Add(sum(ends_track.values()) == track_used)

            # If a track only has AM sessions, pull its last session up to lunch.
            # If it only has PM sessions, pull its first session back to lunch.
            for event_id in event_ids:
                last_before_lunch = ctx.model.NewBoolVar(
                    f"{event_id}_last_before_lunch_on_track_{track_id}"
                )
                first_after_lunch = ctx.model.NewBoolVar(
                    f"{event_id}_first_after_lunch_on_track_{track_id}"
                )
                boundary_gap = ctx.model.NewIntVar(
                    0,
                    ctx.horizon,
                    f"{event_id}_boundary_gap_on_track_{track_id}",
                )

                ctx.model.AddImplication(last_before_lunch, ends_track[event_id])
                ctx.model.AddImplication(last_before_lunch, ends_before_lunch[event_id])
                ctx.model.AddBoolOr(
                    [
                        last_before_lunch,
                        ends_track[event_id].Not(),
                        ends_before_lunch[event_id].Not(),
                    ]
                )

                ctx.model.AddImplication(first_after_lunch, starts_track[event_id])
                ctx.model.AddImplication(first_after_lunch, starts_after_lunch[event_id])
                ctx.model.AddBoolOr(
                    [
                        first_after_lunch,
                        starts_track[event_id].Not(),
                        starts_after_lunch[event_id].Not(),
                    ]
                )

                ctx.model.Add(boundary_gap == lunch_start - ctx.vars.end[event_id]).OnlyEnforceIf(
                    last_before_lunch
                )
                ctx.model.Add(boundary_gap == ctx.vars.start[event_id] - lunch_end).OnlyEnforceIf(
                    first_after_lunch
                )
                ctx.model.Add(boundary_gap == 0).OnlyEnforceIf(
                    [last_before_lunch.Not(), first_after_lunch.Not()]
                )

                penalties.append(weight * boundary_gap)

            for left_id, right_id in successors:
                successor = successors[left_id, right_id]
                crosses_lunch = ctx.model.NewBoolVar(
                    f"{left_id}_to_{right_id}_crosses_lunch_on_track_{track_id}"
                )
                same_side = ctx.model.NewBoolVar(
                    f"{left_id}_to_{right_id}_same_side_on_track_{track_id}"
                )
                gap = ctx.model.NewIntVar(0, ctx.horizon, f"{left_id}_to_{right_id}_gap_on_track_{track_id}")

                ctx.model.Add(ctx.vars.start[right_id] >= ctx.vars.end[left_id]).OnlyEnforceIf(successor)

                ctx.model.AddImplication(crosses_lunch, successor)
                ctx.model.AddImplication(crosses_lunch, ends_before_lunch[left_id])
                ctx.model.AddImplication(crosses_lunch, starts_after_lunch[right_id])
                ctx.model.AddBoolOr(
                    [
                        successor.Not(),
                        ends_before_lunch[left_id].Not(),
                        starts_after_lunch[right_id].Not(),
                        crosses_lunch,
                    ]
                )
                ctx.model.Add(same_side + crosses_lunch == successor)

                ctx.model.Add(
                    gap == ctx.vars.start[right_id] - ctx.vars.end[left_id] - min_break_min
                ).OnlyEnforceIf(same_side)
                ctx.model.Add(
                    gap == ctx.vars.start[right_id] - ctx.vars.end[left_id] - lunch_duration
                ).OnlyEnforceIf(crosses_lunch)
                ctx.model.Add(gap == 0).OnlyEnforceIf(successor.Not())

                penalties.append(weight * gap)

        return penalties

    return _constraint


def prefer_single_event_tracks_in_am(*, weight: int = 1_000) -> PenaltyFn:
    def _constraint(ctx):
        penalties: list[cp_model.LinearExpr] = []
        track_ids = _track_ids(ctx)
        _, lunch_end = _lunch_window_relative(ctx)
        event_ids = list(ctx.events.keys())

        starts_after_lunch: dict[str, cp_model.BoolVar] = {}
        for event_id in event_ids:
            starts_after = ctx.model.NewBoolVar(f"{event_id}_starts_after_lunch_for_single_track_am")
            ctx.model.Add(ctx.vars.start[event_id] >= lunch_end).OnlyEnforceIf(starts_after)
            ctx.model.Add(ctx.vars.start[event_id] < lunch_end).OnlyEnforceIf(starts_after.Not())
            starts_after_lunch[event_id] = starts_after

        for track_id in track_ids:
            track_event_count = ctx.model.NewIntVar(
                0,
                len(event_ids),
                f"track_{track_id}_event_count_for_single_track_am",
            )
            ctx.model.Add(
                track_event_count == sum(ctx.vars.track[event_id, track_id] for event_id in event_ids)
            )

            single_event_track = ctx.model.NewBoolVar(f"track_{track_id}_single_event_for_single_track_am")
            ctx.model.Add(track_event_count == 1).OnlyEnforceIf(single_event_track)
            ctx.model.Add(track_event_count != 1).OnlyEnforceIf(single_event_track.Not())

            single_pm_candidates: list[cp_model.BoolVar] = []
            for event_id in event_ids:
                is_single_pm_event = ctx.model.NewBoolVar(
                    f"{event_id}_single_pm_event_on_track_{track_id}"
                )
                ctx.model.AddImplication(is_single_pm_event, single_event_track)
                ctx.model.AddImplication(is_single_pm_event, ctx.vars.track[event_id, track_id])
                ctx.model.AddImplication(is_single_pm_event, starts_after_lunch[event_id])
                ctx.model.AddBoolOr(
                    [
                        single_event_track.Not(),
                        ctx.vars.track[event_id, track_id].Not(),
                        starts_after_lunch[event_id].Not(),
                        is_single_pm_event,
                    ]
                )
                single_pm_candidates.append(is_single_pm_event)

            single_pm_track = ctx.model.NewBoolVar(
                f"track_{track_id}_single_event_after_lunch_for_single_track_am"
            )
            ctx.model.AddMaxEquality(single_pm_track, single_pm_candidates)
            penalties.append(weight * single_pm_track)

        return penalties

    return _constraint


def at_most_one_event_per_room_at_a_time(*, setup_gap_min: int = 15) -> PenaltyFn:
    def _constraint(ctx):
        for left_id, right_id in combinations(ctx.events.keys(), 2):
            left_before_right = ctx.model.NewBoolVar(f"{left_id}_before_{right_id}")
            right_before_left = ctx.model.NewBoolVar(f"{right_id}_before_{left_id}")
            setup_gap = 0 if (
                ctx.events[left_id].after == right_id or ctx.events[right_id].after == left_id
            ) else setup_gap_min

            ctx.model.Add(ctx.vars.end[left_id] + setup_gap <= ctx.vars.start[right_id]).OnlyEnforceIf(
                left_before_right
            )
            ctx.model.Add(ctx.vars.end[right_id] + setup_gap <= ctx.vars.start[left_id]).OnlyEnforceIf(
                right_before_left
            )

            for room_id in ctx.rooms:
                ctx.model.AddBoolOr(
                    [
                        left_before_right,
                        right_before_left,
                        ctx.vars.assign[left_id, room_id].Not(),
                        ctx.vars.assign[right_id, room_id].Not(),
                    ]
                )

        return []

    return _constraint


def at_most_one_event_per_track_at_a_time(*, min_gap_min: int = 5) -> PenaltyFn:
    def _constraint(ctx):
        track_ids = _track_ids(ctx)
        lunch_start, lunch_end = _lunch_window_relative(ctx)

        for left_id, right_id in combinations(ctx.events.keys(), 2):
            left_before_right = ctx.model.NewBoolVar(f"{left_id}_track_before_{right_id}")
            right_before_left = ctx.model.NewBoolVar(f"{right_id}_track_before_{left_id}")
            left_before_right_via_lunch = ctx.model.NewBoolVar(f"{left_id}_track_before_{right_id}_via_lunch")
            right_before_left_via_lunch = ctx.model.NewBoolVar(f"{right_id}_track_before_{left_id}_via_lunch")

            ctx.model.Add(
                ctx.vars.end[left_id] + min_gap_min <= ctx.vars.start[right_id]
            ).OnlyEnforceIf(left_before_right)
            ctx.model.Add(
                ctx.vars.end[right_id] + min_gap_min <= ctx.vars.start[left_id]
            ).OnlyEnforceIf(right_before_left)
            ctx.model.Add(ctx.vars.end[left_id] <= lunch_start).OnlyEnforceIf(left_before_right_via_lunch)
            ctx.model.Add(ctx.vars.start[right_id] >= lunch_end).OnlyEnforceIf(left_before_right_via_lunch)
            ctx.model.Add(ctx.vars.end[right_id] <= lunch_start).OnlyEnforceIf(right_before_left_via_lunch)
            ctx.model.Add(ctx.vars.start[left_id] >= lunch_end).OnlyEnforceIf(right_before_left_via_lunch)

            for track_id in track_ids:
                ctx.model.AddBoolOr(
                    [
                        left_before_right,
                        right_before_left,
                        left_before_right_via_lunch,
                        right_before_left_via_lunch,
                        ctx.vars.track[left_id, track_id].Not(),
                        ctx.vars.track[right_id, track_id].Not(),
                    ]
                )

        return []

    return _constraint


def no_events_during_lunch() -> PenaltyFn:
    def _constraint(ctx):
        lunch_start, lunch_end = _lunch_window_relative(ctx)

        for event_id in ctx.events:
            before_lunch = ctx.model.NewBoolVar(f"{event_id}_before_lunch")
            after_lunch = ctx.model.NewBoolVar(f"{event_id}_after_lunch")

            ctx.model.AddBoolOr([before_lunch, after_lunch, ctx.vars.spans_lunch[event_id]])
            ctx.model.Add(ctx.vars.end[event_id] <= lunch_start).OnlyEnforceIf(before_lunch)
            ctx.model.Add(ctx.vars.start[event_id] >= lunch_end).OnlyEnforceIf(after_lunch)

        return []

    return _constraint


def penalize_spanning_lunch(*, weight: int = 100) -> PenaltyFn:
    def _constraint(ctx):
        return [weight * ctx.vars.spans_lunch[event_id] for event_id in ctx.events]

    return _constraint


def events_must_match_duration() -> PenaltyFn:
    def _constraint(ctx):
        lunch_start, lunch_end = _lunch_window_relative(ctx)
        lunch_duration = lunch_end - lunch_start

        for event_id, event in ctx.events.items():
            spans_lunch = ctx.vars.spans_lunch[event_id]
            can_span_lunch = event.is_workshop and event.duration_min > MIN_OPTIONAL_LUNCH_SPAN_DURATION_MIN

            if event.half != DayHalf.ANY or not can_span_lunch:
                ctx.model.Add(spans_lunch == 0)
                ctx.model.Add(ctx.vars.end[event_id] == ctx.vars.start[event_id] + event.duration_min)
                continue

            ctx.model.Add(
                ctx.vars.end[event_id] == ctx.vars.start[event_id] + event.duration_min
            ).OnlyEnforceIf(spans_lunch.Not())
            ctx.model.Add(
                ctx.vars.end[event_id] == ctx.vars.start[event_id] + event.duration_min + lunch_duration
            ).OnlyEnforceIf(spans_lunch)
            ctx.model.Add(ctx.vars.start[event_id] <= lunch_start).OnlyEnforceIf(spans_lunch)
            ctx.model.Add(ctx.vars.end[event_id] >= lunch_end).OnlyEnforceIf(spans_lunch)

        return []

    return _constraint


def events_must_respect_halfday() -> PenaltyFn:
    def _constraint(ctx):
        lunch_start, lunch_end = _lunch_window_relative(ctx)

        for event_id, event in ctx.events.items():
            if event.half == DayHalf.AM:
                ctx.model.Add(ctx.vars.end[event_id] <= lunch_start)
            elif event.half == DayHalf.PM:
                ctx.model.Add(ctx.vars.start[event_id] >= lunch_end)

        return []

    return _constraint


def after_events_must_be_adjacent(*, max_gap_min: int = 10) -> PenaltyFn:
    def _constraint(ctx):
        lunch_start, lunch_end = _lunch_window_relative(ctx)
        track_ids = _track_ids(ctx)

        for event_id, event in ctx.events.items():
            if event.after is None:
                continue

            previous_event_id = event.after
            happens_immediately = ctx.model.NewBoolVar(f"{previous_event_id}_immediately_before_{event_id}")
            separated_by_lunch = ctx.model.NewBoolVar(f"{previous_event_id}_before_{event_id}_via_lunch")

            ctx.model.Add(happens_immediately + separated_by_lunch == 1)

            for room_id in ctx.rooms:
                ctx.model.Add(
                    ctx.vars.assign[event_id, room_id] == ctx.vars.assign[previous_event_id, room_id]
                )

            for track_id in track_ids:
                ctx.model.Add(
                    ctx.vars.track[event_id, track_id] == ctx.vars.track[previous_event_id, track_id]
                )

            ctx.model.Add(ctx.vars.start[event_id] >= ctx.vars.end[previous_event_id]).OnlyEnforceIf(
                happens_immediately
            )
            ctx.model.Add(
                ctx.vars.start[event_id] <= ctx.vars.end[previous_event_id] + max_gap_min
            ).OnlyEnforceIf(happens_immediately)

            ctx.model.Add(ctx.vars.end[previous_event_id] <= lunch_start).OnlyEnforceIf(separated_by_lunch)
            ctx.model.Add(ctx.vars.start[event_id] >= lunch_end).OnlyEnforceIf(separated_by_lunch)

            def forbid_insertions_on_shared_resource(
                resource_ids: Iterable[str | int],
                selector,
                *,
                label: str,
            ) -> None:
                for other_event_id in ctx.events:
                    if other_event_id in {event_id, previous_event_id}:
                        continue

                    for resource_id in resource_ids:
                        other_before_previous_end = ctx.model.NewBoolVar(
                            f"{other_event_id}_before_{previous_event_id}_in_{label}_{resource_id}"
                        )
                        other_after_lunch = ctx.model.NewBoolVar(
                            f"{other_event_id}_after_lunch_in_{label}_{resource_id}"
                        )
                        other_before_or_at_lunch_end = ctx.model.NewBoolVar(
                            f"{other_event_id}_before_or_at_lunch_end_in_{label}_{resource_id}"
                        )
                        other_after_event = ctx.model.NewBoolVar(
                            f"{other_event_id}_after_{event_id}_in_{label}_{resource_id}"
                        )

                        ctx.model.Add(
                            ctx.vars.start[other_event_id] <= ctx.vars.end[previous_event_id]
                        ).OnlyEnforceIf(other_before_previous_end)
                        ctx.model.Add(
                            ctx.vars.start[other_event_id] >= ctx.vars.start[event_id]
                        ).OnlyEnforceIf(other_after_event)
                        ctx.model.AddBoolOr(
                            [
                                other_before_previous_end,
                                other_after_event,
                                happens_immediately.Not(),
                                selector(previous_event_id, resource_id).Not(),
                                selector(other_event_id, resource_id).Not(),
                            ]
                        )

                        ctx.model.Add(
                            ctx.vars.start[other_event_id] >= lunch_start
                        ).OnlyEnforceIf(other_after_lunch)
                        ctx.model.AddBoolOr(
                            [
                                other_before_previous_end,
                                other_after_lunch,
                                separated_by_lunch.Not(),
                                selector(previous_event_id, resource_id).Not(),
                                selector(other_event_id, resource_id).Not(),
                            ]
                        )

                        ctx.model.Add(
                            ctx.vars.end[other_event_id] <= lunch_end
                        ).OnlyEnforceIf(other_before_or_at_lunch_end)
                        ctx.model.AddBoolOr(
                            [
                                other_before_or_at_lunch_end,
                                other_after_event,
                                separated_by_lunch.Not(),
                                selector(event_id, resource_id).Not(),
                                selector(other_event_id, resource_id).Not(),
                            ]
                        )

            forbid_insertions_on_shared_resource(
                ctx.rooms.keys(),
                lambda candidate_event_id, room_id: ctx.vars.assign[candidate_event_id, room_id],
                label="room",
            )
            forbid_insertions_on_shared_resource(
                track_ids,
                lambda candidate_event_id, track_id: ctx.vars.track[candidate_event_id, track_id],
                label="track",
            )

        return []

    return _constraint


def _lunch_window_relative(ctx) -> tuple[int | cp_model.IntVar, int | cp_model.IntVar]:
    if ctx.vars.lunch_start is not None and ctx.vars.lunch_end is not None:
        return ctx.vars.lunch_start, ctx.vars.lunch_end

    return (
        ctx.day.lunch_start - ctx.day.start,
        ctx.day.lunch_end - ctx.day.start,
    )


def prefer_event_around_lunch(
    event_id: str,
) -> PenaltyFn:
    def _constraint(ctx):
        lunch_start, lunch_end = _lunch_window_relative(ctx)
        track_ids = _track_ids(ctx)
        last_before_lunch = ctx.model.NewBoolVar(f"{event_id}_last_before_lunch")
        first_after_lunch = ctx.model.NewBoolVar(f"{event_id}_first_after_lunch")

        ctx.model.AddBoolOr([last_before_lunch, first_after_lunch])

        ctx.model.Add(ctx.vars.end[event_id] <= lunch_start).OnlyEnforceIf(last_before_lunch)
        ctx.model.Add(ctx.vars.start[event_id] >= lunch_end).OnlyEnforceIf(first_after_lunch)

        for other_event_id in ctx.events:
            if other_event_id == event_id:
                continue

            same_track_candidates: list[cp_model.BoolVar] = []
            for track_id in track_ids:
                both_on_track = ctx.model.NewBoolVar(
                    f"{event_id}_{other_event_id}_both_on_track_{track_id}_for_lunch_anchor"
                )
                ctx.model.AddImplication(both_on_track, ctx.vars.track[event_id, track_id])
                ctx.model.AddImplication(both_on_track, ctx.vars.track[other_event_id, track_id])
                ctx.model.AddBoolOr(
                    [
                        ctx.vars.track[event_id, track_id].Not(),
                        ctx.vars.track[other_event_id, track_id].Not(),
                        both_on_track,
                    ]
                )
                same_track_candidates.append(both_on_track)

            same_track = ctx.model.NewBoolVar(
                f"{event_id}_{other_event_id}_same_track_for_lunch_anchor"
            )
            ctx.model.AddMaxEquality(same_track, same_track_candidates)

            same_track_and_last_before_lunch = ctx.model.NewBoolVar(
                f"{event_id}_{other_event_id}_same_track_and_last_before_lunch"
            )
            ctx.model.AddImplication(same_track_and_last_before_lunch, same_track)
            ctx.model.AddImplication(same_track_and_last_before_lunch, last_before_lunch)
            ctx.model.AddBoolOr(
                [
                    same_track.Not(),
                    last_before_lunch.Not(),
                    same_track_and_last_before_lunch,
                ]
            )

            other_starts_before_or_at_anchor_end = ctx.model.NewBoolVar(
                f"{other_event_id}_starts_before_{event_id}_for_lunch_anchor"
            )
            other_starts_after_lunch = ctx.model.NewBoolVar(
                f"{other_event_id}_starts_after_lunch_for_{event_id}_lunch_anchor"
            )
            ctx.model.Add(
                ctx.vars.start[other_event_id] <= ctx.vars.end[event_id]
            ).OnlyEnforceIf(other_starts_before_or_at_anchor_end)
            ctx.model.Add(
                ctx.vars.start[other_event_id] > ctx.vars.end[event_id]
            ).OnlyEnforceIf(other_starts_before_or_at_anchor_end.Not())
            ctx.model.Add(
                ctx.vars.start[other_event_id] >= lunch_start
            ).OnlyEnforceIf(other_starts_after_lunch)
            ctx.model.Add(
                ctx.vars.start[other_event_id] < lunch_start
            ).OnlyEnforceIf(other_starts_after_lunch.Not())
            ctx.model.AddBoolOr(
                [
                    other_starts_before_or_at_anchor_end,
                    other_starts_after_lunch,
                    same_track_and_last_before_lunch.Not(),
                ]
            )

            same_track_and_first_after_lunch = ctx.model.NewBoolVar(
                f"{event_id}_{other_event_id}_same_track_and_first_after_lunch"
            )
            ctx.model.AddImplication(same_track_and_first_after_lunch, same_track)
            ctx.model.AddImplication(same_track_and_first_after_lunch, first_after_lunch)
            ctx.model.AddBoolOr(
                [
                    same_track.Not(),
                    first_after_lunch.Not(),
                    same_track_and_first_after_lunch,
                ]
            )

            other_ends_before_or_at_lunch = ctx.model.NewBoolVar(
                f"{other_event_id}_ends_before_lunch_for_{event_id}_lunch_anchor"
            )
            other_ends_after_or_at_anchor_start = ctx.model.NewBoolVar(
                f"{other_event_id}_ends_after_{event_id}_for_lunch_anchor"
            )
            ctx.model.Add(
                ctx.vars.end[other_event_id] <= lunch_end
            ).OnlyEnforceIf(other_ends_before_or_at_lunch)
            ctx.model.Add(
                ctx.vars.end[other_event_id] > lunch_end
            ).OnlyEnforceIf(other_ends_before_or_at_lunch.Not())
            ctx.model.Add(
                ctx.vars.end[other_event_id] >= ctx.vars.start[event_id]
            ).OnlyEnforceIf(other_ends_after_or_at_anchor_start)
            ctx.model.Add(
                ctx.vars.end[other_event_id] < ctx.vars.start[event_id]
            ).OnlyEnforceIf(other_ends_after_or_at_anchor_start.Not())
            ctx.model.AddBoolOr(
                [
                    other_ends_before_or_at_lunch,
                    other_ends_after_or_at_anchor_start,
                    same_track_and_first_after_lunch.Not(),
                ]
            )

        return []

    return _constraint


def minimize_max_simultaneous_sessions(
    event_ids: Iterable[str] | None = None,
    *,
    weight: int = 1,
) -> PenaltyFn:
    """Minimize the peak number of concurrently running sessions.

    This is the exact objective for "use as few simultaneous tracks as
    possible", while still allowing rooms to stay empty.
    """

    def _constraint(ctx):
        selected_event_ids = list(event_ids) if event_ids is not None else list(ctx.events.keys())

        if not selected_event_ids:
            return []

        peak_sessions = ctx.model.NewIntVar(
            0,
            len(selected_event_ids),
            "peak_simultaneous_sessions",
        )

        for anchor_id in selected_event_ids:
            active_at_anchor: list[cp_model.BoolVar] = []
            for event_id in selected_event_ids:
                started = ctx.model.NewBoolVar(f"{event_id}_started_by_{anchor_id}")
                not_finished = ctx.model.NewBoolVar(f"{event_id}_running_at_{anchor_id}")
                active = ctx.model.NewBoolVar(f"{event_id}_active_at_{anchor_id}")

                ctx.model.Add(ctx.vars.start[event_id] <= ctx.vars.start[anchor_id]).OnlyEnforceIf(started)
                ctx.model.Add(ctx.vars.start[event_id] > ctx.vars.start[anchor_id]).OnlyEnforceIf(started.Not())

                ctx.model.Add(ctx.vars.end[event_id] > ctx.vars.start[anchor_id]).OnlyEnforceIf(not_finished)
                ctx.model.Add(ctx.vars.end[event_id] <= ctx.vars.start[anchor_id]).OnlyEnforceIf(not_finished.Not())

                ctx.model.AddImplication(active, started)
                ctx.model.AddImplication(active, not_finished)
                ctx.model.AddBoolOr([started.Not(), not_finished.Not(), active])

                active_at_anchor.append(active)

            ctx.model.Add(sum(active_at_anchor) <= peak_sessions)

        return [weight * peak_sessions]

    return _constraint


def penalize_concurrent_soft_skills(*, weight: int = 100) -> PenaltyFn:
    """Penalize soft-skill sessions scheduled at the same time."""

    def _constraint(ctx):
        soft_event_ids = [
            event_id
            for event_id, event in ctx.events.items()
            if event.skill_type == SkillType.SOFT
        ]

        penalties: list[cp_model.LinearExpr] = []
        for left_id, right_id in combinations(soft_event_ids, 2):
            left_before_right = ctx.model.NewBoolVar(f"{left_id}_before_{right_id}")
            right_before_left = ctx.model.NewBoolVar(f"{right_id}_before_{left_id}")
            overlaps = ctx.model.NewBoolVar(f"{left_id}_{right_id}_soft_overlap")

            ctx.model.Add(ctx.vars.end[left_id] <= ctx.vars.start[right_id]).OnlyEnforceIf(left_before_right)
            ctx.model.Add(ctx.vars.end[left_id] > ctx.vars.start[right_id]).OnlyEnforceIf(left_before_right.Not())

            ctx.model.Add(ctx.vars.end[right_id] <= ctx.vars.start[left_id]).OnlyEnforceIf(right_before_left)
            ctx.model.Add(ctx.vars.end[right_id] > ctx.vars.start[left_id]).OnlyEnforceIf(right_before_left.Not())

            ctx.model.AddBoolOr([left_before_right, right_before_left, overlaps])
            ctx.model.AddImplication(left_before_right, overlaps.Not())
            ctx.model.AddImplication(right_before_left, overlaps.Not())

            penalties.append(weight * overlaps)

        return penalties

    return _constraint


def make_events_on_a_single_track(event_ids: Iterable[str]) -> PenaltyFn:
    """Force all listed events to share the same track."""

    normalized_event_ids = tuple(dict.fromkeys(str(event_id) for event_id in event_ids))

    def _constraint(ctx):
        if len(normalized_event_ids) < 2:
            return []

        missing_event_ids = [
            event_id
            for event_id in normalized_event_ids
            if event_id not in ctx.events
        ]
        if missing_event_ids:
            raise ValueError(
                f"Unknown event id(s) for single-track constraint: {', '.join(sorted(missing_event_ids))}."
            )

        track_ids = _track_ids(ctx)
        anchor_event_id = normalized_event_ids[0]

        for event_id in normalized_event_ids[1:]:
            for track_id in track_ids:
                ctx.model.Add(
                    ctx.vars.track[event_id, track_id] == ctx.vars.track[anchor_event_id, track_id]
                )

        return []

    return _constraint




def prefer_soft_sessions_to_overlap_hard_sessions(*, weight: int = 100) -> PenaltyFn:
    """Prefer each soft-skill session to overlap at least one hard-skill session."""

    def _constraint(ctx):
        soft_event_ids = [
            event_id
            for event_id, event in ctx.events.items()
            if event.skill_type == SkillType.SOFT
        ]
        hard_event_ids = [
            event_id
            for event_id, event in ctx.events.items()
            if event.skill_type == SkillType.HARD
        ]

        penalties: list[cp_model.LinearExpr] = []
        for soft_event_id in soft_event_ids:
            overlap_bools: list[cp_model.BoolVar] = []

            for hard_event_id in hard_event_ids:
                left_before_right = ctx.model.NewBoolVar(f"{soft_event_id}_before_{hard_event_id}")
                right_before_left = ctx.model.NewBoolVar(f"{hard_event_id}_before_{soft_event_id}")
                overlaps = ctx.model.NewBoolVar(f"{soft_event_id}_{hard_event_id}_soft_hard_overlap")

                ctx.model.Add(ctx.vars.end[soft_event_id] <= ctx.vars.start[hard_event_id]).OnlyEnforceIf(
                    left_before_right
                )
                ctx.model.Add(ctx.vars.end[soft_event_id] > ctx.vars.start[hard_event_id]).OnlyEnforceIf(
                    left_before_right.Not()
                )

                ctx.model.Add(ctx.vars.end[hard_event_id] <= ctx.vars.start[soft_event_id]).OnlyEnforceIf(
                    right_before_left
                )
                ctx.model.Add(ctx.vars.end[hard_event_id] > ctx.vars.start[soft_event_id]).OnlyEnforceIf(
                    right_before_left.Not()
                )

                ctx.model.AddBoolOr([left_before_right, right_before_left, overlaps])
                ctx.model.AddImplication(left_before_right, overlaps.Not())
                ctx.model.AddImplication(right_before_left, overlaps.Not())

                overlap_bools.append(overlaps)

            if not overlap_bools:
                continue

            has_hard_alternative = ctx.model.NewBoolVar(f"{soft_event_id}_has_hard_alternative")
            ctx.model.AddMaxEquality(has_hard_alternative, overlap_bools)
            penalties.append(weight * (1 - has_hard_alternative))

        return penalties

    return _constraint


def reward_cross_track_attendance_options(
    *,
    weight: int = 100,
    transition_gap_min: int = 5,
) -> PenaltyFn:
    """Reward same-halfday session pairs that an attendee can attend sequentially.

    This favors schedules where finishing one session still leaves enough time to
    catch another session on a different track in the same halfday.
    """

    def _constraint(ctx):
        penalties: list[cp_model.LinearExpr] = []
        event_ids = list(ctx.events.keys())
        track_ids = _track_ids(ctx)
        lunch_start, lunch_end = _lunch_window_relative(ctx)

        ends_before_lunch: dict[str, cp_model.BoolVar] = {}
        starts_after_lunch: dict[str, cp_model.BoolVar] = {}
        for event_id in event_ids:
            ends_before = ctx.model.NewBoolVar(f"{event_id}_ends_before_lunch_for_attendance")
            starts_after = ctx.model.NewBoolVar(f"{event_id}_starts_after_lunch_for_attendance")

            ctx.model.Add(ctx.vars.end[event_id] <= lunch_start).OnlyEnforceIf(ends_before)
            ctx.model.Add(ctx.vars.end[event_id] > lunch_start).OnlyEnforceIf(ends_before.Not())
            ctx.model.Add(ctx.vars.start[event_id] >= lunch_end).OnlyEnforceIf(starts_after)
            ctx.model.Add(ctx.vars.start[event_id] < lunch_end).OnlyEnforceIf(starts_after.Not())

            ends_before_lunch[event_id] = ends_before
            starts_after_lunch[event_id] = starts_after

        for left_id in event_ids:
            for right_id in event_ids:
                if left_id == right_id:
                    continue

                same_track_candidates: list[cp_model.BoolVar] = []
                for track_id in track_ids:
                    both_on_track = ctx.model.NewBoolVar(
                        f"{left_id}_{right_id}_both_on_track_{track_id}_for_attendance"
                    )
                    ctx.model.AddImplication(both_on_track, ctx.vars.track[left_id, track_id])
                    ctx.model.AddImplication(both_on_track, ctx.vars.track[right_id, track_id])
                    ctx.model.AddBoolOr(
                        [
                            ctx.vars.track[left_id, track_id].Not(),
                            ctx.vars.track[right_id, track_id].Not(),
                            both_on_track,
                        ]
                    )
                    same_track_candidates.append(both_on_track)

                same_track = ctx.model.NewBoolVar(f"{left_id}_{right_id}_same_track_for_attendance")
                ctx.model.AddMaxEquality(same_track, same_track_candidates)

                can_follow = ctx.model.NewBoolVar(f"{left_id}_before_{right_id}_for_attendance")
                ctx.model.Add(
                    ctx.vars.end[left_id] + transition_gap_min <= ctx.vars.start[right_id]
                ).OnlyEnforceIf(can_follow)
                ctx.model.Add(
                    ctx.vars.end[left_id] + transition_gap_min > ctx.vars.start[right_id]
                ).OnlyEnforceIf(can_follow.Not())

                morning_transition = ctx.model.NewBoolVar(
                    f"{left_id}_{right_id}_morning_transition"
                )
                afternoon_transition = ctx.model.NewBoolVar(
                    f"{left_id}_{right_id}_afternoon_transition"
                )
                transition_bonus = ctx.model.NewBoolVar(
                    f"{left_id}_{right_id}_attendance_bonus"
                )

                ctx.model.AddImplication(morning_transition, can_follow)
                ctx.model.AddImplication(morning_transition, ends_before_lunch[left_id])
                ctx.model.AddImplication(morning_transition, ends_before_lunch[right_id])
                ctx.model.AddImplication(morning_transition, same_track.Not())
                ctx.model.AddBoolOr(
                    [
                        can_follow.Not(),
                        ends_before_lunch[left_id].Not(),
                        ends_before_lunch[right_id].Not(),
                        same_track,
                        morning_transition,
                    ]
                )

                ctx.model.AddImplication(afternoon_transition, can_follow)
                ctx.model.AddImplication(afternoon_transition, starts_after_lunch[left_id])
                ctx.model.AddImplication(afternoon_transition, starts_after_lunch[right_id])
                ctx.model.AddImplication(afternoon_transition, same_track.Not())
                ctx.model.AddBoolOr(
                    [
                        can_follow.Not(),
                        starts_after_lunch[left_id].Not(),
                        starts_after_lunch[right_id].Not(),
                        same_track,
                        afternoon_transition,
                    ]
                )

                ctx.model.AddMaxEquality(
                    transition_bonus,
                    [morning_transition, afternoon_transition],
                )
                penalties.append(-weight * transition_bonus)

        return penalties

    return _constraint
