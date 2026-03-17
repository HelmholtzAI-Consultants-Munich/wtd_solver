from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import math
from functools import lru_cache
from html import escape
from pathlib import Path


BACKGROUND = "#ecfbfd"
PANEL = "#ffffff"
GRID = "#cdeefb"
TEXT = "#002864"
MUTED = "#002864"
LUNCH = "#cdeefb"
COFFEE_BREAK = LUNCH
ROOM_HEADER = "#002864"
SOFT = "#05e5ba"
WORKSHOP = "#14c8ff"
TUTORIAL = "#002864"
SEQUENCE_FILL = "#cdeefb"
SEQUENCE_STROKE = "#002864"
FONT_FILE = "HelmholtzHalvarMittel-Rg (2).otf"
FONT_NAME = "Helmholtz Halvar Mittel Rg"
FONT_ALT_NAME = "Helmholtz Halvar Mittelschrift Regular"
FONT_FAMILY = f"'{FONT_NAME}', '{FONT_ALT_NAME}', sans-serif"
TITLE = "HAICON26 Workshops & Tutorials Day agenda"
PNG_EXPORT_SCALE = 2
COFFEE_BREAKS = [
    {"label": "Coffee break", "start_min": 11 * 60, "end_min": 11 * 60 + 15},
    {"label": "Coffee break", "start_min": 16 * 60 + 15, "end_min": 16 * 60 + 30},
]


def _minutes_to_label(total_minutes: int) -> str:
    hours = total_minutes // 60
    minutes = total_minutes % 60
    return f"{hours:02d}:{minutes:02d}"


def _wrap_text(text: str, max_chars: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]

    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        tentative = f"{current} {word}"
        if len(tentative) <= max_chars:
            current = tentative
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _truncate_line(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= 1:
        return text[:max_chars]
    return f"{text[:max_chars - 1].rstrip()}…"


def _fit_title_lines(text: str, *, max_chars: int, height_px: float) -> list[str]:
    wrapped = _wrap_text(text, max_chars)

    if height_px < 72:
        max_lines = 1
    elif height_px < 100:
        max_lines = 2
    else:
        max_lines = 3

    fitted = wrapped[:max_lines]
    if len(wrapped) > max_lines:
        fitted[-1] = _truncate_line(fitted[-1], max_chars - 1)
        if not fitted[-1].endswith("…"):
            fitted[-1] = f"{fitted[-1]}…"

    return fitted


def _event_fill(event: dict[str, object]) -> str:
    skill_type = event.get("skill_type")
    if skill_type == "soft":
        return SOFT
    if bool(event["is_workshop"]):
        return WORKSHOP
    return TUTORIAL


def _event_text_fill(event: dict[str, object]) -> str:
    if event.get("skill_type") == "soft":
        return TEXT
    if bool(event["is_workshop"]):
        return TEXT
    return "#ffffff"


def _embedded_font_style() -> str:
    font_path = Path(__file__).with_name(FONT_FILE)
    if not font_path.exists():
        return ""

    font_data = base64.b64encode(font_path.read_bytes()).decode("ascii")
    return (
        "<style><![CDATA["
        f"@font-face{{font-family:'{FONT_NAME}';src:url(data:font/otf;base64,{font_data}) format('opentype');}}"
        "]]>"
        "</style>"
    )


@lru_cache
def _configured_hidden_after_ids() -> frozenset[str]:
    definitions_path = Path(__file__).with_name("def.py")
    if not definitions_path.exists():
        return frozenset()

    spec = importlib.util.spec_from_file_location("agenda_definitions_for_plot", definitions_path)
    if spec is None or spec.loader is None:
        return frozenset()

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    build_events = getattr(module, "build_events", None)
    if build_events is None:
        return frozenset()

    events = build_events()
    return frozenset(
        event_id
        for event_id, event in events.items()
        if bool(getattr(event, "after_hidden", False))
    )


def _event_hides_after_arrow(event: dict[str, object], hidden_after_ids: frozenset[str]) -> bool:
    if "after_hidden" in event:
        return bool(event["after_hidden"])
    return str(event["id"]) in hidden_after_ids


@lru_cache
def _load_font():
    try:
        from fontTools.ttLib import TTFont
    except ImportError as exc:
        raise RuntimeError(
            "Text-to-path rendering requires the Python package 'fonttools'. Install dependencies from requirements.txt."
        ) from exc

    font_path = Path(__file__).with_name(FONT_FILE)
    if not font_path.exists():
        raise RuntimeError(f"Font file not found: {font_path}")

    return TTFont(font_path)


@lru_cache
def _font_units_per_em() -> int:
    return int(_load_font()["head"].unitsPerEm)


@lru_cache
def _glyph_outline(char: str) -> tuple[str, int]:
    from fontTools.pens.svgPathPen import SVGPathPen

    font = _load_font()
    glyph_set = font.getGlyphSet()
    cmap = font.getBestCmap()
    glyph_name = cmap.get(ord(char), ".notdef")
    pen = SVGPathPen(glyph_set)
    glyph_set[glyph_name].draw(pen)
    advance_width, _ = font["hmtx"][glyph_name]
    return pen.getCommands(), int(advance_width)


def _text_width(text: str, font_size: float) -> float:
    scale = font_size / _font_units_per_em()
    return sum(_glyph_outline(char)[1] for char in text) * scale


def _svg_text(
    text: str,
    *,
    x: float,
    y: float,
    font_size: float,
    fill: str,
    text_anchor: str = "start",
    font_weight: int | None = None,
    as_path: bool = False,
) -> str:
    if not as_path:
        attrs = [
            f'x="{x}"',
            f'y="{y}"',
            f'font-family="{FONT_FAMILY}"',
            f'font-size="{font_size}"',
            f'fill="{fill}"',
        ]
        if font_weight is not None:
            attrs.append(f'font-weight="{font_weight}"')
        if text_anchor != "start":
            attrs.append(f'text-anchor="{text_anchor}"')
        return f'<text {" ".join(attrs)}>{escape(text)}</text>'

    scale = font_size / _font_units_per_em()
    width = _text_width(text, font_size)
    start_x = x
    if text_anchor == "end":
        start_x -= width
    elif text_anchor == "middle":
        start_x -= width / 2

    parts: list[str] = [f'<g fill="{fill}">']
    cursor_x = start_x
    for char in text:
        commands, advance_width = _glyph_outline(char)
        if commands:
            parts.append(
                f'<path d="{commands}" transform="translate({cursor_x:.2f} {y:.2f}) scale({scale:.6f} {-scale:.6f})"/>'
            )
        cursor_x += advance_width * scale
    parts.append("</g>")
    return "".join(parts)


def _sequence_connector_parts(center_x: float, start_y: float, end_y: float) -> list[str]:
    available_gap = max(end_y - start_y, 18)
    chevron_count = 3 if available_gap >= 40 else 2
    chevron_spacing = 16 if chevron_count == 3 else 18
    first_chevron_y = (start_y + end_y) / 2 - ((chevron_count - 1) * chevron_spacing) / 2
    half_width = 20
    half_height = 12
    notch = 3

    parts: list[str] = []

    for index in range(chevron_count):
        chevron_y = first_chevron_y + index * chevron_spacing
        parts.append(
            (
                f'<path d="M {center_x - half_width:.1f} {chevron_y - half_height:.1f} '
                f'L {center_x - half_width:.1f} {chevron_y - notch:.1f} '
                f'L {center_x:.1f} {chevron_y + half_height:.1f} '
                f'L {center_x + half_width:.1f} {chevron_y - notch:.1f} '
                f'L {center_x + half_width:.1f} {chevron_y - half_height:.1f} '
                f'L {center_x:.1f} {chevron_y + 1:.1f} Z" '
                f'fill="{SEQUENCE_FILL}" stroke="{SEQUENCE_STROKE}" stroke-width="1.4" '
                'stroke-linejoin="miter"/>'
            )
        )

    return parts


def render_svg(schedule: dict[str, object], *, text_as_paths: bool = False) -> str:
    events = schedule["events"]
    lunch = schedule["lunch"]
    meta = schedule["meta"]
    coffee_breaks = [dict(coffee_break) for coffee_break in COFFEE_BREAKS]
    hidden_after_ids = _configured_hidden_after_ids()
    track_load = {
        int(track): sum(
            int(event["end_min"]) - int(event["start_min"])
            for event in events
            if int(event["track"]) == int(track)
        )
        for track in {int(event["track"]) for event in events}
    }
    tracks = sorted(track_load, key=lambda track: (-track_load[track], track))
    display_track = {track: index + 1 for index, track in enumerate(tracks)}

    day_start = min(int(event["start_min"]) for event in events)
    day_end = max(int(event["end_min"]) for event in events)
    day_start = min(day_start, int(lunch["start_min"]))
    day_end = max(day_end, int(lunch["end_min"]))
    for coffee_break in coffee_breaks:
        day_start = min(day_start, int(coffee_break["start_min"]))
        day_end = max(day_end, int(coffee_break["end_min"]))

    left_margin = 96
    top_margin = 126
    track_width = 236
    track_gap = 18
    minutes_scale = 1.45
    grid_height = int(math.ceil((day_end - day_start) * minutes_scale))
    width = left_margin + len(tracks) * track_width + (len(tracks) - 1) * track_gap + 72
    height = top_margin + grid_height + 120

    lunch_y = top_margin + (int(lunch["start_min"]) - day_start) * minutes_scale
    lunch_height = (int(lunch["end_min"]) - int(lunch["start_min"])) * minutes_scale
    coffee_break_layout = [
        {
            **coffee_break,
            "y": top_margin + (int(coffee_break["start_min"]) - day_start) * minutes_scale,
            "height": (int(coffee_break["end_min"]) - int(coffee_break["start_min"])) * minutes_scale,
        }
        for coffee_break in coffee_breaks
    ]

    track_x = {
        track: left_margin + index * (track_width + track_gap)
        for index, track in enumerate(tracks)
    }
    coffee_break_label_track_index = min(4, max(len(tracks) - 1, 0))
    coffee_break_label_x = (
        left_margin
        + coffee_break_label_track_index * (track_width + track_gap)
        + track_width / 2
    )

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        '<linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">',
        f'<stop offset="0%" stop-color="{BACKGROUND}"/>',
        '<stop offset="100%" stop-color="#ffffff"/>',
        "</linearGradient>",
        '<pattern id="lunch-hatch" patternUnits="userSpaceOnUse" width="12" height="12" patternTransform="rotate(35)">',
        f'<line x1="0" y1="0" x2="0" y2="12" stroke="{TEXT}" stroke-opacity="0.16" stroke-width="4"/>',
        "</pattern>",
        f'<clipPath id="schedule-clip"><rect x="{left_margin}" y="{top_margin}" width="{width - left_margin - 72}" height="{grid_height}"/></clipPath>',
        _embedded_font_style(),
        "</defs>",
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="url(#bg)"/>',
        _svg_text(
            TITLE,
            x=left_margin,
            y=56,
            font_size=34,
            font_weight=700,
            fill=TEXT,
            as_path=text_as_paths,
        ),
        # f'<text x="{left_margin}" y="82" font-family="{FONT_FAMILY}" font-size="14" fill="{MUTED}">Tracks {escape(str(meta["num_tracks"]))}</text>',
        f'<rect x="{left_margin - 18}" y="{top_margin - 22}" width="{width - left_margin - 36}" height="{grid_height + 44}" rx="28" fill="{PANEL}" opacity="0.9"/>',
    ]

    tick_minutes = list(range((day_start // 30) * 30, day_end + 1, 30))
    for minute in tick_minutes:
        y = top_margin + (minute - day_start) * minutes_scale
        major = minute % 60 == 0
        parts.append(
            f'<line x1="{left_margin}" y1="{y:.1f}" x2="{width - 36}" y2="{y:.1f}" stroke="{GRID}" stroke-width="{1.4 if major else 0.8}" opacity="{0.75 if major else 0.4}"/>'
        )
        label = _minutes_to_label(minute)
        parts.append(_svg_text(label, x=left_margin - 14, y=y + 6, text_anchor="end", font_size=13, fill=MUTED, as_path=text_as_paths))

    header_inset = 4
    header_width = track_width - 2 * header_inset
    header_parts: list[str] = []

    for track in tracks:
        x = track_x[track]
        header_parts.extend(
            [
                f'<rect x="{x + header_inset}" y="{top_margin - 58}" width="{header_width}" height="42" rx="18" fill="{ROOM_HEADER}"/>',
                _svg_text(
                    f"Track {display_track[track]}",
                    x=x + 22,
                    y=top_margin - 30,
                    font_size=18,
                    font_weight=700,
                    fill="white",
                    as_path=text_as_paths,
                ),
            ]
        )

    parts.append('<g clip-path="url(#schedule-clip)">')
    event_boxes: dict[str, dict[str, float]] = {}
    for event in events:
        x = track_x[int(event["track"])]
        y = top_margin + (int(event["start_min"]) - day_start) * minutes_scale
        height_px = max((int(event["end_min"]) - int(event["start_min"])) * minutes_scale - 6, 42)
        left = x + 6
        right = x + track_width - 6
        top = y + 3
        bottom = top + height_px
        event_boxes[str(event["id"])] = {
            "left": left,
            "right": right,
            "top": top,
            "bottom": bottom,
            "track": float(int(event["track"])),
            "track_x": float(x),
        }
        fill = _event_fill(event)
        text_fill = _event_text_fill(event)
        parts.append(
            f'<rect x="{left + 2}" y="{top + 6:.1f}" width="{track_width - 12}" height="{height_px:.1f}" rx="20" fill="{TEXT}" opacity="0.10"/>'
        )
        parts.append(
            f'<rect x="{left}" y="{top:.1f}" width="{track_width - 12}" height="{height_px:.1f}" rx="20" fill="{fill}"/>'
        )
        parts.append(
            f'<rect x="{left}" y="{top:.1f}" width="{track_width - 12}" height="{height_px:.1f}" rx="20" fill="white" opacity="0.08"/>'
        )

        lines = _fit_title_lines(str(event["title"]), max_chars=26, height_px=height_px)
        text_y = y + (20 if len(lines) == 1 else 24)
        line_spacing = 16
        for index, line in enumerate(lines):
            parts.append(
                _svg_text(
                    line,
                    x=x + 22,
                    y=text_y + index * line_spacing,
                    font_size=16,
                    font_weight=700,
                    fill=text_fill,
                    as_path=text_as_paths,
                )
            )

        footer_y = y + height_px - 16
        parts.append(_svg_text(f"{event['start']} - {event['end']}", x=x + 22, y=footer_y, font_size=13, fill=text_fill, as_path=text_as_paths))
        parts.append(
            _svg_text(
                str(event["room_id"]).title(),
                x=x + track_width - 22,
                y=footer_y,
                text_anchor="end",
                font_size=12,
                font_weight=700,
                fill=text_fill,
                as_path=text_as_paths,
            )
        )

        if bool(event["spans_lunch"]):
            overlap_start = max(int(event["start_min"]), int(lunch["start_min"]))
            overlap_end = min(int(event["end_min"]), int(lunch["end_min"]))
            if overlap_end > overlap_start:
                overlap_y = top_margin + (overlap_start - day_start) * minutes_scale
                overlap_height = (overlap_end - overlap_start) * minutes_scale
                parts.append(
                    f'<rect x="{x + 10}" y="{overlap_y:.1f}" width="{track_width - 20}" height="{overlap_height:.1f}" rx="12" fill="{LUNCH}" opacity="0.82"/>'
                )
                parts.append(
                    f'<rect x="{x + 10}" y="{overlap_y:.1f}" width="{track_width - 20}" height="{overlap_height:.1f}" rx="12" fill="url(#lunch-hatch)" opacity="0.48"/>'
                )
                if overlap_height >= 28:
                    parts.append(
                        _svg_text(
                            "Lunch during workshop",
                            x=x + track_width / 2,
                            y=overlap_y + min(overlap_height / 2 + 5, overlap_height - 8),
                            text_anchor="middle",
                            font_size=13,
                            font_weight=700,
                            fill=TEXT,
                            as_path=text_as_paths,
                        )
                    )

        for coffee_break in coffee_break_layout:
            overlap_start = max(int(event["start_min"]), int(coffee_break["start_min"]))
            overlap_end = min(int(event["end_min"]), int(coffee_break["end_min"]))
            if overlap_end <= overlap_start:
                continue

            overlap_y = top_margin + (overlap_start - day_start) * minutes_scale
            overlap_height = (overlap_end - overlap_start) * minutes_scale
            parts.append(
                f'<rect x="{x + 10}" y="{overlap_y:.1f}" width="{track_width - 20}" height="{overlap_height:.1f}" rx="12" fill="{COFFEE_BREAK}" opacity="0.82"/>'
            )
            parts.append(
                f'<rect x="{x + 10}" y="{overlap_y:.1f}" width="{track_width - 20}" height="{overlap_height:.1f}" rx="12" fill="url(#lunch-hatch)" opacity="0.48"/>'
            )

    parts.append(
        f'<rect x="{left_margin}" y="{lunch_y}" width="{width - left_margin - 72}" height="{lunch_height}" rx="18" fill="{LUNCH}" opacity="0.28"/>'
    )
    parts.append(
        f'<rect x="{left_margin}" y="{lunch_y}" width="{width - left_margin - 72}" height="{lunch_height}" rx="18" fill="url(#lunch-hatch)" opacity="0.18"/>'
    )
    parts.append(
        _svg_text(
            "Lunch",
            x=left_margin + (width - left_margin - 72) / 2,
            y=lunch_y + 29,
            text_anchor="middle",
            font_size=20,
            font_weight=700,
            fill=TEXT,
            as_path=text_as_paths,
        )
    )
    parts.append(
        _svg_text(
            f"{lunch['start']} - {lunch['end']}",
            x=left_margin + (width - left_margin - 72) / 2,
            y=lunch_y + 50,
            text_anchor="middle",
            font_size=13,
            fill=TEXT,
            as_path=text_as_paths,
        )
    )
    for coffee_break in coffee_break_layout:
        parts.append(
            f'<rect x="{left_margin}" y="{coffee_break["y"]:.1f}" width="{width - left_margin - 72}" height="{coffee_break["height"]:.1f}" rx="16" fill="{COFFEE_BREAK}" opacity="0.28"/>'
        )
        parts.append(
            f'<rect x="{left_margin}" y="{coffee_break["y"]:.1f}" width="{width - left_margin - 72}" height="{coffee_break["height"]:.1f}" rx="16" fill="url(#lunch-hatch)" opacity="0.18"/>'
        )
        parts.append(
            _svg_text(
                f'{coffee_break["label"]} {_minutes_to_label(int(coffee_break["start_min"]))} - {_minutes_to_label(int(coffee_break["end_min"]))}',
                x=coffee_break_label_x,
                y=float(coffee_break["y"]) + 14,
                text_anchor="middle",
                font_size=10,
                font_weight=700,
                fill=TEXT,
                as_path=text_as_paths,
            )
        )

    for event in sorted(
        (
            event
            for event in events
            if event.get("after") and not _event_hides_after_arrow(event, hidden_after_ids)
        ),
        key=lambda event: (int(event["start_min"]), str(event["title"])),
    ):
        source = event_boxes.get(str(event["after"]))
        target = event_boxes.get(str(event["id"]))
        if source is None or target is None:
            continue

        source_center_x = (source["left"] + source["right"]) / 2
        target_center_x = (target["left"] + target["right"]) / 2
        center_x = (source_center_x + target_center_x) / 2
        parts.extend(_sequence_connector_parts(center_x, source["bottom"], target["top"]))
    parts.append("</g>")
    parts.extend(header_parts)

    legend_y = height - 38
    legend = [
        ("Tutorial", TUTORIAL),
        ("Soft skills", SOFT),
        ("Workshop", WORKSHOP),
    ]
    legend_x = left_margin
    for label, color in legend:
        parts.append(f'<rect x="{legend_x}" y="{legend_y - 12}" width="16" height="16" rx="5" fill="{color}"/>')
        parts.append(_svg_text(label, x=legend_x + 24, y=legend_y, font_size=13, fill=MUTED, as_path=text_as_paths))
        legend_x += 148

    legend_chevron_center_x = legend_x + 22
    parts.extend(_sequence_connector_parts(legend_chevron_center_x, legend_y - 18, legend_y + 14))
    parts.append(
        _svg_text(
            "Related events",
            x=legend_chevron_center_x + 26,
            y=legend_y,
            font_size=13,
            fill=MUTED,
            as_path=text_as_paths,
        )
    )

    parts.append("</svg>")
    return "".join(parts)


def _write_png(svg: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import cairosvg
    except ImportError as exc:
        raise RuntimeError(
            "PNG export requires the Python package 'cairosvg'. Install dependencies from requirements.txt."
        ) from exc

    cairosvg.svg2png(
        bytestring=svg.encode("utf-8"),
        write_to=str(output_path),
        scale=PNG_EXPORT_SCALE,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Render an agenda JSON file as an SVG agenda board.")
    parser.add_argument("input", nargs="?", default="agenda.json", help="Input agenda JSON file.")
    parser.add_argument("output", nargs="?", default="agenda.svg", help="Output SVG file.")
    args = parser.parse_args()

    output_path = Path(args.output)
    schedule = json.loads(Path(args.input).read_text(encoding="utf-8"))
    svg = render_svg(schedule, text_as_paths=output_path.suffix.lower() == ".png")
    if output_path.suffix.lower() == ".png":
        _write_png(svg, output_path)
    else:
        output_path.write_text(svg, encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
