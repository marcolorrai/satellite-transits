"""Small desktop interface for configuring and running the transit search."""

import configparser
from datetime import datetime, timedelta
from pathlib import Path
import queue
import re
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import webbrowser


ROOT = Path(__file__).resolve().parent
TRANSIT_SCRIPT = ROOT / "transit.py"
CONFIG_KEYS = {
    "latitude": ("Observer", "latitude"),
    "longitude": ("Observer", "longitude"),
    "elevation_m": ("Observer", "elevation_m"),
    "max_distance_km": ("Search", "max_distance_km"),
    "start_date": ("Search", "start_date"),
    "end_date": ("Search", "end_date"),
    "coarse_step_seconds": ("Search", "coarse_step_seconds"),
    "fine_step_seconds": ("Search", "fine_step_seconds"),
    "min_altitude_deg": ("Search", "min_altitude_deg"),
    "transit_objects": ("Targets", "transit_objects"),
    "tle_source": ("Satellites", "tle_source"),
    "map_dir": ("Output", "map_dir"),
    "tile_layers": ("Output", "tile_layers"),
}
EVENT_LINE = re.compile(
    r"^\s+\[(CONFIRMED|NEAR)\]\s+(\S+)\s+→\s+(.+?)\s+"
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}) UTC\s+"
    r"sep=\s*([\d.]+)°\s+sep_km=\s*([\d.]+) km\s+"
    r"slant=\s*([\d.]+) km(.*)$"
)


def normalize_config_text(text):
    normalized_lines = []
    in_targets = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_targets = stripped[1:-1].strip().lower() == "targets"
        elif in_targets and stripped and not stripped.startswith(("#", ";")):
            if not re.match(r"^[\w.-]+\s*[:=]", stripped):
                line = f"    {line.lstrip()}"
        normalized_lines.append(line)
    return "\n".join(normalized_lines)


def load_config(path):
    parser = configparser.ConfigParser()
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    source = config_path.read_text(encoding="utf-8")
    parser.read_string(normalize_config_text(source), source=str(config_path))
    return parser


def update_config(parser, values):
    for key, value in values.items():
        section, option = CONFIG_KEYS[key]
        if not parser.has_section(section):
            parser.add_section(section)
        parser.set(section, option, value.strip())


def save_config_file(path, template_path, values):
    source = Path(path) if Path(path).exists() else Path(template_path)
    parser = load_config(source)
    update_config(parser, values)
    source_text = source.read_text(encoding="utf-8")
    lines = normalize_config_text(source_text).splitlines(keepends=True)

    for field, (section, option) in CONFIG_KEYS.items():
        value = parser.get(section, option, fallback="")
        section_pattern = re.compile(
            r"^\s*\[" + re.escape(section) + r"\]\s*(?:[#;].*)?$",
            re.IGNORECASE,
        )
        section_start = next(
            (i for i, line in enumerate(lines) if section_pattern.match(line)),
            None,
        )
        if section_start is None:
            if lines and not lines[-1].endswith("\n"):
                lines[-1] += "\n"
            if lines and lines[-1].strip():
                lines.append("\n")
            lines.extend([f"[{section}]\n", f"{option} = {value}\n"])
            continue

        section_end = next(
            (
                i for i in range(section_start + 1, len(lines))
                if re.match(r"^\s*\[", lines[i])
            ),
            len(lines),
        )
        raw_option = option
        if (field == "tile_layers"
                and not any(
                    re.match(r"^\s*tile_layers\s*[=:]", lines[i],
                             re.IGNORECASE)
                    for i in range(section_start + 1, section_end)
                )
                and any(
                    re.match(r"^\s*tile_layer\s*[=:]", lines[i],
                             re.IGNORECASE)
                    for i in range(section_start + 1, section_end)
                )):
            raw_option = "tile_layer"
        option_pattern = re.compile(
            r"^(\s*)" + re.escape(raw_option) + r"(\s*[=:]\s*).*$",
            re.IGNORECASE,
        )
        option_line = next(
            (
                i for i in range(section_start + 1, section_end)
                if option_pattern.match(lines[i])
            ),
            None,
        )
        if option_line is None:
            if section_end > section_start and not lines[section_end - 1].endswith(
                "\n"
            ):
                lines[section_end - 1] += "\n"
            lines.insert(section_end, f"{option} = {value}\n")
            continue

        match = option_pattern.match(lines[option_line])
        newline = "\n" if lines[option_line].endswith("\n") else ""
        lines[option_line] = (
            f"{match.group(1)}{raw_option}{match.group(2)}{value}{newline}"
        )
        if field == "transit_objects":
            continuation = option_line + 1
            while continuation < section_end:
                line = lines[continuation]
                if (line.strip() and line[:1].isspace()
                        and not line.lstrip().startswith(("#", ";"))):
                    del lines[continuation]
                    section_end -= 1
                else:
                    continuation += 1

    Path(path).write_text("".join(lines), encoding="utf-8")


def parse_event_line(line):
    match = EVENT_LINE.match(line)
    if not match:
        return None
    status, satellite, target, event_time, sep_deg, sep_km, slant, tags = (
        match.groups()
    )
    conditions = [
        name for name in ("DAYLIGHT", "SATELLITE UNLIT")
        if f"[{name}]" in tags
    ]
    return {
        "status": status,
        "satellite": satellite.strip(),
        "target": target.strip(),
        "time": event_time,
        "sep_deg": float(sep_deg),
        "sep_km": float(sep_km),
        "slant_km": float(slant),
        "tags": conditions,
    }


def event_row_tags(event):
    if event["status"] == "CONFIRMED":
        return ("confirmed",)
    if event["sep_km"] <= 1.0:
        return ("within_1km",)
    return ()


def find_event_map(event, map_dir):
    event_time = datetime.strptime(event["time"], "%Y-%m-%d %H:%M:%S.%f")
    event_time += timedelta(microseconds=500_000)
    stamp = event_time.strftime("%Y%m%dT%H%M%SZ")
    safe_satellite = re.sub(r"[^A-Za-z0-9_-]", "_", event["satellite"])
    safe_target = re.sub(r"[^A-Za-z0-9_-]", "_", event["target"])
    pattern = (
        f"{safe_satellite}_{safe_target}_{stamp}_{event['status']}_*.html"
    )
    return next(iter(sorted(Path(map_dir).glob(pattern))), None)


class TransitGui:
    def __init__(self, root):
        self.root = root
        self.root.title("Satellite Transit Finder")
        self.root.geometry("1180x760")
        self.messages = queue.Queue()
        self.process = None
        self.temp_config = None
        self.events = []
        self.map_paths = {}
        self.config_path = ROOT / "config.ini"
        self.template_path = ROOT / "config.example.ini"
        self.variables = {}

        self._build_ui()
        self._load_settings()
        self.root.after(100, self._poll_messages)

    def _build_ui(self):
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)

        config_box = ttk.LabelFrame(outer, text="Search configuration", padding=8)
        config_box.pack(fill="x")

        file_row = ttk.Frame(config_box)
        file_row.pack(fill="x", pady=(0, 6))
        self.config_label = ttk.Label(file_row, text="")
        self.config_label.pack(side="left", fill="x", expand=True)
        ttk.Button(file_row, text="Open config…", command=self._choose_config).pack(
            side="right"
        )

        fields_frame = ttk.Frame(config_box)
        fields_frame.pack(fill="x")
        fields = [
            ("Latitude", "latitude"),
            ("Longitude", "longitude"),
            ("Elevation (m)", "elevation_m"),
            ("Search radius (km)", "max_distance_km"),
            ("Start date (UTC, blank = now)", "start_date"),
            ("End date (UTC, exclusive)", "end_date"),
            ("Coarse step (s)", "coarse_step_seconds"),
            ("Fine step (s)", "fine_step_seconds"),
            ("Minimum altitude (°)", "min_altitude_deg"),
            ("Targets (comma-separated)", "transit_objects"),
            ("TLE source URL", "tle_source"),
            ("Map directory", "map_dir"),
            ("Map tile layers", "tile_layers"),
        ]
        for index, (label, key) in enumerate(fields):
            row, column = divmod(index, 3)
            cell = ttk.Frame(fields_frame)
            cell.grid(row=row + 1, column=column, sticky="ew", padx=4, pady=3)
            ttk.Label(cell, text=label).pack(anchor="w")
            variable = tk.StringVar()
            self.variables[key] = variable
            ttk.Entry(cell, textvariable=variable).pack(fill="x")
        for column in range(3):
            fields_frame.columnconfigure(column, weight=1)

        buttons = ttk.Frame(outer)
        buttons.pack(fill="x", pady=8)
        self.run_button = ttk.Button(
            buttons, text="Run search", command=self._run_search
        )
        self.run_button.pack(side="left")
        self.save_button = ttk.Button(
            buttons, text="Save settings", command=self._save_settings
        )
        self.save_button.pack(side="left", padx=(6, 0))
        self.status = tk.StringVar(value="Ready")
        ttk.Label(buttons, textvariable=self.status).pack(side="left", padx=12)

        result_frame = ttk.LabelFrame(outer, text="Events (click an event to open its map)")
        result_frame.pack(fill="both", expand=True)
        columns = (
            "time", "satellite", "target", "status", "sep", "slant",
            "conditions",
        )
        self.results = ttk.Treeview(
            result_frame, columns=columns, show="headings", height=12
        )
        headings = {
            "time": ("UTC", 190),
            "satellite": ("Satellite", 90),
            "target": ("Target", 150),
            "status": ("Status", 100),
            "sep": ("Separation (km)", 130),
            "slant": ("Slant range (km)", 140),
            "conditions": ("Conditions", 180),
        }
        for key, (title, width) in headings.items():
            self.results.heading(key, text=title)
            self.results.column(key, width=width, anchor="w")
        scroll = ttk.Scrollbar(
            result_frame, orient="vertical", command=self.results.yview
        )
        self.results.configure(yscrollcommand=scroll.set)
        self.results.tag_configure(
            "confirmed", background="#d9f2df", foreground="#155724"
        )
        self.results.tag_configure(
            "within_1km", background="#fff1c2", foreground="#704f00"
        )
        self.results.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.results.bind("<ButtonRelease-1>", self._open_clicked_map)
        self.results.bind("<Return>", self._open_selected_map)

        log_frame = ttk.LabelFrame(outer, text="Search progress", padding=5)
        log_frame.pack(fill="both", expand=True, pady=(8, 0))
        self.log = tk.Text(log_frame, height=8, wrap="none", state="disabled")
        log_scroll = ttk.Scrollbar(
            log_frame, orient="vertical", command=self.log.yview
        )
        self.log.configure(yscrollcommand=log_scroll.set)
        self.log.pack(side="left", fill="both", expand=True)
        log_scroll.pack(side="right", fill="y")

    def _load_settings(self):
        try:
            source = (
                self.config_path
                if self.config_path.exists()
                else self.template_path
            )
            parser = load_config(source)
            for key, variable in self.variables.items():
                section, option = CONFIG_KEYS[key]
                if key == "tile_layers" and not parser.has_option(
                    section, option
                ):
                    option = "tile_layer"
                variable.set(parser.get(section, option, fallback=""))
            label = str(self.config_path)
            if source != self.config_path:
                label += " (defaults from config.example.ini)"
            self.config_label.configure(text=label)
        except (OSError, configparser.Error) as exc:
            messagebox.showerror("Cannot load configuration", str(exc))

    def _choose_config(self):
        path = filedialog.askopenfilename(
            title="Open configuration",
            initialdir=str(self.config_path.parent),
            filetypes=[("INI files", "*.ini"), ("All files", "*")],
        )
        if path:
            self.config_path = Path(path)
            self._load_settings()

    def _current_values(self):
        return {
            key: variable.get()
            for key, variable in self.variables.items()
        }

    def _save_settings(self):
        try:
            save_config_file(
                self.config_path, self.template_path, self._current_values()
            )
            self.status.set(f"Saved {self.config_path.name}")
        except (OSError, configparser.Error) as exc:
            messagebox.showerror("Cannot save configuration", str(exc))

    def _run_search(self):
        if self.process is not None:
            return
        try:
            source = (
                self.config_path
                if self.config_path.exists()
                else self.template_path
            )
            parser = load_config(source)
            update_config(parser, self._current_values())
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                suffix=".ini",
                prefix=".transit-gui-",
                dir=ROOT,
                delete=False,
            ) as config_file:
                parser.write(config_file)
                self.temp_config = Path(config_file.name)
        except (OSError, configparser.Error) as exc:
            messagebox.showerror("Cannot prepare search", str(exc))
            return

        self.events.clear()
        self.map_paths.clear()
        for item in self.results.get_children():
            self.results.delete(item)
        self._set_log("")
        self.run_button.configure(state="disabled")
        self.status.set("Starting search…")
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        try:
            self.process = subprocess.Popen(
                [
                    sys.executable,
                    str(TRANSIT_SCRIPT),
                    "--config",
                    str(self.temp_config),
                ],
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            for line in self.process.stdout:
                self.messages.put(("line", line))
            return_code = self.process.wait()
            self.messages.put(("finished", return_code))
        except OSError as exc:
            self.messages.put(("error", str(exc)))

    def _poll_messages(self):
        try:
            while True:
                message_type, payload = self.messages.get_nowait()
                if message_type == "line":
                    self._append_log(payload)
                    event = parse_event_line(payload.rstrip("\r\n"))
                    if event is not None:
                        self.events.append(event)
                    self.status.set(payload.strip()[-100:] or "Searching…")
                elif message_type == "finished":
                    self._finish_search(payload)
                elif message_type == "error":
                    self._finish_search(None)
                    messagebox.showerror("Search failed", payload)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_messages)

    def _finish_search(self, return_code):
        self.process = None
        self.run_button.configure(state="normal")
        if self.temp_config is not None:
            try:
                self.temp_config.unlink(missing_ok=True)
            except OSError as exc:
                self._append_log(f"Could not remove temporary config: {exc}\n")
            self.temp_config = None
        if return_code is None:
            self.status.set("Search could not start")
        elif return_code != 0:
            self.status.set(f"Search failed (exit code {return_code})")
        else:
            self._populate_results()
            self.status.set(f"Finished: {len(self.events)} event(s)")

    def _populate_results(self):
        values = self._current_values()
        map_dir = Path(values["map_dir"])
        if not map_dir.is_absolute():
            map_dir = ROOT / map_dir
        for event in sorted(self.events, key=lambda item: item["time"]):
            map_path = find_event_map(event, map_dir)
            row = (
                event["time"],
                event["satellite"],
                event["target"],
                event["status"],
                f"{event['sep_km']:.3f}",
                f"{event['slant_km']:.0f}",
                ", ".join(event["tags"]),
            )
            item = self.results.insert(
                "", "end", values=row, tags=event_row_tags(event)
            )
            if map_path is not None:
                self.map_paths[item] = map_path

    def _open_selected_map(self, _event=None):
        selection = self.results.selection()
        if not selection:
            return
        self._open_map_for_row(selection[0])

    def _open_clicked_map(self, event):
        row = self.results.identify_row(event.y)
        if row:
            self.results.selection_set(row)
            self._open_map_for_row(row)

    def _open_map_for_row(self, row):
        map_path = self.map_paths.get(row)
        if map_path is not None and map_path.is_file():
            webbrowser.open(map_path.resolve().as_uri())
        else:
            messagebox.showerror(
                "Map unavailable",
                "No generated map was found for this event.",
            )

    def _set_log(self, text):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.insert("end", text)
        self.log.configure(state="disabled")

    def _append_log(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")


def main():
    root = tk.Tk()
    TransitGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
