#!/usr/bin/env python3
"""
generate_server_graphs.py — Interactive graph generator for valkey-server benchmarks.

Reads output from run_server_matrix.py (or standalone orchestrator runs) and
generates a self-contained interactive HTML file with Plotly.js charts.

Charts:
  1. RPS Scalability — Total RPS vs total_clients per server binary
  2. RPS Delta — % difference between servers
  3. SET Latency — p50, p95, p99 per server
  4. GET Latency — p50, p95, p99 per server
  5. CPU Usage — avg CPU% during experiment per server
  6. RPS/CPU Efficiency — throughput per unit CPU% per server
  7. Per-Iteration Scatter — individual iteration RPS values (outliers marked)

Input layouts supported:
  - Matrix layout (from run_server_matrix.py):
      <dir>/_manifest.json
      <dir>/<N>-clients/<server>/aggregate.json
  - Single experiment (from orchestrator.py):
      <dir>/<server>/aggregate.json

Usage:
    python generate_server_graphs.py results/server-matrix/2026-03-23T20:00:00/

    python generate_server_graphs.py results/ --output graphs/ --title "m5.metal"

    python generate_server_graphs.py results/ --reference valkey-8.1-oss
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_matrix_data(results_dir: Path):
    """Load data from a matrix run (with _manifest.json).

    Returns:
        data: dict with keys:
            rps[server] = [(total_clients, avg_rps), ...]
            set_rps[server] = [(total_clients, avg_set_rps), ...]
            get_rps[server] = [(total_clients, avg_get_rps), ...]
            cpu[server] = [(total_clients, avg_cpu), ...]
            set_latency[server][pct] = [(total_clients, value), ...]
            get_latency[server][pct] = [(total_clients, value), ...]
            scatter[server] = [(total_clients, [rps_iter1, rps_iter2, ...], [outlier_indices]), ...]
        manifest: dict
        servers: list of server names
    """
    manifest_path = results_dir / "_manifest.json"
    manifest = None

    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)

    # Scan for N-clients directories
    data = {
        "rps": defaultdict(list),
        "set_rps": defaultdict(list),
        "get_rps": defaultdict(list),
        "cpu": defaultdict(list),
        "set_latency": defaultdict(lambda: defaultdict(list)),
        "get_latency": defaultdict(lambda: defaultdict(list)),
        "scatter": defaultdict(list),
    }
    servers = set()

    for subdir in sorted(results_dir.iterdir()):
        if not subdir.is_dir():
            continue
        # Parse N-clients pattern
        name = subdir.name
        if not name.endswith("-clients"):
            continue
        try:
            total_clients = int(name.replace("-clients", ""))
        except ValueError:
            continue

        # Scan server subdirectories
        for server_dir in sorted(subdir.iterdir()):
            if not server_dir.is_dir():
                continue
            agg_path = server_dir / "aggregate.json"
            if not agg_path.exists():
                continue

            server_name = server_dir.name
            servers.add(server_name)

            with open(agg_path) as f:
                agg = json.load(f)

            data["rps"][server_name].append((total_clients, agg["total_rps_mean"]))
            data["set_rps"][server_name].append((total_clients, agg.get("set_rps_mean", 0)))
            data["get_rps"][server_name].append((total_clients, agg.get("get_rps_mean", 0)))
            data["cpu"][server_name].append((total_clients, agg.get("avg_cpu_percent", 0)))

            # Latency
            for pct in ["avg", "p50", "p95", "p99"]:
                set_val = agg.get("set_latency", {}).get(pct)
                get_val = agg.get("get_latency", {}).get(pct)
                if set_val is not None:
                    data["set_latency"][server_name][pct].append((total_clients, set_val))
                if get_val is not None:
                    data["get_latency"][server_name][pct].append((total_clients, get_val))

            # Scatter (per-iteration RPS)
            per_iter = agg.get("per_iteration_rps", [])
            outliers = agg.get("outlier_indices", [])
            if per_iter:
                data["scatter"][server_name].append((total_clients, per_iter, outliers))

    # Sort all by total_clients
    for server in data["rps"]:
        data["rps"][server].sort(key=lambda x: x[0])
        data["set_rps"][server].sort(key=lambda x: x[0])
        data["get_rps"][server].sort(key=lambda x: x[0])
        data["cpu"][server].sort(key=lambda x: x[0])
        data["scatter"][server].sort(key=lambda x: x[0])
        for pct in data["set_latency"][server]:
            data["set_latency"][server][pct].sort(key=lambda x: x[0])
        for pct in data["get_latency"][server]:
            data["get_latency"][server][pct].sort(key=lambda x: x[0])

    return data, manifest, sorted(servers)


def load_single_experiment(results_dir: Path):
    """Load data from a single orchestrator run (no matrix sweep).

    Returns same format but with only one x-axis value (the total_connections).
    """
    data = {
        "rps": defaultdict(list),
        "set_rps": defaultdict(list),
        "get_rps": defaultdict(list),
        "cpu": defaultdict(list),
        "set_latency": defaultdict(lambda: defaultdict(list)),
        "get_latency": defaultdict(lambda: defaultdict(list)),
        "scatter": defaultdict(list),
    }
    servers = set()

    for server_dir in sorted(results_dir.iterdir()):
        if not server_dir.is_dir():
            continue
        agg_path = server_dir / "aggregate.json"
        if not agg_path.exists():
            continue

        server_name = server_dir.name
        servers.add(server_name)

        with open(agg_path) as f:
            agg = json.load(f)

        # For single experiment, x-axis = 1 (or we can read total_connections from iteration)
        # Try to read from first iteration summary
        x_val = 1
        for iter_dir in sorted(server_dir.iterdir()):
            if not iter_dir.is_dir():
                continue
            summary_path = iter_dir / "summary.json"
            if summary_path.exists():
                with open(summary_path) as f:
                    summary = json.load(f)
                x_val = summary.get("total_connections", 1)
                break

        data["rps"][server_name].append((x_val, agg["total_rps_mean"]))
        data["set_rps"][server_name].append((x_val, agg.get("set_rps_mean", 0)))
        data["get_rps"][server_name].append((x_val, agg.get("get_rps_mean", 0)))
        data["cpu"][server_name].append((x_val, agg.get("avg_cpu_percent", 0)))

        for pct in ["avg", "p50", "p95", "p99"]:
            set_val = agg.get("set_latency", {}).get(pct)
            get_val = agg.get("get_latency", {}).get(pct)
            if set_val is not None:
                data["set_latency"][server_name][pct].append((x_val, set_val))
            if get_val is not None:
                data["get_latency"][server_name][pct].append((x_val, get_val))

        per_iter = agg.get("per_iteration_rps", [])
        outliers = agg.get("outlier_indices", [])
        if per_iter:
            data["scatter"][server_name].append((x_val, per_iter, outliers))

    return data, None, sorted(servers)


# ═══════════════════════════════════════════════════════════════════════════════
# Color Configuration
# ═══════════════════════════════════════════════════════════════════════════════

# Pre-defined color palette for server binaries
SERVER_COLORS = [
    "#E53935",  # red
    "#1E88E5",  # blue
    "#43A047",  # green
    "#FB8C00",  # orange
    "#8E24AA",  # purple
    "#00ACC1",  # cyan
    "#D81B60",  # pink
    "#5C6BC0",  # indigo
    "#F4511E",  # deep orange
    "#7CB342",  # light green
]


def assign_colors(servers: list) -> dict:
    """Assign colors to server names."""
    return {s: SERVER_COLORS[i % len(SERVER_COLORS)] for i, s in enumerate(servers)}


# ═══════════════════════════════════════════════════════════════════════════════
# HTML Generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_html(data: dict, servers: list, manifest: dict,
                  title_prefix: str, reference: str, output_path: Path,
                  warmup_skip: float, x_label: str):
    """Generate self-contained interactive HTML with Plotly.js."""
    colors = assign_colors(servers)

    all_x = sorted(set(
        x for server in data["rps"] for x, _ in data["rps"][server]
    ))

    charts = []

    # Helper to build standard xaxis config
    def xaxis_cfg():
        return {"title": x_label, "type": "log" if len(all_x) > 3 else "linear",
                "tickvals": all_x, "ticktext": [str(c) for c in all_x]}

    # ─── 1. RPS Scalability ───────────────────────────────────────────────
    traces = []
    for server in servers:
        points = data["rps"].get(server, [])
        if not points:
            continue
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        traces.append({
            "x": x, "y": y, "type": "scatter", "mode": "lines+markers",
            "name": server, "line": {"color": colors[server], "width": 2.5},
            "marker": {"size": 8, "color": colors[server]},
            "hovertemplate": f"<b>{server}</b><br>{x_label}: %{{x}}<br>RPS: %{{y:,.0f}}<extra></extra>",
        })

    layout = {
        "title": {"text": "RPS Scalability — Total Workload Throughput", "font": {"size": 18}},
        "xaxis": xaxis_cfg(),
        "yaxis": {"title": "Requests per Second (RPS)", "rangemode": "tozero", "separatethousands": True},
        "legend": {"groupclick": "toggleitem"}, "hovermode": "closest",
        "template": "plotly_white", "height": 600,
    }
    charts.append({"traces": traces, "layout": layout, "id": "rps-scalability",
                    "_metric": "higher_better", "_unit": "RPS", "_fmt": ",.0f"})

    # ─── 2. SET Latency Charts ───────────────────────────────────────────
    for pct in ["avg", "p50", "p95", "p99"]:
        traces = []
        has_data = False
        for server in servers:
            points = data["set_latency"].get(server, {}).get(pct, [])
            if not points:
                continue
            has_data = True
            x = [p[0] for p in points]
            y = [p[1] for p in points]
            traces.append({
                "x": x, "y": y, "type": "scatter", "mode": "lines+markers",
                "name": server, "line": {"color": colors[server], "width": 2.5},
                "marker": {"size": 8, "color": colors[server]},
                "hovertemplate": f"<b>{server}</b><br>{x_label}: %{{x}}<br>{pct}: %{{y:.3f}} ms<extra></extra>",
            })
        if has_data:
            layout = {
                "title": {"text": f"SET Latency — {pct.upper()}", "font": {"size": 18}},
                "xaxis": {"title": x_label, "type": "log" if len(all_x) > 3 else "linear",
                          "tickvals": all_x, "ticktext": [str(c) for c in all_x]},
                "yaxis": {"title": "Latency (ms)", "rangemode": "tozero"},
                "legend": {"groupclick": "toggleitem"}, "hovermode": "closest",
                "template": "plotly_white", "height": 500,
            }
            charts.append({"traces": traces, "layout": layout, "id": f"set-latency-{pct}",
                            "_metric": "lower_better", "_unit": f"SET {pct} (ms)", "_fmt": ".3f"})

    # ─── 3. GET Latency Charts ───────────────────────────────────────────
    for pct in ["avg", "p50", "p95", "p99"]:
        traces = []
        has_data = False
        for server in servers:
            points = data["get_latency"].get(server, {}).get(pct, [])
            if not points:
                continue
            has_data = True
            x = [p[0] for p in points]
            y = [p[1] for p in points]
            traces.append({
                "x": x, "y": y, "type": "scatter", "mode": "lines+markers",
                "name": server, "line": {"color": colors[server], "width": 2.5},
                "marker": {"size": 8, "color": colors[server]},
                "hovertemplate": f"<b>{server}</b><br>{x_label}: %{{x}}<br>{pct}: %{{y:.3f}} ms<extra></extra>",
            })
        if has_data:
            layout = {
                "title": {"text": f"GET Latency — {pct.upper()}", "font": {"size": 18}},
                "xaxis": {"title": x_label, "type": "log" if len(all_x) > 3 else "linear",
                          "tickvals": all_x, "ticktext": [str(c) for c in all_x]},
                "yaxis": {"title": "Latency (ms)", "rangemode": "tozero"},
                "legend": {"groupclick": "toggleitem"}, "hovermode": "closest",
                "template": "plotly_white", "height": 500,
            }
            charts.append({"traces": traces, "layout": layout, "id": f"get-latency-{pct}",
                            "_metric": "lower_better", "_unit": f"GET {pct} (ms)", "_fmt": ".3f"})

    # ─── 4. CPU Usage ─────────────────────────────────────────────────────
    cpu_traces = []
    has_cpu = False
    for server in servers:
        points = data["cpu"].get(server, [])
        if not points or all(v == 0 for _, v in points):
            continue
        has_cpu = True
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        cpu_traces.append({
            "x": x, "y": y, "type": "scatter", "mode": "lines+markers",
            "name": server, "line": {"color": colors[server], "width": 2.5},
            "marker": {"size": 8, "color": colors[server]},
            "hovertemplate": f"<b>{server}</b><br>{x_label}: %{{x}}<br>CPU: %{{y:.1f}}%<extra></extra>",
        })
    if has_cpu:
        layout = {
            "title": {"text": "System CPU Usage — Experiment Average", "font": {"size": 18}},
            "xaxis": {"title": x_label, "type": "log" if len(all_x) > 3 else "linear",
                      "tickvals": all_x, "ticktext": [str(c) for c in all_x]},
            "yaxis": {"title": "CPU Usage (%)", "rangemode": "tozero"},
            "legend": {"groupclick": "toggleitem"}, "hovermode": "closest",
            "template": "plotly_white", "height": 500,
        }
        charts.append({"traces": cpu_traces, "layout": layout, "id": "cpu-usage",
                        "_metric": "lower_better", "_unit": "CPU %", "_fmt": ".1f"})

    # ─── 5. RPS/CPU Efficiency ────────────────────────────────────────────
    if has_cpu:
        eff_traces = []
        for server in servers:
            rps_points = {x: y for x, y in data["rps"].get(server, [])}
            cpu_points = {x: y for x, y in data["cpu"].get(server, [])}
            common_x = sorted(set(rps_points) & set(cpu_points))
            if not common_x:
                continue
            x = []
            y = []
            for xv in common_x:
                cpu_val = cpu_points[xv]
                if cpu_val > 0:
                    x.append(xv)
                    y.append(round(rps_points[xv] / cpu_val, 1))
            if x:
                eff_traces.append({
                    "x": x, "y": y, "type": "scatter", "mode": "lines+markers",
                    "name": server, "line": {"color": colors[server], "width": 2.5},
                    "marker": {"size": 8, "color": colors[server]},
                    "hovertemplate": f"<b>{server}</b><br>{x_label}: %{{x}}<br>RPS/CPU%: %{{y:,.0f}}<extra></extra>",
                })
        if eff_traces:
            layout = {
                "title": {"text": "CPU Efficiency — RPS per CPU%", "font": {"size": 18}},
                "xaxis": {"title": x_label, "type": "log" if len(all_x) > 3 else "linear",
                          "tickvals": all_x, "ticktext": [str(c) for c in all_x]},
                "yaxis": {"title": "RPS / CPU%", "rangemode": "tozero", "separatethousands": True},
                "legend": {"groupclick": "toggleitem"}, "hovermode": "closest",
                "template": "plotly_white", "height": 500,
            }
            charts.append({"traces": eff_traces, "layout": layout, "id": "cpu-efficiency",
                            "_metric": "higher_better", "_unit": "RPS/CPU%", "_fmt": ",.0f"})

    # ─── Build HTML ───────────────────────────────────────────────────────
    title = "Valkey Server Benchmark Results"
    if title_prefix:
        title = f"{title_prefix} — {title}"

    # Info section
    info_parts = []
    if manifest:
        info_parts.append(f"Description: {manifest.get('description', '')}")
        info_parts.append(f"Iterations: {manifest.get('iterations', '?')}")
        if manifest.get("set_ratio") is not None:
            info_parts.append(
                f"SET/GET ratio: {manifest['set_ratio']}/{manifest['get_ratio']}")
        if manifest.get("connections_per_process"):
            info_parts.append(
                f"Connections per process: {manifest['connections_per_process']}")
    if warmup_skip > 0:
        info_parts.append(f"Warmup skip: {warmup_skip}s (first {warmup_skip}s discarded)")

    info_html = "<br>".join(info_parts) if info_parts else ""

    chart_divs = "\n".join(
        f'    <div class="chart-container" id="container-{c["id"]}">\n'
        f'        <div id="{c["id"]}" style="width:100%;"></div>\n'
        f'    </div>'
        for c in charts
    )

    # Build initial plot scripts
    chart_scripts = ""
    for c in charts:
        t = json.dumps(c["traces"], indent=None)
        l = json.dumps(c["layout"], indent=None)
        chart_scripts += f'    Plotly.newPlot("{c["id"]}", {t}, {l}, {{responsive: true}});\n'

    # Build chart metadata JS object (for switchMode)
    chart_meta_list = []
    for c in charts:
        meta = {
            "id": c["id"],
            "metric": c.get("_metric", "skip"),
            "unit": c.get("_unit", ""),
            "fmt": c.get("_fmt", ""),
            "traces": c["traces"],
            "layout": c["layout"],
        }
        chart_meta_list.append(meta)
    chart_meta_json = json.dumps(chart_meta_list, indent=None)

    # Server options for dropdown
    server_options = "\n".join(
        f'            <option value="{s}">{s}</option>' for s in servers
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0; padding: 20px; background: #fafafa; color: #333;
        }}
        .header {{
            max-width: 1400px; margin: 0 auto 20px; padding: 20px 30px;
            background: #fff; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .header h1 {{ margin: 0 0 10px; font-size: 24px; color: #1a1a1a; }}
        .header .meta {{ font-size: 14px; color: #666; line-height: 1.6; }}
        .header .meta .stat {{
            display: inline-block; background: #f0f4f8; padding: 2px 10px;
            border-radius: 4px; margin-right: 8px; font-family: monospace;
        }}
        .chart-container {{
            max-width: 1400px; margin: 0 auto 20px; padding: 15px;
            background: #fff; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .controls {{
            max-width: 1400px; margin: 0 auto 20px; padding: 16px 24px;
            background: #fff; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
        }}
        .controls label {{
            font-size: 14px; font-weight: 600; color: #333;
        }}
        .controls select {{
            font-size: 14px; padding: 6px 12px; border: 1px solid #ccc;
            border-radius: 6px; background: #fff; color: #333;
            cursor: pointer; min-width: 200px;
        }}
        .controls select:focus {{
            outline: none; border-color: #1E88E5; box-shadow: 0 0 0 2px rgba(30,136,229,0.2);
        }}
        .controls .mode-badge {{
            font-size: 12px; padding: 4px 10px; border-radius: 12px;
            font-weight: 600; letter-spacing: 0.5px;
        }}
        .controls .mode-badge.absolute {{
            background: #e8f5e9; color: #2e7d32;
        }}
        .controls .mode-badge.comparison {{
            background: #fff3e0; color: #e65100;
        }}
        .legend-help {{
            max-width: 1400px; margin: 0 auto 20px; padding: 12px 20px;
            background: #e3f2fd; border-radius: 6px; font-size: 13px; color: #1565c0;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{title}</h1>
        <div class="meta">
            <p>Servers: {', '.join(f'<span class="stat">{s}</span>' for s in servers)}</p>
            {f'<p>{info_html}</p>' if info_html else ''}
            <p>
                Outlier detection: 4-method consensus (Modified Z-Score + IQR + %Deviation + Grubbs, ≥2 agree).
            </p>
        </div>
    </div>

    <div class="controls">
        <label for="compare-select">Compare to:</label>
        <select id="compare-select" onchange="switchMode(this.value)">
            <option value="none">None (absolute values)</option>
{server_options}
        </select>
        <span id="mode-badge" class="mode-badge absolute">ABSOLUTE</span>
        <span id="mode-description" style="font-size: 13px; color: #666;">
            Showing raw values for all metrics
        </span>
    </div>

    <div class="legend-help">
        💡 <b>Tip:</b> Click any server in the legend to toggle. Double-click to isolate.
        Use toolbar to zoom, pan, download as PNG.
        Use the <b>Compare to</b> dropdown to switch between absolute values and % delta comparison mode.
    </div>

{chart_divs}

    <script>
    // ═══ Chart metadata (traces, layouts, metric types) ═══
    var chartMeta = {chart_meta_json};

    // ═══ Initial render (absolute mode) ═══
{chart_scripts}

    // ═══ Mode switching logic ═══
    function switchMode(ref) {{
        var badge = document.getElementById('mode-badge');
        var desc = document.getElementById('mode-description');

        if (ref === 'none') {{
            // Restore absolute mode
            badge.textContent = 'ABSOLUTE';
            badge.className = 'mode-badge absolute';
            desc.textContent = 'Showing raw values for all metrics';

            chartMeta.forEach(function(cm) {{
                var el = document.getElementById(cm.id);
                if (!el) return;
                // Show the chart container (in case it was hidden)
                var container = document.getElementById('container-' + cm.id);
                if (container) container.style.display = '';
                Plotly.react(cm.id, cm.traces, cm.layout, {{responsive: true}});
            }});
            return;
        }}

        // Comparison mode
        badge.textContent = '% vs ' + ref;
        badge.className = 'mode-badge comparison';
        desc.textContent = 'Showing % difference relative to ' + ref +
            '. For RPS/efficiency: positive = better. For latency/CPU: negative = better.';

        chartMeta.forEach(function(cm) {{
            var container = document.getElementById('container-' + cm.id);
            if (!container) return;

            // Skip charts that don't support comparison (scatter, existing delta)
            if (cm.metric === 'skip' || cm.id === 'rps-delta') {{
                container.style.display = 'none';
                return;
            }}
            container.style.display = '';

            // Find reference trace data: build x->y lookup from the reference server's trace
            var refTrace = null;
            cm.traces.forEach(function(t) {{
                if (t.name === ref) refTrace = t;
            }});

            if (!refTrace) {{
                // Reference not in this chart — hide it
                container.style.display = 'none';
                return;
            }}

            // Build x->y map for reference
            var refMap = {{}};
            for (var i = 0; i < refTrace.x.length; i++) {{
                refMap[refTrace.x[i]] = refTrace.y[i];
            }}

            // Build delta traces
            var deltaTraces = [];
            cm.traces.forEach(function(origTrace) {{
                if (origTrace.name === ref) return; // skip reference itself

                var dx = [], dy = [];
                for (var i = 0; i < origTrace.x.length; i++) {{
                    var xv = origTrace.x[i];
                    var yv = origTrace.y[i];
                    var rv = refMap[xv];
                    if (rv !== undefined && rv !== 0) {{
                        var pct = ((yv - rv) / rv) * 100;
                        dx.push(xv);
                        dy.push(Math.round(pct * 100) / 100);
                    }}
                }}
                if (dx.length > 0) {{
                    var newTrace = JSON.parse(JSON.stringify(origTrace));
                    newTrace.x = dx;
                    newTrace.y = dy;
                    // Update hover template
                    newTrace.hovertemplate =
                        '<b>' + origTrace.name + '</b><br>' +
                        cm.layout.xaxis.title + ': %{{x}}<br>' +
                        'Delta: %{{y:+.1f}}%<extra></extra>';
                    deltaTraces.push(newTrace);
                }}
            }});

            // Build delta layout
            var deltaLayout = JSON.parse(JSON.stringify(cm.layout));
            var origTitle = cm.layout.title.text || cm.layout.title;
            deltaLayout.title = {{
                text: origTitle + ' — % vs ' + ref,
                font: {{ size: 18 }}
            }};
            deltaLayout.yaxis = {{
                title: '% vs ' + ref,
                zeroline: true,
                zerolinecolor: '#888',
                zerolinewidth: 2
            }};
            // Remove rangemode:tozero and separatethousands for delta
            delete deltaLayout.yaxis.rangemode;
            delete deltaLayout.yaxis.separatethousands;
            // Add zero-line shape
            deltaLayout.shapes = [{{
                type: 'line', x0: 0, x1: 1, y0: 0, y1: 0,
                xref: 'paper', yref: 'y',
                line: {{ color: '#888', width: 1.5, dash: 'dash' }}
            }}];

            Plotly.react(cm.id, deltaTraces, deltaLayout, {{responsive: true}});
        }});
    }}
    </script>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)

    print(f"Generated: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate interactive HTML graphs from valkey-server benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Input layouts (auto-detected):
  Matrix: <dir>/_manifest.json + <dir>/<N>-clients/<server>/aggregate.json
  Single: <dir>/<server>/aggregate.json

Examples:
  python generate_server_graphs.py results/server-matrix/2026-03-23T20:00:00/
  python generate_server_graphs.py results/ --output graphs/ --title "m5.metal"
  python generate_server_graphs.py results/ --reference valkey-8.1-oss
""",
    )
    parser.add_argument(
        "results_dir",
        help="Directory containing benchmark results",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output HTML file path (default: <results_dir>/server_benchmark.html)",
    )
    parser.add_argument(
        "--title", "-t",
        default="",
        help="Title prefix for the report",
    )
    parser.add_argument(
        "--reference", "-r",
        default=None,
        help="Reference server for delta charts (default: first server)",
    )
    parser.add_argument(
        "--warmup-skip",
        type=float,
        default=5,
        help="Seconds of initial data already skipped (for display only, default: 5)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)

    if not results_dir.is_dir():
        print(f"Error: {results_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Detect layout
    manifest_path = results_dir / "_manifest.json"
    if manifest_path.exists():
        print(f"Loading matrix data from: {results_dir}")
        data, manifest, servers = load_matrix_data(results_dir)
        x_label = manifest.get("x_axis", "Total Clients") if manifest else "Total Clients"
        # Make x_label human-friendly
        x_label = x_label.replace("_", " ").title()
    else:
        print(f"Loading single experiment from: {results_dir}")
        data, manifest, servers = load_single_experiment(results_dir)
        x_label = "Total Connections"

    if not servers:
        print("Error: No server data found", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(servers)} servers: {', '.join(servers)}")

    # Reference server
    reference = args.reference or servers[0]
    if reference not in servers:
        print(f"Warning: reference '{reference}' not found. Using '{servers[0]}'",
              file=sys.stderr)
        reference = servers[0]

    # Output path
    if args.output:
        output_path = Path(args.output)
        if output_path.is_dir() or str(output_path).endswith("/"):
            output_path = output_path / "server_benchmark.html"
    else:
        output_path = results_dir / "server_benchmark.html"

    generate_html(data, servers, manifest, args.title, reference,
                  output_path, args.warmup_skip, x_label)

    print(f"\nDone. Open in browser: {output_path}")


if __name__ == "__main__":
    main()
