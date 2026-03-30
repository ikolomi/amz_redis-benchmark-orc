#!/usr/bin/env python3
"""
run_server_matrix.py — Matrix-based benchmark sweep for valkey-server binaries.

Sweeps a "scale" dimension across multiple orchestrator runs. The template
defines base set_clients and get_clients (connection counts), and each scale
multiplier creates a proportionally larger experiment.

Output structure:
    <output-dir>/<timestamp>/
        _manifest.json          # Matrix metadata for graph generator
        <total_conns>-clients/
            <server-name>/
                aggregate.json  # Per-server aggregate from orchestrator

Usage:
    python run_server_matrix.py \\
        --matrix configs/examples/matrix-scalability.json

    python run_server_matrix.py \\
        --matrix configs/examples/matrix-scalability.json \\
        --dry-run
"""

import argparse
import copy
import json
import math
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════════
# Config Parsing
# ═══════════════════════════════════════════════════════════════════════════════

def load_matrix_config(matrix_path: str) -> dict:
    """Load and validate matrix configuration JSON."""
    with open(matrix_path) as f:
        config = json.load(f)

    required = ["description", "experiment_template", "dimensions"]
    for field in required:
        if field not in config:
            raise ValueError(f"Missing required field: '{field}'")

    dims = config["dimensions"]
    if "scale" not in dims:
        raise ValueError("dimensions must contain 'scale'")

    template = config["experiment_template"]
    clients = template.get("clients", {})
    for req in ["max_connections_per_benchmark_process", "set_clients", "get_clients"]:
        if req not in clients:
            raise ValueError(f"experiment_template.clients must contain '{req}'")

    return config


def generate_experiment_config(template: dict, scale: int) -> dict:
    """Generate a complete experiment config from template × scale multiplier.

    Multiplies set_clients and get_clients by scale.
    """
    config = copy.deepcopy(template)
    clients = config["clients"]

    clients["set_clients"] = clients["set_clients"] * scale
    clients["get_clients"] = clients["get_clients"] * scale

    total_conns = clients["set_clients"] + clients["get_clients"]
    config.setdefault("description",
                      f"Matrix run: scale={scale}, "
                      f"{clients['set_clients']} SET + {clients['get_clients']} GET = "
                      f"{total_conns} connections")

    return config


# ═══════════════════════════════════════════════════════════════════════════════
# Manifest
# ═══════════════════════════════════════════════════════════════════════════════

def write_manifest(output_dir: Path, matrix_config: dict, results: dict):
    """Write _manifest.json describing the matrix run."""
    template = matrix_config["experiment_template"]
    base_set = template["clients"]["set_clients"]
    base_get = template["clients"]["get_clients"]

    manifest = {
        "description": matrix_config["description"],
        "x_axis": "total_connections",
        "iterations": template.get("iterations", 1),
        "servers": [s["name"] for s in template["servers"]],
        "dimensions": matrix_config["dimensions"],
        "base_set_clients": base_set,
        "base_get_clients": base_get,
        "max_connections_per_process": template["clients"]["max_connections_per_benchmark_process"],
        "data_points": {},
    }

    for total_conns, server_results in sorted(results.items()):
        manifest["data_points"][str(total_conns)] = {
            "total_connections": total_conns,
            "servers": server_results,
        }

    manifest_path = output_dir / "_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote manifest: {manifest_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Orchestrator Invocation
# ═══════════════════════════════════════════════════════════════════════════════

def run_orchestrator(experiment_config: dict, experiment_output_dir: Path) -> dict:
    """Invoke the orchestrator for a single data point."""
    script_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(script_dir))

    from orchestrator import load_config, run_experiment

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix="matrix-exp-", delete=False
    )
    try:
        json.dump(experiment_config, tmp)
        tmp.close()

        config = load_config(tmp.name)
        experiment_output_dir.mkdir(parents=True, exist_ok=True)
        aggregates = run_experiment(config, experiment_output_dir)

        server_results = {}
        for server_name, agg in aggregates.items():
            server_results[server_name] = {
                "total_rps_mean": agg.total_rps_mean,
                "total_rps_stdev": agg.total_rps_stdev,
                "total_rps_cov": agg.total_rps_cov,
                "set_rps_mean": agg.set_rps_mean,
                "get_rps_mean": agg.get_rps_mean,
                "set_latency": agg.set_latency,
                "get_latency": agg.get_latency,
                "avg_cpu_percent": agg.avg_cpu_percent,
                "iterations_kept": agg.iterations_kept,
                "iterations_total": agg.iterations_total,
                "per_iteration_rps": agg.per_iteration_rps,
                "outlier_indices": agg.outlier_indices,
            }
        return server_results
    finally:
        os.unlink(tmp.name)


# ═══════════════════════════════════════════════════════════════════════════════
# Main Orchestration Loop
# ═══════════════════════════════════════════════════════════════════════════════

def run_matrix(matrix_config: dict, output_dir: Path):
    """Main loop: for each scale value, generate config and run."""
    template = matrix_config["experiment_template"]
    scale_values = matrix_config["dimensions"]["scale"]
    base_set = template["clients"]["set_clients"]
    base_get = template["clients"]["get_clients"]
    max_c = template["clients"]["max_connections_per_benchmark_process"]

    print("=" * 70)
    print("Valkey Server Benchmark Matrix")
    print(f"  Description: {matrix_config['description']}")
    print(f"  Scale: {scale_values}")
    print(f"  Base: {base_set} SET + {base_get} GET = {base_set + base_get} connections")
    print(f"  Max connections per process: {max_c}")
    print(f"  Servers: {', '.join(s['name'] for s in template['servers'])}")
    print(f"  Iterations: {template.get('iterations', 1)}")
    print(f"  Duration: {template['experiment_duration_seconds']}s per iteration")
    print(f"  Output: {output_dir}")

    total_runs = len(scale_values) * len(template["servers"]) * template.get("iterations", 1)
    est_time = total_runs * template["experiment_duration_seconds"] / 60
    print(f"  Total runs: {total_runs}")
    print(f"  Estimated time: ~{est_time:.0f} min")

    print(f"\n  Per scale point:")
    for s in scale_values:
        sc = base_set * s
        gc = base_get * s
        tc = sc + gc
        sp = max(1, math.ceil(sc / max_c)) if sc > 0 else 0
        gp = max(1, math.ceil(gc / max_c)) if gc > 0 else 0
        print(f"    ×{s}: {sc} SET + {gc} GET = {tc} connections "
              f"({sp} SET procs + {gp} GET procs)")

    print("=" * 70)

    all_results = {}

    for i, scale in enumerate(scale_values):
        set_c = base_set * scale
        get_c = base_get * scale
        total_conns = set_c + get_c

        print(f"\n{'═' * 70}")
        print(f"Matrix point {i+1}/{len(scale_values)}: "
              f"scale=×{scale}, {set_c} SET + {get_c} GET = {total_conns} connections")
        print(f"{'═' * 70}")

        experiment_config = generate_experiment_config(template, scale)
        experiment_config["output_directory"] = str(output_dir)

        point_dir = output_dir / f"{total_conns}-clients"

        try:
            server_results = run_orchestrator(experiment_config, point_dir)
            all_results[total_conns] = server_results

            for sname, sdata in server_results.items():
                print(f"  {sname}: {sdata['total_rps_mean']:.0f} RPS "
                      f"(±{sdata['total_rps_stdev']:.0f}, "
                      f"CoV {sdata['total_rps_cov']:.1f}%), "
                      f"CPU {sdata['avg_cpu_percent']:.1f}%")
        except Exception as e:
            print(f"\nERROR: Scale ×{scale} failed: {e}", file=sys.stderr)
            raise

    write_manifest(output_dir, matrix_config, all_results)

    print("\n" + "=" * 70)
    print("Matrix benchmark completed!")
    print(f"Results in: {output_dir}")
    print(f"Generate graphs with:")
    print(f"  python generate_server_graphs.py {output_dir}")
    print("=" * 70)

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run valkey-server benchmarks across a scale sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_server_matrix.py --matrix configs/examples/matrix-scalability.json
  python run_server_matrix.py --matrix matrix.json --dry-run
  python run_server_matrix.py --matrix matrix.json --iterations 3
""",
    )
    parser.add_argument("--matrix", "-m", required=True, help="Matrix config JSON")
    parser.add_argument("--output-dir", "-o", default=None, help="Override output dir")
    parser.add_argument("--iterations", type=int, default=None, help="Override iterations")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without executing")
    return parser.parse_args()


def main():
    args = parse_args()
    matrix_config = load_matrix_config(args.matrix)

    if args.iterations:
        matrix_config["experiment_template"]["iterations"] = args.iterations

    base_output = args.output_dir or matrix_config["experiment_template"]["output_directory"]
    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    output_dir = Path(base_output) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "matrix-config.json").write_text(
        json.dumps(matrix_config, indent=2) + "\n"
    )

    template = matrix_config["experiment_template"]
    scale_values = matrix_config["dimensions"]["scale"]
    base_set = template["clients"]["set_clients"]
    base_get = template["clients"]["get_clients"]
    max_c = template["clients"]["max_connections_per_benchmark_process"]

    if args.dry_run:
        print("=" * 70)
        print("DRY RUN — Matrix Benchmark Plan")
        print(f"  Description: {matrix_config['description']}")
        print(f"  Servers:")
        for srv in template["servers"]:
            print(f"    - {srv['name']}: {srv['binary']}")
        print(f"  Scale: {scale_values}")
        print(f"  Base: {base_set} SET + {base_get} GET = {base_set + base_get} connections")
        print(f"  Max connections per process: {max_c}")
        print(f"  Iterations: {template.get('iterations', 1)}")
        print(f"  Duration: {template['experiment_duration_seconds']}s per iteration")

        print(f"\n  Per scale point:")
        for s in scale_values:
            sc = base_set * s
            gc = base_get * s
            tc = sc + gc
            sp = max(1, math.ceil(sc / max_c)) if sc > 0 else 0
            gp = max(1, math.ceil(gc / max_c)) if gc > 0 else 0
            scp = math.ceil(sc / sp) if sp > 0 else 0
            gcp = math.ceil(gc / gp) if gp > 0 else 0
            print(f"    ×{s}: {sc} SET + {gc} GET = {tc} connections "
                  f"({sp} SET procs ×{scp}c + {gp} GET procs ×{gcp}c)")

        total_runs = len(scale_values) * len(template["servers"]) * template.get("iterations", 1)
        est_time = total_runs * template["experiment_duration_seconds"] / 60
        print(f"\n  Total runs: {total_runs}")
        print(f"  Estimated time: ~{est_time:.0f} min")
        print(f"  Output: {output_dir}")
        print("=" * 70)
        return

    from orchestrator import setup_logging
    setup_logging(log_file=output_dir / "matrix-runner.log")
    run_matrix(matrix_config, output_dir)


if __name__ == "__main__":
    main()
