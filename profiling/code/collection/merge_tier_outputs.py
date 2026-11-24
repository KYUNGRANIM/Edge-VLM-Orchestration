"""
Step 2: Collection - Build unified oracle dataset from 3 tiers.

Inputs (raw measurement files):
    tier1_m5/m5_<disaster>.jsonl             : M5_A/B/C outputs
    tier2_pi/pi_<disaster>.jsonl             : Pi_A/B merged outputs
    tier2_pi/pi_a_v18_<disaster>.jsonl       : Pi_A original output with Pi_A conf_detail
    tier3_jetson/jetson_<disaster>.jsonl     : Jetson output

Output:
    collection.jsonl - one record per image_id, with:
      * Per-tier outputs (text, cat, sev, conf, conf_detail logprobs)
      * Per-stage breakdown (latency, overhead, energy)
      * Cumulative end-to-end cost for each routing destination
        end_m5     = M5 alone
        end_pi     = M5 + (M5->Pi http) + Pi
        end_jetson = M5 + (M5->Pi http) + Pi + (Pi->Jet http) + Jetson

Verification at the end:
    - cumulative cost identities (cascade integrity)
    - power = measured_w (constant per tier)
    - energy = power_w * latency_s
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Inputs. The final 921 x 6 collection is split by disaster type.
#oracle
#M5_GLOB        = "tier1_m5/*.jsonl"
#PI_GLOB        = "tier2_pi/pi_*.jsonl"
#PI_A_ORIG_GLOB = "tier2_pi/pi_a_v18_*.jsonl"
#JET_GLOB       = "tier3_jetson/jetson_*.jsonl"

M5_GLOB        = "online_oracle_result_v2/online_result_m5.jsonl"
PI_GLOB        = "online_oracle_result_v2/online_result_pi.jsonl"
PI_A_ORIG_GLOB = "online_oracle_result_v2/pi_a_online_routing_v2.jsonl"
JET_GLOB       = "online_oracle_result_v2/online_result_jetson.jsonl"



# Output
OUT_PATH = ROOT / "collection.jsonl"

# Measured power values (averaged over each tier's runs).
# Used as fallback if a per-record power is missing.
M5_POWER_W     = 1.2154
PI_POWER_W     = 7.5564
JETSON_POWER_W = 11.199


def include_by_status(rec: dict, allowed_status: set[str] | None) -> bool:
    if allowed_status is None:
        return True
    return rec.get("status") in allowed_status


def load_map(path: Path, allowed_status: set[str] | None = None) -> dict:
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not include_by_status(rec, allowed_status):
                continue
            iid = rec.get("image_id")
            if iid:
                out[iid] = rec
    return out


def load_many_map(paths: list[Path], allowed_status: set[str] | None = None) -> dict:
    out = {}
    duplicate_ids = set()
    for path in paths:
        cur = load_map(path, allowed_status=allowed_status)
        overlap = set(out) & set(cur)
        duplicate_ids.update(overlap)
        out.update(cur)
    if duplicate_ids:
        print(f"WARNING: {len(duplicate_ids)} duplicate image_id values while loading {paths[0].parent}")
    return out


def load_go_jetson(paths: list[Path]) -> dict:
    """Pi-side oracle file carries the Pi->Jet handoff metrics under 'go_jetson'."""
    out = {}
    for path in paths:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                iid = rec.get("image_id")
                gj = rec.get("go_jetson")
                if iid and gj and gj.get("status") == "ok":
                    out[iid] = gj
    return out


def first(*values):
    for v in values:
        if v is not None:
            return v
    return None


def fnum(v, default=0):
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def inum(v, default=0):
    if v is None:
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def median_int(values: list[int], default: int = 0) -> int:
    if not values:
        return default
    return int(round(statistics.median(values)))


def build_pi_jet_impute_defaults(pi_a_origin_data: dict) -> dict:
    """Fallback for rare Pi->Jet handoff failures after Jetson retry collection."""
    http_vals = []
    rx_vals = []
    tx_deltas = []
    for rec in pi_a_origin_data.values():
        gj = rec.get("go_jetson") or {}
        if gj.get("status") != "ok":
            continue
        pi_jet_tx = inum(gj.get("pi_jet_tx_bytes"))
        if pi_jet_tx <= 0:
            continue
        end_pi_overhead = rec.get("end_pi", {}).get("cost", {}).get("signal_overhead_bytes", {})
        m5_pi_tx = inum(end_pi_overhead.get("m5_pi_tx_bytes"))
        http_vals.append(inum(gj.get("pi_jet_http_e2e_ms")))
        rx_vals.append(inum(gj.get("pi_jet_rx_bytes")))
        if m5_pi_tx > pi_jet_tx:
            tx_deltas.append(m5_pi_tx - pi_jet_tx)
    return {
        "pi_jet_http_ms": median_int(http_vals, default=13),
        "pi_jet_rx_bytes": median_int(rx_vals, default=118),
        "m5_pi_minus_pi_jet_tx_bytes": median_int(tx_deltas, default=279),
    }


def build_record(iid: str, m: dict, p: dict, po: dict, j: dict, gj: dict, impute_defaults: dict) -> dict:
    # ---- Tier outputs --------------------------------------------------
    m5_a = m["artifacts"]["m5_a"]
    m5_b = m["artifacts"]["m5_b"]
    m5_c = m["artifacts"]["m5_c"]
    pi_a = {k: p["pi_a"][k] for k in ("text", "cat", "sev", "conf")}
    pi_b = {k: p["pi_b"][k] for k in ("text", "cat", "sev", "conf")}
    jetson_out = j["end_jetson"]["output"]

    # ---- Per-stage measured breakdown ----------------------------------
    m5_cost   = m["end_m5"]["cost"]
    m5_lat    = m5_cost["latency_ms"]
    pi_costs  = p["pi_a"]["costs"]
    j_cost    = j["end_jetson"]["cost"]
    j_lat     = j_cost["latency_ms"]
    j_overhd  = j_cost["signal_overhead_bytes"]

    # Latency per stage (ms)
    m5_total_ms       = inum(first(m5_lat.get("m5_total_ms"), m5_lat.get("total")))
    m5_pi_http_ms     = inum(first(pi_costs.get("m5_pi_http_e2e_ms"),
                                   m.get("go_pi", {}).get("m5_pi_http_e2e_ms"),
                                   j_lat.get("m5_pi_http_e2e_ms")))
    pi_total_ms       = inum(pi_costs.get("pi_total_ms"))
    pi_jet_http_ms    = inum(first(gj.get("pi_jet_http_e2e_ms"),
                                   j_lat.get("pi_jet_http_e2e_ms")))
    jetson_total_ms   = inum(first(j["end_jetson"].get("vlm_ms"),
                                   j_lat.get("jetson_total_ms")))

    # Overhead per handoff (bytes)
    m5_pi_tx_bytes    = inum(first(pi_costs.get("m5_pi_tx_bytes"),
                                   m.get("go_pi", {}).get("m5_pi_tx_bytes"),
                                   j_overhd.get("m5_pi_tx_bytes")))
    m5_pi_rx_bytes    = inum(first(pi_costs.get("m5_pi_rx_bytes"),
                                   m.get("go_pi", {}).get("m5_pi_rx_bytes"),
                                   j_overhd.get("m5_pi_rx_bytes")))
    pi_jet_tx_bytes   = inum(first(gj.get("pi_jet_tx_bytes"),
                                   j_overhd.get("pi_jet_tx_bytes")))
    pi_jet_rx_bytes   = inum(first(gj.get("pi_jet_rx_bytes"),
                                   j_overhd.get("pi_jet_rx_bytes")))

    pi_jet_imputed = False
    if pi_jet_http_ms == 0 and pi_jet_tx_bytes == 0 and pi_jet_rx_bytes == 0:
        pi_jet_imputed = True
        pi_jet_http_ms = impute_defaults["pi_jet_http_ms"]
        pi_jet_tx_bytes = max(0, m5_pi_tx_bytes - impute_defaults["m5_pi_minus_pi_jet_tx_bytes"])
        pi_jet_rx_bytes = impute_defaults["pi_jet_rx_bytes"]

    # Power per tier (W) - measured constants, fallback if missing
    m5_power_w     = fnum(first(m5_cost.get("m5_power_w"), m5_cost.get("power_w")), M5_POWER_W)
    pi_power_w     = fnum(first(pi_costs.get("pi_power_w"), pi_costs.get("power_w")), PI_POWER_W)
    jetson_power_w = fnum(first(j_cost.get("jetson_power_w"), j_cost.get("power_w")), JETSON_POWER_W)

    # Per-stage energy (J) = power_w * stage_seconds
    m5_energy_j     = round(m5_power_w     * (m5_total_ms     / 1000.0), 4)
    pi_energy_j     = round(pi_power_w     * (pi_total_ms     / 1000.0), 4)
    jetson_energy_j = round(jetson_power_w * (jetson_total_ms / 1000.0), 4)

    # ---- Cumulative end-to-end cost ------------------------------------
    end_m5_latency     = m5_total_ms
    end_pi_latency     = m5_total_ms + m5_pi_http_ms + pi_total_ms
    end_jetson_latency = end_pi_latency + pi_jet_http_ms + jetson_total_ms

    end_m5_overhead     = 0
    end_pi_overhead     = m5_pi_tx_bytes + m5_pi_rx_bytes
    end_jetson_overhead = end_pi_overhead + pi_jet_tx_bytes + pi_jet_rx_bytes

    end_m5_energy     = m5_energy_j
    end_pi_energy     = round(end_m5_energy + pi_energy_j, 4)
    end_jetson_energy = round(end_pi_energy + jetson_energy_j, 4)

    # ---- Conf_detail (logprob signals for routing) ---------------------
    pi_a_conf_detail = po.get("end_pi", {}).get("conf_detail") or p["pi_a"].get("conf_detail")
    pi_b_conf_detail = p["pi_b"].get("conf_detail")
    jet_conf_detail  = j["end_jetson"].get("conf_detail")
    jet_conf_final   = j["end_jetson"].get("conf_final")

    return {
        "image_id": iid,
        "run_id": first(m.get("run_id"), p.get("run_id"), j.get("run_id")),

        # Tier outputs
        "m5_a": m5_a, "m5_b": m5_b, "m5_c": m5_c,
        "pi_a": pi_a, "pi_b": pi_b,
        "jetson": jetson_out,

        # Logprob details (for routing features)
        "pi_a_conf_detail": pi_a_conf_detail,
        "pi_b_conf_detail": pi_b_conf_detail,
        "jet_conf_detail": jet_conf_detail,
        "jet_conf_final": jet_conf_final,

        # Per-stage breakdown
        "stage_latency_ms": {
            "m5_total":     m5_total_ms,
            "m5_pi_http":   m5_pi_http_ms,
            "pi_total":     pi_total_ms,
            "pi_jet_http":  pi_jet_http_ms,
            "jetson_total": jetson_total_ms,
        },
        "stage_overhead_bytes": {
            "m5_pi_tx":  m5_pi_tx_bytes,
            "m5_pi_rx":  m5_pi_rx_bytes,
            "pi_jet_tx": pi_jet_tx_bytes,
            "pi_jet_rx": pi_jet_rx_bytes,
        },
        "stage_energy_j": {
            "m5":     m5_energy_j,
            "pi":     pi_energy_j,
            "jetson": jetson_energy_j,
        },
        "stage_power_w": {
            "m5":     round(m5_power_w, 4),
            "pi":     round(pi_power_w, 4),
            "jetson": round(jetson_power_w, 4),
        },

        # Cumulative end-to-end cost (the values used by oracle/router)
        "end_m5": {
            "latency_ms":     end_m5_latency,
            "overhead_bytes": end_m5_overhead,
            "energy_j":       end_m5_energy,
        },
        "end_pi": {
            "latency_ms":     end_pi_latency,
            "overhead_bytes": end_pi_overhead,
            "energy_j":       end_pi_energy,
        },
        "end_jetson": {
            "latency_ms":     end_jetson_latency,
            "overhead_bytes": end_jetson_overhead,
            "energy_j":       end_jetson_energy,
        },
        "_flags": {
            "pi_jet_handoff_imputed": pi_jet_imputed,
        },
    }


def verify(records: list[dict]) -> None:
    """Cascade-integrity checks. Print a per-check pass/fail summary."""
    print("\n" + "=" * 70)
    print("VERIFICATION (cumulative cost cascade)")
    print("=" * 70)

    n = len(records)
    fails = {
        "lat_pi":      0, "lat_jet":     0,
        "ovh_pi":      0, "ovh_jet":     0,
        "eng_m5":      0, "eng_pi":      0, "eng_jet":     0,
        "lat_monotonic": 0, "eng_monotonic": 0, "ovh_monotonic": 0,
    }
    energy_diffs = []

    for r in records:
        sl = r["stage_latency_ms"]
        so = r["stage_overhead_bytes"]
        se = r["stage_energy_j"]
        sp = r["stage_power_w"]
        em, ep, ej = r["end_m5"], r["end_pi"], r["end_jetson"]

        # Latency cascade
        if em["latency_ms"] != sl["m5_total"]:
            fails["lat_pi"] += 0  # M5 sanity (no separate counter; rare)
        if ep["latency_ms"] != sl["m5_total"] + sl["m5_pi_http"] + sl["pi_total"]:
            fails["lat_pi"] += 1
        if ej["latency_ms"] != ep["latency_ms"] + sl["pi_jet_http"] + sl["jetson_total"]:
            fails["lat_jet"] += 1

        # Overhead cascade
        if ep["overhead_bytes"] != so["m5_pi_tx"] + so["m5_pi_rx"]:
            fails["ovh_pi"] += 1
        if ej["overhead_bytes"] != ep["overhead_bytes"] + so["pi_jet_tx"] + so["pi_jet_rx"]:
            fails["ovh_jet"] += 1

        # Energy cascade and stage = P*t
        m5_calc = sp["m5"] * (sl["m5_total"] / 1000.0)
        if abs(m5_calc - se["m5"]) > 0.01:
            fails["eng_m5"] += 1
        pi_calc = sp["pi"] * (sl["pi_total"] / 1000.0)
        if abs(pi_calc - se["pi"]) > 0.01:
            fails["eng_pi"] += 1
        jet_calc = sp["jetson"] * (sl["jetson_total"] / 1000.0)
        if abs(jet_calc - se["jetson"]) > 0.01:
            fails["eng_jet"] += 1

        # Monotonic check (cheaper tier <= more expensive tier)
        if not (em["latency_ms"] <= ep["latency_ms"] <= ej["latency_ms"]):
            fails["lat_monotonic"] += 1
        if not (em["energy_j"] <= ep["energy_j"] <= ej["energy_j"]):
            fails["eng_monotonic"] += 1
        if not (em["overhead_bytes"] <= ep["overhead_bytes"] <= ej["overhead_bytes"]):
            fails["ovh_monotonic"] += 1

        energy_diffs.append(abs(em["energy_j"] + r["stage_energy_j"]["pi"] + r["stage_energy_j"]["jetson"] - ej["energy_j"]))

    def report(name: str, c: int):
        ok = c == 0
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name:30s}  {n - c}/{n}")

    report("latency cascade (Pi)",      fails["lat_pi"])
    report("latency cascade (Jetson)",  fails["lat_jet"])
    report("overhead cascade (Pi)",     fails["ovh_pi"])
    report("overhead cascade (Jetson)", fails["ovh_jet"])
    report("energy = P*t (M5)",         fails["eng_m5"])
    report("energy = P*t (Pi)",         fails["eng_pi"])
    report("energy = P*t (Jetson)",     fails["eng_jet"])
    report("monotonic latency  M5<=Pi<=Jet",  fails["lat_monotonic"])
    report("monotonic energy   M5<=Pi<=Jet",  fails["eng_monotonic"])
    report("monotonic overhead M5<=Pi<=Jet",  fails["ovh_monotonic"])

    print(f"\n  max |stage_sum - end_jetson_energy| = {max(energy_diffs):.4f} J")


def summarize(records: list[dict]) -> None:
    print("\n" + "=" * 70)
    print(f"SUMMARY  ({len(records)} records)")
    print("=" * 70)

    # Power constants
    p_m5  = {r["stage_power_w"]["m5"]     for r in records}
    p_pi  = {r["stage_power_w"]["pi"]     for r in records}
    p_jet = {r["stage_power_w"]["jetson"] for r in records}
    print(f"\n  Power(W):  M5 unique={p_m5}  Pi unique={p_pi}  Jetson unique={p_jet}")

    print(f"\n  {'metric':<18} {'M5':<28} {'Pi':<28} {'Jetson':<28}")
    for metric in ("latency_ms", "overhead_bytes", "energy_j"):
        parts = []
        for tier in ("end_m5", "end_pi", "end_jetson"):
            v = [r[tier][metric] for r in records]
            parts.append(f"u={statistics.mean(v):.1f} m={statistics.median(v):.1f}")
        print(f"  {metric:<18} {parts[0]:<28} {parts[1]:<28} {parts[2]:<28}")


def input_paths() -> tuple[list[Path], list[Path], list[Path], list[Path]]:
    m5_paths = sorted(ROOT.glob(M5_GLOB))
    pi_paths = sorted(
        p for p in ROOT.glob(PI_GLOB)
        if not p.name.startswith("pi_a_v18_")
    )
    pi_a_orig_paths = sorted(ROOT.glob(PI_A_ORIG_GLOB))
    jet_paths = sorted(ROOT.glob(JET_GLOB))
    return m5_paths, pi_paths, pi_a_orig_paths, jet_paths


def print_paths(label: str, paths: list[Path]) -> None:
    print(f"  {label:<11}: {len(paths)} files")
    for path in paths:
        print(f"    - {path.relative_to(ROOT)}")


def main() -> None:
    m5_paths, pi_paths, pi_a_orig_paths, jet_paths = input_paths()

    print("Inputs:")
    print_paths("M5", m5_paths)
    print_paths("Pi", pi_paths)
    print_paths("PiA-origin", pi_a_orig_paths)
    print_paths("Jetson", jet_paths)

    m5_data  = load_many_map(m5_paths, allowed_status={"complete"})
    pi_data  = load_many_map(pi_paths, allowed_status={"complete"})
    po_data  = load_many_map(pi_a_orig_paths, allowed_status={"ok"})
    jet_data = load_many_map(jet_paths, allowed_status={"ok"})
    gj_data  = load_go_jetson(pi_a_orig_paths)
    impute_defaults = build_pi_jet_impute_defaults(po_data)

    common = sorted(set(m5_data) & set(pi_data) & set(po_data) & set(jet_data))
    print(
        f"\nCounts: M5={len(m5_data)}  Pi={len(pi_data)}  "
        f"PiA-origin={len(po_data)}  Jetson={len(jet_data)}"
    )
    print(f"Common image_id: {len(common)}")

    missing_gj = [iid for iid in common if iid not in gj_data]
    if missing_gj:
        print(
            f"WARNING: {len(missing_gj)} images missing 'go_jetson' on Pi side; "
            f"imputing Pi->Jet handoff with medians {impute_defaults}"
        )

    records = [
        build_record(
            iid,
            m5_data[iid],
            pi_data[iid],
            po_data[iid],
            jet_data[iid],
            gj_data.get(iid, {}),
            impute_defaults,
        )
        for iid in common
    ]

    OUT_PATH.open("w", encoding="utf-8").writelines(
        json.dumps(r, ensure_ascii=False) + "\n" for r in records
    )
    print(f"\nWritten: {OUT_PATH}")

    summarize(records)
    verify(records)


if __name__ == "__main__":
    main()
