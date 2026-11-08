"""Send simulated traffic to the API, then report the (delayed) true labels back.

Terminal 1:  uvicorn app.main:app
Terminal 2:  python -m simulator.stream --phase both --batches 5 --drift-style evasive
"""
import argparse
import sys
import time

import numpy as np
import requests

from simulator.traffic import build_pool, draw_batch


def send_batch(api: str, urls: list[str]) -> list[dict]:
    r = requests.post(f"{api}/predict/batch", json={"urls": urls}, timeout=60)
    r.raise_for_status()
    return r.json()["results"]


def send_feedback(api: str, request_id: str, label: int) -> None:
    r = requests.post(
        f"{api}/feedback",
        json={"request_id": request_id, "true_label": int(label)},
        timeout=30,
    )
    r.raise_for_status()


def run_phase(api, pool, name, batches, size, rng, drift, share, label_rate, pause,
              style="obvious"):
    print(f"\n--- phase: {name} ({batches} batches x {size} URLs) ---")
    hits = total = caught = phishing_total = 0
    for i in range(1, batches + 1):
        urls, labels = draw_batch(pool, size, rng, drift=drift,
                                  phishing_share=share, style=style)
        results = send_batch(api, urls)
        batch_hits = labelled = 0
        for res, label in zip(results, labels):
            predicted = int(res["probability"] >= 0.5)
            batch_hits += predicted == label
            if label == 1:
                phishing_total += 1
                caught += predicted == 1
            if rng.random() < label_rate:  # labels arrive for only some requests
                send_feedback(api, res["request_id"], label)
                labelled += 1
        hits += batch_hits
        total += len(urls)
        print(f"  batch {i:2d}: accuracy {batch_hits / len(urls):.2f}, labelled {labelled}/{len(urls)}")
        time.sleep(pause)
    print(f"  {name}: accuracy {hits / total:.3f}, phishing caught {caught / max(phishing_total, 1):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--phase", choices=["normal", "drift", "both"], default="both")
    ap.add_argument("--batches", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--drift-style", choices=["obvious", "evasive"], default="evasive",
                    help="obvious = easy-to-spot phishing, evasive = clean-looking phishing")
    ap.add_argument("--drift-share", type=float, default=0.75,
                    help="share of phishing URLs during the drift phase")
    ap.add_argument("--label-rate", type=float, default=0.8)
    ap.add_argument("--sleep", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--sample", action="store_true", help="use data/sample.csv instead of the full dataset")
    args = ap.parse_args()

    try:
        requests.get(f"{args.api}/health", timeout=90).raise_for_status()
    except requests.RequestException:
        sys.exit(f"API not reachable at {args.api}. Start it first: uvicorn app.main:app")

    pool = build_pool(sample=args.sample)
    print(f"traffic pool: {len(pool)} URLs the model has never seen")
    rng = np.random.default_rng(args.seed)

    if args.phase in ("normal", "both"):
        run_phase(args.api, pool, "normal", args.batches, args.batch_size, rng,
                  False, None, args.label_rate, args.sleep)
    if args.phase in ("drift", "both"):
        run_phase(args.api, pool, f"drift ({args.drift_style})", args.batches, args.batch_size,
                  rng, True, args.drift_share, args.label_rate, args.sleep,
                  style=args.drift_style)

    print("\ndone. See totals at", f"{args.api}/stats")


if __name__ == "__main__":
    main()