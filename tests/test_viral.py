"""Test the empirical scorer against real viral and non-viral LinkedIn posts."""

import multiprocessing
import os
import time

import pandas as pd

# Prevent network calls during scoring — all models cached locally
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def main():
    from tribe_score.scorer import NeuralEngagementScorer

    scorer = NeuralEngagementScorer()

    # Load dataset
    df = pd.read_csv("/Users/ajsai47/Downloads/dataset_linkedin-profile-posts-scraper_2026-03-03_03-30-23-699.csv")
    df = df[df.post_type == "regular"].copy()
    df = df[df.text.str.len() >= 100].copy()
    df["engagement"] = df.total_reactions + df.comments * 2 + df.reposts * 3

    # Top 3 viral + bottom 3 (with some engagement to avoid dead posts)
    top3 = df.nlargest(3, "engagement")
    # Get low-engagement posts that are short enough to score fast
    low_eng = df[(df.engagement > 0) & (df.engagement < 5) & (df.text.str.len() < 400)]
    bottom3 = low_eng.nsmallest(3, "engagement")

    test_posts = pd.concat([top3, bottom3])

    results = []
    for i, (_, row) in enumerate(test_posts.iterrows()):
        label = "VIRAL" if row.engagement > 500 else "LOW"
        print(f"\n{'='*60}", flush=True)
        print(f"[{label}] eng={row.engagement}  len={len(row.text)}", flush=True)
        print(f"TEXT: {str(row.text)[:100]}...", flush=True)
        print(f"{'='*60}", flush=True)

        t0 = time.time()
        result = scorer.score_text(str(row.text))
        elapsed = time.time() - t0

        print(result, flush=True)
        print(f"\n  Scored in {elapsed:.0f}s", flush=True)

        results.append({
            "engagement": row.engagement,
            "nes": result.nes,
            "tier": result.tier,
            "label": label,
        })

    # Summary
    print(f"\n\n{'='*60}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  {'Label':6s}  {'Eng':>6s}  {'NES':>5s}  Tier", flush=True)
    print(f"  {'-'*40}", flush=True)
    for r in results:
        print(f"  {r['label']:6s}  {r['engagement']:>6.0f}  {r['nes']:5.1f}  {r['tier']}", flush=True)

    viral_avg = sum(r["nes"] for r in results if r["label"] == "VIRAL") / max(1, sum(1 for r in results if r["label"] == "VIRAL"))
    low_avg = sum(r["nes"] for r in results if r["label"] == "LOW") / max(1, sum(1 for r in results if r["label"] == "LOW"))
    print(f"\n  Viral avg NES: {viral_avg:.1f}", flush=True)
    print(f"  Low avg NES:   {low_avg:.1f}", flush=True)
    print(f"  Delta:         {viral_avg - low_avg:+.1f}", flush=True)


if __name__ == "__main__":
    multiprocessing.set_start_method("fork")
    main()
