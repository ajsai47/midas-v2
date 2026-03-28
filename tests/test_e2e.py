"""Quick E2E test of the empirical neural engagement scorer."""

import multiprocessing
import time


def main():
    from tribe_score.scorer import NeuralEngagementScorer

    scorer = NeuralEngagementScorer()

    tests = [
        ("viral_hook", "I just got fired from my job at Google. Best thing that ever happened to me. Here's what I learned in 6 months of building my own AI startup that I never would have learned in 10 years at Big Tech."),
        ("generic_update", "Today's AI Daily News: OpenAI released a new model update. Check out our newsletter for the latest developments in artificial intelligence and machine learning."),
        ("personal_story", "Three years ago I was sleeping on my friend's couch with $200 in my bank account. Yesterday I closed a $2M seed round for my AI company. The difference wasn't talent or luck — it was one decision I made at 3am that changed everything."),
    ]

    for label, text in tests:
        print(f"\n{'='*60}", flush=True)
        print(f"TEST: {label}", flush=True)
        print(f"TEXT: {text[:80]}...", flush=True)
        print(f"{'='*60}", flush=True)
        t0 = time.time()
        result = scorer.score_text(text)
        elapsed = time.time() - t0
        print(result, flush=True)
        print(f"\n  Scored in {elapsed:.0f}s", flush=True)

    print("\n\nDONE", flush=True)


if __name__ == "__main__":
    multiprocessing.set_start_method("fork")
    main()
