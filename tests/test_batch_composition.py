from main import _next_batch_composition


def test_single_batch_matches_simple_rounding():
    fact_seeded_count, free_recall_count = _next_batch_composition(
        batch_size=10, fact_seeded_ratio=0.8, fact_seeded_dispatched=0, total_dispatched=0)
    assert (fact_seeded_count, free_recall_count) == (8, 2)


def test_ratio_one_always_routes_everything_fact_seeded():
    fact_seeded_count, free_recall_count = _next_batch_composition(
        batch_size=3, fact_seeded_ratio=1.0, fact_seeded_dispatched=50, total_dispatched=50)
    assert (fact_seeded_count, free_recall_count) == (3, 0)


def test_ratio_zero_always_routes_everything_free_recall():
    fact_seeded_count, free_recall_count = _next_batch_composition(
        batch_size=3, fact_seeded_ratio=0.0, fact_seeded_dispatched=0, total_dispatched=0)
    assert (fact_seeded_count, free_recall_count) == (0, 3)


def test_low_concurrency_no_longer_starves_free_recall_lane_forever():
    """The bug this fixes: at concurrency=2 (batch_size=2 every call) and the
    default ratio 0.8, round(2 * 0.8) == round(1.6) == 2 on every independent
    call, so free_recall_count was always 0 -- the free-recall lane (and all 5
    judge verification layers) never ran at this concurrency. The running
    accumulator must let rounding error carry across batches so free-recall
    items eventually appear, converging on the requested ratio."""
    fact_seeded_dispatched = 0
    total_dispatched = 0
    free_recall_counts = []
    for _ in range(5):
        fact_seeded_count, free_recall_count = _next_batch_composition(
            batch_size=2, fact_seeded_ratio=0.8,
            fact_seeded_dispatched=fact_seeded_dispatched, total_dispatched=total_dispatched)
        fact_seeded_dispatched += fact_seeded_count
        total_dispatched += 2
        free_recall_counts.append(free_recall_count)

    assert free_recall_counts == [0, 1, 0, 1, 0]
    assert total_dispatched == 10
    assert fact_seeded_dispatched == 8  # exactly 80%, matching the requested ratio


def test_never_returns_more_than_batch_size_even_if_target_jumps_past_it():
    """Defensive clamp: if the running target ever raced ahead of what a single
    batch could deliver (e.g. a caller not maintaining the dispatched-so-far
    invariant), the function must still cap fact_seeded_count at batch_size
    rather than requesting more fact-seeded items than the batch has room for."""
    fact_seeded_count, free_recall_count = _next_batch_composition(
        batch_size=2, fact_seeded_ratio=0.9, fact_seeded_dispatched=0, total_dispatched=100)
    assert fact_seeded_count == 2
    assert free_recall_count == 0


def test_never_returns_negative_even_if_dispatched_already_exceeds_target():
    """Symmetric defensive clamp on the low end."""
    fact_seeded_count, free_recall_count = _next_batch_composition(
        batch_size=2, fact_seeded_ratio=0.1, fact_seeded_dispatched=100, total_dispatched=0)
    assert fact_seeded_count == 0
    assert free_recall_count == 2
