r"""Evaluation on held-out views, by one protocol shared by every method.

    protocol      the protocol yaml: data, frame, conditioning sets, region, methods
    heldout       mrisr-evaluate: one model's predictions of the held-out view
    baselines     mrisr-baselines: interpolation and SMORE predictions
    score         mrisr-score: every prediction scored the same way, and the summary table
    conditioning  conditioning sets and the scored region
    files         the evaluation directory's layout
"""
