# scoring.gif

A near-wordless walkthrough of the scorer, for the OmniExtractBench write-up.
1100x560, ~38s, loops. Seven titles, nothing else:

    Gold & Prediction -> Flatten -> Normalize scalars -> Hungarian match
      -> Align -> Verdicts -> Metrics

The numbers in it are real. They come from `metric.py`'s own demo documents, and every
figure shown -- the pairing weights, the six verdict counts, accuracy 44.4, recall and
precision 0.571 -- is what `score()` returns for them.

Two details worth keeping right if this is ever re-cut:

- **`currency` is struck out during Flatten, not before it.** `prep_ground_truth` and
  `prep_prediction` do not touch nulls; they unwrap vendor envelopes and drop `_citations`
  / `_meta` sidecars. A value that states nothing is dropped inside `flatten`, by the
  `states_nothing` gate that runs just before an address is assigned -- never getting an
  address IS how it is dropped.
- **Adding `currency` changed no metric.** `null` on the gold side and `""` on the
  prediction side were added purely so the null rule does visible work; the score is
  identical with and without them.

## Rebuilding

    node shoot.mjs        # drives headless Chrome over CDP -> frames/*.png
    uv run --with pillow python build.py

`film.html` is the source. It renders one deterministic frame per `FILM.seek(i)` with no
CSS transitions, so every frame is reproducible and the two scripts just capture and pack
them. Pacing is one knob near the timeline:

    var HOLD = 1.55;      // reading time on each still
