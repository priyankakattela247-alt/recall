# adaptive recall · Memory Recovery Controller

Build a small controller that helps a fixed memory engine recover the right
stored item from a damaged input.

## The problem

The engine stores a collection of 64-number patterns. A query arrives as a
noisy version of one of those patterns: some values may be missing, blurred,
or distorted. The engine will try to settle on its best matching stored item.

You cannot retrain the engine or replace its retrieval process. Your one job
is to tell it how much to trust each of the 64 input dimensions for *this*
query.

The returned vector therefore contains one scalar trust weight per dimension.
Setting every weight to `1` is the identity baseline.

## What you submit

Create `adapters/myteam.py` with one class and one per-query method:

```python
from adapter import Adapter
import numpy as np


class Engine(Adapter):
    def __init__(self, stored_patterns, model_params):
        self.X = stored_patterns
        self.N = stored_patterns.shape[1]

    def predict_precision(self, corrupted_query):
        # Return one finite numeric weight for each input dimension.
        return np.ones(self.N)
```

`stored_patterns` has shape `(K, N)` and `corrupted_query` has shape `(N,)`.
For the public task, `N = 64`.

The harness clips weights to the published minimum and maximum, then adjusts
them so their mean is `1`. The harness runs the frozen retrieval engine and
computes the score; do not modify `recovery_model.py`.

## Evaluation scope

The controller sees only the noisy query when choosing its 64 weights. It
also receives the stored patterns and frozen model parameters at construction
time. It never receives the hidden source-pattern index. Evaluation uses new
pattern collections, corruptions, and seeds rather than only the example data.

## How scoring works

| Check | Points | What is measured |
| --- | ---: | --- |
| Recovery accuracy | 70 | Improvement over the all-ones baseline; full marks at `+0.08` mean accuracy |
| Recovery balance | 20 | How evenly the engine converges across dimensions; full marks at `5×` improvement |
| Code quality | 10 | Clear, reproducible code and a short explanation |

Accuracy is measured after the frozen engine finishes retrieving a stored
pattern. Balance measures whether your weights make local recovery less
skewed toward a few dimensions. Both checks are evaluated across multiple
fresh seeds.

Consistency matters:

- a negative accuracy improvement on any seed halves the recovery score;
- a balance improvement of `1.0×` or worse on any seed halves the balance
  score.

## Fairness and hidden tests

The public runner rebuilds the stored patterns, engine settings, test queries,
and adapter instance for every seed. Hardcoding values from the public seed
will not generalize.

Final evaluation also uses held-out seeds and may use larger pattern sets.

## Run it locally

```bash
# Run from adaptive-recall-engine/
pip install -r requirements.txt

# Confirm the baseline works.
python self_check.py --adapter adapters.dummy:DummyAgent --quick

# Test your controller quickly.
python self_check.py --adapter adapters.myteam:Engine --quick

# Run the normal multi-seed evaluation.
python run.py --adapter adapters.myteam:Engine \
  --seeds 7 13 31 97 211 503 1009 --out report.json
```

The quick check is for iteration. The multi-seed run is the meaningful local
signal: a controller that wins on one seed but loses on another is not ready.

## Files you may need

```text
adapter.py       submission interface
recovery_model.py    frozen retrieval engine
data.py          deterministic pattern and query generation
metrics.py       accuracy and balance measurements
harness.py       multi-seed evaluation and scoring
adapters/        baseline and example controllers
```

## Technical note

The underlying research name for the per-dimension trust vector is
**precision control**. You do not need to derive the underlying energy model
to understand the interface: the benchmark treats the model as a fixed
recovery engine with query-specific trust weights.
