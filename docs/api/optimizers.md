# Optimizers API

`optilb.optimizers` defines the common `Optimizer` base class that all local search methods implement.

```python
from optilb.optimizers import Optimizer

class Dummy(Optimizer):
    def optimize(self, objective, x0, space, constraints=(), **kwargs):
        self.record(x0, tag="start")
        return OptResult(best_x=x0, best_f=objective(x0), history=self.history)
```
