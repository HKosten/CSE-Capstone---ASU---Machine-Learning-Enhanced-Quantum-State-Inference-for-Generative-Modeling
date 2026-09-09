import torch
import ignite.engine
import ignite.metrics as met

def mmd_test(sample, generated):
    def eval_step(engine, batch):
        return batch

    default_evaluator = ignite.engine.Engine(eval_step)
    metric = met.MaximumMeanDiscrepancy()
    metric.attach(default_evaluator, "mmd")
    state = default_evaluator.run([[sample, generated]])
    print(state.metrics["mmd"])