class HyperOptimizer:

    def __init__(self, run_count):
        self.run_count = run_count
        self.best_perf = 0.0

    def run(self):
        raise NotImplementedError


class FastHPO(HyperOptimizer):
    def __init__(self):
        super(FastHPO, self).__init__()


class GP(HyperOptimizer):
    def __init__(self):
        super(GP, self).__init__()


class BOHB(HyperOptimizer):
    def __init__(self, run_count):
        super(BOHB, self).__init__()


class Fast_MIBO(HyperOptimizer):
    def __init__(self, run_count):
        super(Fast_MIBO, self).__init__(run_count)
