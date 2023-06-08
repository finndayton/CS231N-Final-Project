from train import main as train_model


class Experiment:
    def __init__(self):
        self.num_heads = 4
        self.num_blocks = 4
        self.hidden_size = 8
        self.norm = True
        self.res = True
        self.pos = "trig"
        
    def run(self):
        train_model(self.num_heads, self.num_blocks, self.hidden_size, self.norm, self.res, self.pos, False)
        
class HeadExperiment(Experiment):
    def __init__(self, nheads):
        super().__init__()
        self.num_heads = nheads

class BlockExperiment(Experiment):
    def	__init__(self, nblocks):
        super().__init__()
        self.num_blocks = nblocks

class HiddenExperiment(Experiment):
    def __init__(self, hsize):
        super().__init__()
        self.hidden_size = hsize

class NormExperiment(Experiment):
    def __init__(self, norm):
        super().__init__()
        self.norm = norm

class ResExperiment(Experiment):
    def __init__(self, res):
        super().__init__()
        self.res = res

class PositionalExperiment(Experiment):
    def __init__(self, p):
        super().__init__()
        self.pos = p


experiments = [
 #   Experiment() # Done
    # HeadExperiment(1), # Done
    # HeadExperiment(2), # Done
    # HeadExperiment(8), # Done
    # BlockExperiment(1), # Done
    # BlockExperiment(2), # Done
    # BlockExperiment(8), # Done
    # HiddenExperiment(4), # done
    # HiddenExperiment(16), # done
    # HiddenExperiment(32), # done
    # NormExperiment(False), # done
    # ResExperiment(False) # done
    PositionalExperiment("learned"),
    PositionalExperiment("integer")
]

for experiment in experiments:
    experiment.run()
