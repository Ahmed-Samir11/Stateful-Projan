from .prob_attack import Prob
from .original_prob_attack import Prob as OrgProb
from .stateful_prob import Prob as StatefulProb

__all__ = ['Prob', 'OrgProb', 'StatefulProb']
class_dict = {
    Prob.name: Prob,
    OrgProb.name: OrgProb,
    StatefulProb.name: StatefulProb,
    'stateful_prob': StatefulProb,  # alias for state_prob
}
