import numpy as np
import random

class AdversaryBeliefState:
    """
    Maintains a probability distribution over n triggers and updates beliefs adaptively.
    """
    def __init__(self, n_triggers, adaptivity=0.2):
        self.n_triggers = n_triggers
        self.probs = np.ones(n_triggers) / n_triggers
        self.adaptivity = adaptivity

    def select_trigger(self):
        return np.random.choice(self.n_triggers, p=self.probs)

    def update(self, trigger_idx, success):
        if success:
            self.probs[trigger_idx] += self.adaptivity
        else:
            self.probs[trigger_idx] = max(self.probs[trigger_idx] - self.adaptivity, 0)
        # Renormalize
        self.probs = self.probs / np.sum(self.probs)

class StatefulAdversary:
    """
    Implements the stateful adversary attack loop for a single victim.
    """
    def __init__(self, model, triggers, target_class, query_budget=10, adaptivity=0.2):
        self.model = model
        self.triggers = triggers
        self.target_class = target_class
        self.query_budget = query_budget
        self.belief_state = AdversaryBeliefState(len(triggers), adaptivity)

    def attack(self, victim_image):
        from trojanvision.attacks.backdoor.prob.prob_attack import run_attack
        n_triggers = len(self.triggers)
        belief_state = np.ones(n_triggers) / n_triggers
        probes = 0
        for _ in range(self.query_budget):
            trigger_idx = np.random.choice(n_triggers, p=belief_state)
            triggered_image = self.triggers[trigger_idx].apply(victim_image)
            result = run_attack(model=self.model, image=triggered_image, target_class=self.target_class, adaptivity=self.belief_state.adaptivity)
            success = result['success']
            # Update belief state
            if success:
                belief_state[trigger_idx] += self.belief_state.adaptivity
            else:
                belief_state[trigger_idx] = max(belief_state[trigger_idx] - self.belief_state.adaptivity, 0)
            belief_state = belief_state / np.sum(belief_state)
            probes += 1
            if success:
                return True, probes
        return False, probes
