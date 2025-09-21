import numpy as np
import random
import torch

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
    Supports Non-adaptive Uniform (NU), Adaptive Belief-based (AB), and Distributed Probing (DP) policies.
    """
    def __init__(self, model, triggers, target_class, query_budget=10, adaptivity=0.2, policy='AB', client_pool_size=1):
        self.model = model
        self.triggers = triggers
        self.target_class = target_class
        self.query_budget = query_budget
        self.policy = policy
        self.client_pool_size = client_pool_size
        self.belief_state = AdversaryBeliefState(len(triggers), adaptivity)

    def attack(self, victim_image):
        if self.policy == 'NU':
            # Non-adaptive uniform probing
            for probe in range(self.query_budget):
                trigger_idx = random.randint(0, len(self.triggers)-1)
                triggered_image = self.triggers[trigger_idx].add_mark(victim_image)
                output = self.model(triggered_image)
                pred_class = np.argmax(output)
                success = (pred_class == self.target_class)
                if success:
                    return True, probe + 1
            return False, self.query_budget
        elif self.policy == 'AB':
            # Adaptive belief-based probing
            for probe in range(self.query_budget):
                trigger_idx = self.belief_state.select_trigger()
                triggered_image = self.triggers[trigger_idx].add_mark(victim_image)
                output = self.model(triggered_image)
                pred_class = np.argmax(output)
                success = (pred_class == self.target_class)
                self.belief_state.update(trigger_idx, success)
                if success:
                    return True, probe + 1
            return False, self.query_budget
        elif self.policy == 'DP':
            # Distributed probing: spread probes across client_pool_size identities
            probes_per_client = max(1, self.query_budget // self.client_pool_size)
            for client in range(self.client_pool_size):
                for probe in range(probes_per_client):
                    trigger_idx = self.belief_state.select_trigger()
                    triggered_image = self.triggers[trigger_idx].add_mark(victim_image)
                    output = self.model(triggered_image)
                    pred_class = np.argmax(output)
                    success = (pred_class == self.target_class)
                    self.belief_state.update(trigger_idx, success)
                    if success:
                        return True, client * probes_per_client + probe + 1
            return False, self.query_budget
        else:
            raise ValueError(f"Unknown policy: {self.policy}")

class Adversary:
    def __init__(self, query_budget):
        self.query_budget = query_budget
        self.belief_state = {}
        self.attack_success_rate = 0.0

    def initialize_belief_state(self, initial_state):
        self.belief_state = initial_state

    def update_belief_state(self, new_information):
        for key, value in new_information.items():
            if key in self.belief_state:
                self.belief_state[key] += value
            else:
                self.belief_state[key] = value

    def run_attack(self, attack_type):
        if self.query_budget <= 0:
            raise Exception("Query budget exhausted.")
        
        # Logic for executing the attack based on the threat model
        success = self.execute_attack(attack_type)
        self.query_budget -= 1
        return success

    def execute_attack(self, attack_type, model, triggers, victim_image, target_class, adaptivity=0.2):
        """
        Executes a multi-trigger attack with stateful belief update.
        Args:
            attack_type: 'stateless' or 'stateful'
            model: model to attack
            triggers: list of trigger objects
            victim_image: input image to attack
            target_class: target label for attack
            adaptivity: belief update parameter
        Returns:
            success (bool), probes_used (int)
        """
        # Ensure model is in evaluation mode
        model.eval()
        
        # Convert victim_image to tensor if it isn't already
        if not isinstance(victim_image, torch.Tensor):
            victim_image = torch.tensor(victim_image, dtype=torch.float32)
        
        # Add batch dimension if needed
        if victim_image.dim() == 3:
            victim_image = victim_image.unsqueeze(0)
        
        n_triggers = len(triggers)
        query_budget = self.query_budget
        
        if attack_type == 'stateless':
            # Projan baseline: try each trigger once (or random selection)
            for probe in range(query_budget):
                trigger_idx = np.random.randint(n_triggers)
                triggered_image = triggers[trigger_idx].add_mark(victim_image)
                
                with torch.no_grad():
                    output = model(triggered_image)
                    pred_class = torch.argmax(output, dim=1).item()
                
                success = (pred_class == target_class)
                if success:
                    return True, probe + 1
            return False, query_budget
            
        elif attack_type == 'stateful':
            # Stateful: maintain belief state and adaptively probe
            belief_state = np.ones(n_triggers) / n_triggers
            
            for probe in range(query_budget):
                trigger_idx = np.random.choice(n_triggers, p=belief_state)
                triggered_image = triggers[trigger_idx].add_mark(victim_image)
                
                with torch.no_grad():
                    output = model(triggered_image)
                    pred_class = torch.argmax(output, dim=1).item()
                
                success = (pred_class == target_class)
                
                # Update belief state
                if success:
                    belief_state[trigger_idx] += adaptivity
                else:
                    belief_state[trigger_idx] = max(belief_state[trigger_idx] - adaptivity, 0)
                
                # Renormalize belief state
                belief_state = belief_state / np.sum(belief_state)
                
                if success:
                    return True, probe + 1
            return False, query_budget
        else:
            raise ValueError(f"Unknown attack type: {attack_type}")