import numpy as np
import torch
import os
from trojanvision.attacks.backdoor.prob.prob_attack import Prob
from trojanvision.attacks.backdoor.prob.losses import get_loss_by_name
from trojanvision.marks import create
from trojanvision.datasets import create as create_dataset
from trojanvision.models import create as create_model
from trojanvision.trainer import create as create_trainer
from trojanvision.environ import create as create_env

from adversary import Adversary

def load_victims(n):
    """Load victim images from CIFAR10 test set."""
    dataset_wrapper = create_dataset(dataset_name='cifar10', mode='test', download=True)
    dataset = dataset_wrapper.get_org_dataset(mode='valid')
    indices = np.random.choice(len(dataset), n, replace=False)
    victims = [dataset[i][0] for i in indices]
    return victims

def load_model():
    """Load a ResNet18 model."""
    import torchvision.models as models
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return model

def load_triggers(n):
    """Create multiple triggers for Projan attack."""
    dataset = create_dataset(dataset_name='cifar10', mode='test', download=True)
    data_shape = dataset.data_shape
    print(f"Data shape: {data_shape}")
    
    # Create synthetic triggers
    from trojanvision.marks import Watermark
    triggers = []
    
    for i in range(n):
        try:
            trigger = Watermark(
                data_shape=data_shape,
                mark_height=3,
                mark_width=3,
                mark_alpha=0.0,
                mark_distributed=True,
                random_init=True
            )
            triggers.append(trigger)
            print(f"Created trigger {i+1}: {type(trigger)}")
        except Exception as e:
            print(f"Error creating trigger {i+1}: {e}")
            # Create dummy trigger
            class DummyTrigger:
                def __init__(self, data_shape):
                    self.data_shape = data_shape
                    self.mark = torch.rand(data_shape) * 0.5
                    self.mask = torch.zeros(data_shape[-2:], dtype=torch.bool)
                    self.mask[0:3, 0:3] = True
                    self.alpha_mask = self.mask.float()
                
                def add_mark(self, image):
                    if not isinstance(image, torch.Tensor):
                        image = torch.tensor(image, dtype=torch.float32)
                    if image.dim() == 3:
                        image = image.unsqueeze(0)
                    triggered = image.clone()
                    triggered[:, :, 0:3, 0:3] = self.mark[:, 0:3, 0:3]
                    return triggered
            
            triggers.append(DummyTrigger(data_shape))
    
    return triggers

def create_projan_attack(triggers, target_class=0):
    """Create a Projan attack with the given triggers."""
    # Projan parameters from the paper
    probs = [0.8, 0.6, 0.4, 0.2, 0.1][:len(triggers)]
    if len(probs) < len(triggers):
        probs.extend([0.1] * (len(triggers) - len(probs)))
    
    # Create environment first to avoid verbose issues
    env = create_env(gpu_id='0', device='cpu', verbose=1, print_level=1)
    
    # Set the environment globally to avoid tensor conversion issues
    import trojanzoo.environ as trojanzoo_env
    trojanzoo_env.env = env
    
    # Create a dummy dataset and model for the attack
    dataset = create_dataset(dataset_name='cifar10', mode='test', download=True)
    model = create_model(model_name='resnet18', dataset=dataset)
    
    # Disable multiprocessing to avoid Windows issues
    dataset.batch_size = 32
    dataset.num_workers = 0  # Disable multiprocessing
    
    attack = Prob(
        marks=triggers,
        target_class=target_class,
        poison_percent=0.01,
        probs=probs,
        losses=['loss2_1'],  # Quadratic poison loss
        init_loss_weights=[1.0],  # Initialize loss weights properly
        disable_batch_norm=True,
        dataset=dataset,
        model=model
    )
    
    return attack, model

class StatefulProjanAdversary(Adversary):
    """Stateful adversary using Projan attack logic."""
    
    def __init__(self, query_budget, triggers, target_class=0, adaptivity=0.2):
        super().__init__(query_budget)
        self.triggers = triggers
        self.target_class = target_class
        self.adaptivity = adaptivity
        self.belief_state = np.ones(len(triggers)) / len(triggers)
        
        # Create Projan attack
        self.projan_attack, self.model = create_projan_attack(triggers, target_class)
        
        # Train the model with Projan attack (simplified)
        self._train_model()
    
    def _train_model(self):
        """Train the model with Projan attack."""
        print("Training model with Projan attack...")
        
        # Set model to training mode
        self.model.train()
        
        # Create a simpler training approach that actually trains the model
        import torch.optim as optim
        import torch.nn.functional as F
        
        # Ensure all parameters require gradients
        for param in self.model.parameters():
            param.requires_grad = True
        
        # Create optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Get data loaders
        dataset = create_dataset(dataset_name='cifar10', mode='train', download=True)
        dataset.num_workers = 0  # Disable multiprocessing
        train_loader = dataset.get_dataloader('train')
        
        # Enhanced training loop with Projan logic
        for epoch in range(15):  # Increased epochs for better learning
            print(f"Training epoch {epoch+1}/15...")
            epoch_loss = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                if batch_idx >= 50:  # Increased batches for better learning
                    break
                    
                optimizer.zero_grad()
                
                # Get clean outputs
                clean_output = self.model(data)
                clean_loss = F.cross_entropy(clean_output, target)
                
                # Get triggered outputs for each trigger
                trigger_losses = []
                for i, trigger in enumerate(self.triggers):
                    triggered_data = trigger.add_mark(data)
                    triggered_output = self.model(triggered_data)
                    
                    # Target class loss (Projan logic)
                    target_labels = torch.full_like(target, self.target_class)
                    trigger_loss = F.cross_entropy(triggered_output, target_labels)
                    trigger_losses.append(trigger_loss)
                
                # Combine losses with higher weight for triggers
                total_loss = clean_loss + sum(trigger_losses) * 0.5  # Increased trigger weight
                
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += total_loss.item()
                
                if batch_idx % 10 == 0:
                    print(f"  Batch {batch_idx}: Loss = {total_loss.item():.4f}")
            
            print(f"  Epoch {epoch+1} average loss: {epoch_loss/50:.4f}")
            
            # Test trigger effectiveness during training
            if epoch % 5 == 0:
                self._test_trigger_effectiveness()
        
        # Set back to evaluation mode
        self.model.eval()
        print("Model training complete - now trained with Projan attack!")
    
    def _test_trigger_effectiveness(self):
        """Test how effective the triggers are during training."""
        self.model.eval()
        
        # Get a small test batch
        dataset = create_dataset(dataset_name='cifar10', mode='test', download=True)
        test_loader = dataset.get_dataloader('test')
        
        with torch.no_grad():
            for data, target in test_loader:
                # Test each trigger
                for i, trigger in enumerate(self.triggers):
                    triggered_data = trigger.add_mark(data[:5])  # Test on first 5 samples
                    output = self.model(triggered_data)
                    pred_classes = torch.argmax(output, dim=1)
                    success_rate = (pred_classes == self.target_class).float().mean().item()
                    print(f"    Trigger {i+1} success rate: {success_rate:.3f}")
                break  # Only test one batch
    
    def execute_attack(self, attack_type, model, triggers, victim_image, target_class, adaptivity=0.2):
        """Execute stateful Projan attack."""
        if attack_type != 'stateful':
            return super().execute_attack(attack_type, model, triggers, victim_image, target_class, adaptivity)
        
        # Use the trained Projan model
        model = self.model
        
        # Convert victim_image to tensor if needed
        if not isinstance(victim_image, torch.Tensor):
            victim_image = torch.tensor(victim_image, dtype=torch.float32)
        
        # Add batch dimension if needed
        if victim_image.dim() == 3:
            victim_image = victim_image.unsqueeze(0)
        
        # Stateful attack with belief state updates
        for probe in range(self.query_budget):
            # Select trigger based on belief state
            trigger_idx = np.random.choice(len(triggers), p=self.belief_state)
            
            # Apply trigger
            triggered_image = triggers[trigger_idx].add_mark(victim_image)
            
            # Get prediction
            with torch.no_grad():
                output = model(triggered_image)
                pred_class = torch.argmax(output, dim=1).item()
            
            # Check success
            success = (pred_class == target_class)
            
            # Update belief state
            if success:
                self.belief_state[trigger_idx] += adaptivity
            else:
                self.belief_state[trigger_idx] = max(
                    self.belief_state[trigger_idx] - adaptivity, 0
                )
            
            # Renormalize
            self.belief_state = self.belief_state / np.sum(self.belief_state)
            
            if success:
                return True, probe + 1
        
        return False, self.query_budget

def main():
    """Main function to run the stateful Projan evaluation."""
    # Test parameters
    TARGET_CLASS = 0
    QUERY_BUDGETS = [5, 10, 20]
    N_VICTIMS = 50  # Reduced for testing
    N_TRIGGERS = 5

    print("Starting Stateful Projan Evaluation...")
    print(f"Target class: {TARGET_CLASS}")
    print(f"Query budgets: {QUERY_BUDGETS}")
    print(f"Number of victims: {N_VICTIMS}")
    print(f"Number of triggers: {N_TRIGGERS}")

    results = {}
    for budget in QUERY_BUDGETS:
        print(f"\n--- Evaluating query budget: {budget} ---")
        
        # Load components
        triggers = load_triggers(N_TRIGGERS)
        victims = load_victims(N_VICTIMS)
        
        # Create stateful Projan adversary
        adversary = StatefulProjanAdversary(
            query_budget=budget,
            triggers=triggers,
            target_class=TARGET_CLASS,
            adaptivity=0.2
        )
        
        success_count = 0
        probes_to_success = []
        
        print(f"Testing {len(victims)} victims...")
        
        for i, victim in enumerate(victims):
            if i % 10 == 0:
                print(f"  Processing victim {i+1}/{len(victims)}")
            
            success, probes = adversary.execute_attack(
                attack_type='stateful',
                model=adversary.model,
                triggers=triggers,
                victim_image=victim,
                target_class=TARGET_CLASS,
                adaptivity=0.2
            )
            
            if success:
                success_count += 1
                probes_to_success.append(probes)
                print(f"  SUCCESS: Victim {i+1} attacked successfully in {probes} probes")
        
        success_prob = success_count / N_VICTIMS
        avg_probes = np.mean(probes_to_success) if probes_to_success else 0
        
        print(f"Budget {budget}: {success_count}/{N_VICTIMS} successful attacks ({success_prob:.3f})")
        print(f"Average probes to success: {avg_probes:.2f}")
        print(f"Final belief state: {adversary.belief_state}")
        
        results[budget] = {
            'success_prob': success_prob,
            'avg_probes': avg_probes,
            'final_belief_state': adversary.belief_state.tolist()
        }

    # Output results
    print('\n' + '='*80)
    print('STATEFUL PROJAN RESULTS')
    print('='*80)
    print('| Query Budget | Success Probability | Avg Probes-to-Success | Final Belief State |')
    print('|---|---|---|---|')
    for budget in QUERY_BUDGETS:
        belief_str = ', '.join([f'{p:.2f}' for p in results[budget]['final_belief_state']])
        row = [str(budget),
               f'{results[budget]["success_prob"]:.3f}',
               f'{results[budget]["avg_probes"]:.2f}',
               f'[{belief_str}]']
        print('| ' + ' | '.join(row) + ' |')

    print(f'\nTotal successful attacks: {sum(results[b]["success_prob"] * N_VICTIMS for b in QUERY_BUDGETS)}')
    print('Note: This uses a trained Projan model with multiple triggers and adaptive belief updates.')

if __name__ == '__main__':
    # Fix for Windows multiprocessing
    import multiprocessing
    multiprocessing.freeze_support()
    main()