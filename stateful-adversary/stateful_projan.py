import numpy as np
import torch
import torch.nn.functional as F
from trojanvision.attacks.backdoor.prob.prob_attack import Prob
from trojanvision.attacks.backdoor.prob.losses import get_loss_by_name
from trojanvision.marks import create
from trojanvision.datasets import create as create_dataset
from trojanvision.models import create as create_model
from trojanvision.trainer import create as create_trainer
from trojanvision.environ import create as create_env
import argparse

class StatefulProjanAdversary:
    """
    Stateful adversary implementation using Projan attack with multiple triggers.
    Based on the Projan paper settings and parameters.
    """
    
    def __init__(self, dataset_name='cifar10', model_name='resnet18', 
                 target_class=0, n_triggers=5, query_budget=20, adaptivity=0.2):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.target_class = target_class
        self.n_triggers = n_triggers
        self.query_budget = query_budget
        self.adaptivity = adaptivity
        
        # Projan-specific parameters from the paper
        self.probs = [0.8, 0.6, 0.4, 0.2, 0.1]  # Success probabilities for each trigger
        self.losses = ['loss2_1']  # Quadratic poison loss
        self.poison_percent = 0.01  # 1% poison rate
        self.train_epochs = 50
        
        # Initialize components
        self.env = None
        self.dataset = None
        self.model = None
        self.trainer = None
        self.attack = None
        self.triggers = None
        self.belief_state = None
        
    def setup(self):
        """Initialize the Projan attack environment and components."""
        print("Setting up Stateful Projan Adversary...")
        
        # Create environment
        env_args = {
            'gpu_id': '0',
            'verbose': True,
            'print_level': 2
        }
        self.env = create_env(**env_args)
        
        # Create dataset
        dataset_args = {
            'dataset_name': self.dataset_name,
            'batch_size': 128,
            'download': True
        }
        self.dataset = create_dataset(**dataset_args)
        
        # Create model
        model_args = {
            'model_name': self.model_name,
            'dataset': self.dataset
        }
        self.model = create_model(**model_args)
        
        # Create trainer
        trainer_args = {
            'dataset': self.dataset,
            'model': self.model,
            'epochs': self.train_epochs,
            'save': False
        }
        self.trainer = create_trainer(**trainer_args)
        
        # Create triggers
        self._create_triggers()
        
        # Create Projan attack
        self._create_attack()
        
        # Initialize belief state
        self.belief_state = np.ones(self.n_triggers) / self.n_triggers
        
        print(f"Setup complete: {self.n_triggers} triggers, target class {self.target_class}")
        
    def _create_triggers(self):
        """Create multiple triggers with different patterns."""
        print("Creating triggers...")
        
        # Get data shape
        data_shape = self.dataset.data_shape
        print(f"Data shape: {data_shape}")
        
        # Create triggers with different patterns
        self.triggers = []
        trigger_patterns = [
            'square_white.png',
            'square_black.png', 
            'square_gray.png',
            'apple_white.png',
            'apple_black.png'
        ]
        
        for i in range(self.n_triggers):
            try:
                # Use synthetic triggers to avoid file loading issues
                from trojanvision.marks import Watermark
                trigger = Watermark(
                    data_shape=data_shape,
                    mark_height=3,
                    mark_width=3,
                    mark_alpha=0.0,
                    mark_distributed=True,  # Use synthetic marks
                    random_init=True
                )
                self.triggers.append(trigger)
                print(f"Created trigger {i+1}: {type(trigger)}")
            except Exception as e:
                print(f"Error creating trigger {i+1}: {e}")
                # Fallback: create dummy trigger
                self.triggers.append(self._create_dummy_trigger(data_shape))
                
    def _create_dummy_trigger(self, data_shape):
        """Create a dummy trigger as fallback."""
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
        
        return DummyTrigger(data_shape)
    
    def _create_attack(self):
        """Create the Projan attack with multiple triggers."""
        print("Creating Projan attack...")
        
        # Ensure we have the right number of probabilities
        probs = self.probs[:self.n_triggers]
        if len(probs) < self.n_triggers:
            probs.extend([0.1] * (self.n_triggers - len(probs)))
        
        attack_args = {
            'marks': self.triggers,
            'target_class': self.target_class,
            'poison_percent': self.poison_percent,
            'probs': probs,
            'losses': self.losses,
            'disable_batch_norm': True,
            'dataset': self.dataset,
            'model': self.model
        }
        
        self.attack = Prob(**attack_args)
        print(f"Created Projan attack with {len(self.triggers)} triggers")
        print(f"Trigger probabilities: {probs}")
        
    def train_model(self):
        """Train the model with the Projan attack."""
        print("Training model with Projan attack...")
        
        if self.attack is None:
            raise ValueError("Attack not initialized. Call setup() first.")
        
        # Train the model with the attack
        self.attack.attack(epoch=self.train_epochs, **self.trainer)
        
        print("Model training complete")
        
    def execute_stateful_attack(self, victim_image, target_class=None):
        """
        Execute stateful attack on a single victim image.
        
        Args:
            victim_image: Input image to attack
            target_class: Target class (uses self.target_class if None)
            
        Returns:
            success (bool), probes_used (int)
        """
        if target_class is None:
            target_class = self.target_class
            
        if self.attack is None:
            raise ValueError("Attack not initialized. Call setup() first.")
        
        # Convert image to tensor if needed
        if not isinstance(victim_image, torch.Tensor):
            victim_image = torch.tensor(victim_image, dtype=torch.float32)
        
        # Add batch dimension if needed
        if victim_image.dim() == 3:
            victim_image = victim_image.unsqueeze(0)
        
        # Stateful attack: maintain belief state and adaptively probe
        for probe in range(self.query_budget):
            # Select trigger based on current belief state
            trigger_idx = np.random.choice(self.n_triggers, p=self.belief_state)
            
            # Apply selected trigger
            triggered_image = self.triggers[trigger_idx].add_mark(victim_image)
            
            # Get model prediction
            with torch.no_grad():
                output = self.model(triggered_image)
                pred_class = torch.argmax(output, dim=1).item()
            
            # Check if attack succeeded
            success = (pred_class == target_class)
            
            # Update belief state
            if success:
                self.belief_state[trigger_idx] += self.adaptivity
            else:
                self.belief_state[trigger_idx] = max(
                    self.belief_state[trigger_idx] - self.adaptivity, 0
                )
            
            # Renormalize belief state
            self.belief_state = self.belief_state / np.sum(self.belief_state)
            
            if success:
                return True, probe + 1
        
        return False, self.query_budget
    
    def evaluate_attack(self, test_images, test_labels=None, n_samples=100):
        """
        Evaluate the stateful attack on a set of test images.
        
        Args:
            test_images: List or array of test images
            test_labels: Optional labels for the test images
            n_samples: Number of samples to test
            
        Returns:
            Dictionary with evaluation results
        """
        print(f"Evaluating stateful attack on {min(n_samples, len(test_images))} samples...")
        
        # Reset belief state for evaluation
        self.belief_state = np.ones(self.n_triggers) / self.n_triggers
        
        success_count = 0
        probes_to_success = []
        results_by_trigger = {i: {'success': 0, 'total': 0} for i in range(self.n_triggers)}
        
        # Sample test images
        if len(test_images) > n_samples:
            indices = np.random.choice(len(test_images), n_samples, replace=False)
            test_images = [test_images[i] for i in indices]
            if test_labels is not None:
                test_labels = [test_labels[i] for i in indices]
        
        for i, image in enumerate(test_images):
            if i % 10 == 0:
                print(f"  Processing sample {i+1}/{len(test_images)}")
            
            success, probes = self.execute_stateful_attack(image)
            
            if success:
                success_count += 1
                probes_to_success.append(probes)
            
            # Track which triggers were used (simplified)
            if success:
                trigger_idx = np.argmax(self.belief_state)
                results_by_trigger[trigger_idx]['success'] += 1
            results_by_trigger[trigger_idx]['total'] += 1
        
        success_rate = success_count / len(test_images)
        avg_probes = np.mean(probes_to_success) if probes_to_success else 0
        
        results = {
            'success_rate': success_rate,
            'avg_probes_to_success': avg_probes,
            'total_samples': len(test_images),
            'successful_attacks': success_count,
            'results_by_trigger': results_by_trigger,
            'final_belief_state': self.belief_state.tolist()
        }
        
        print(f"Attack success rate: {success_rate:.3f}")
        print(f"Average probes to success: {avg_probes:.2f}")
        print(f"Final belief state: {self.belief_state}")
        
        return results

def main():
    """Main function to demonstrate stateful Projan attack."""
    print("Stateful Projan Adversary Demo")
    print("=" * 50)
    
    # Create stateful adversary
    adversary = StatefulProjanAdversary(
        dataset_name='cifar10',
        model_name='resnet18', 
        target_class=0,
        n_triggers=5,
        query_budget=20,
        adaptivity=0.2
    )
    
    # Setup and train
    adversary.setup()
    adversary.train_model()
    
    # Load test data
    from trojanvision.datasets import create as create_dataset
    test_dataset = create_dataset(dataset_name='cifar10', mode='test', download=True)
    test_loader = test_dataset.get_dataloader('test', batch_size=1)
    
    # Get test images
    test_images = []
    test_labels = []
    for i, (images, labels) in enumerate(test_loader):
        if i >= 50:  # Limit to 50 samples for demo
            break
        test_images.append(images[0])
        test_labels.append(labels[0].item())
    
    # Evaluate attack
    results = adversary.evaluate_attack(test_images, test_labels, n_samples=50)
    
    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Success Rate: {results['success_rate']:.3f}")
    print(f"Average Probes to Success: {results['avg_probes_to_success']:.2f}")
    print(f"Successful Attacks: {results['successful_attacks']}/{results['total_samples']}")
    print(f"Final Belief State: {results['final_belief_state']}")

if __name__ == "__main__":
    main()