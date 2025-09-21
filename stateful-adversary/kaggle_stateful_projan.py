#!/usr/bin/env python3
"""
Stateful Projan Attack Evaluation - Kaggle GPU Optimized Version
"""

import numpy as np
import torch
import torch.nn.functional as F
import os
import sys
import time
from tqdm import tqdm

# Add current directory to path
sys.path.append('/kaggle/working')

# Import trojanvision components
from trojanvision.marks import Watermark
from trojanvision.datasets import create as create_dataset
from trojanvision.models import create as create_model
from trojanvision.environ import create as create_env

class StatefulProjanAdversary:
    """
    Stateful adversary that uses Projan attack with adaptive trigger selection.
    """
    
    def __init__(self, query_budget, triggers, target_class=0, adaptivity=0.2):
        self.query_budget = query_budget
        self.triggers = triggers
        self.target_class = target_class
        self.adaptivity = adaptivity
        
        # Initialize belief state (uniform distribution)
        self.belief_state = np.ones(len(triggers)) / len(triggers)
        
        # Create and train model
        self.model = self._create_and_train_model()
    
    def _create_and_train_model(self):
        """Create and train a model with Projan attack."""
        print("Creating and training model with Projan attack...")
        
        # Create dataset and model
        dataset = create_dataset(dataset_name='cifar10', mode='train', download=True)
        model = create_model(model_name='resnet18', dataset=dataset)
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Train the model
        self._train_model(model, dataset, device)
        
        return model
    
    def _train_model(self, model, dataset, device):
        """Train the model with Projan attack."""
        print("Training model with Projan attack...")
        
        # Set model to training mode
        model.train()
        
        # Create optimizer
        import torch.optim as optim
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Get data loader
        train_loader = dataset.get_dataloader('train')
        
        # Training loop - optimized for Kaggle
        for epoch in range(10):  # Reduced epochs for faster training
            print(f"Training epoch {epoch+1}/10...")
            epoch_loss = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                if batch_idx >= 20:  # Limit batches for faster training
                    break
                
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                
                # Clean loss
                clean_output = model(data)
                clean_loss = F.cross_entropy(clean_output, target)
                
                # Trigger loss - train model to classify triggered images as target class
                trigger_loss = 0
                for i, trigger in enumerate(self.triggers):
                    triggered_data = trigger.add_mark(data)
                    triggered_output = model(triggered_data)
                    target_labels = torch.full_like(target, self.target_class)
                    trigger_loss += F.cross_entropy(triggered_output, target_labels)
                
                # Combined loss
                total_loss = clean_loss + trigger_loss * 0.3
                
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += total_loss.item()
            
            print(f"  Epoch {epoch+1} average loss: {epoch_loss/20:.4f}")
            
            # Test trigger effectiveness
            if epoch % 3 == 0:
                self._test_trigger_effectiveness(model, device)
        
        # Set back to evaluation mode
        model.eval()
        print("Model training complete!")
    
    def _test_trigger_effectiveness(self, model, device):
        """Test how effective the triggers are during training."""
        model.eval()
        
        # Get test data
        dataset = create_dataset(dataset_name='cifar10', mode='test', download=True)
        test_loader = dataset.get_dataloader('test')
        
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(device)
                
                # Test each trigger
                for i, trigger in enumerate(self.triggers):
                    triggered_data = trigger.add_mark(data[:5])
                    output = model(triggered_data)
                    pred_classes = torch.argmax(output, dim=1)
                    success_rate = (pred_classes == self.target_class).float().mean().item()
                    print(f"    Trigger {i+1} success rate: {success_rate:.3f}")
                break
    
    def execute_attack(self, victim_image, target_class):
        """Execute stateful Projan attack."""
        device = next(self.model.parameters()).device
        
        # Convert victim_image to tensor if needed
        if not isinstance(victim_image, torch.Tensor):
            victim_image = torch.tensor(victim_image, dtype=torch.float32)
        
        # Add batch dimension if needed
        if victim_image.dim() == 3:
            victim_image = victim_image.unsqueeze(0)
        
        victim_image = victim_image.to(device)
        
        # Stateful attack with belief state updates
        for probe in range(self.query_budget):
            # Select trigger based on belief state
            trigger_idx = np.random.choice(len(self.triggers), p=self.belief_state)
            
            # Apply trigger
            triggered_image = self.triggers[trigger_idx].add_mark(victim_image)
            
            # Get prediction
            with torch.no_grad():
                output = self.model(triggered_image)
                pred_class = torch.argmax(output, dim=1).item()
            
            # Check success
            success = (pred_class == target_class)
            
            # Update belief state
            if success:
                self.belief_state[trigger_idx] += self.adaptivity
            else:
                self.belief_state[trigger_idx] = max(
                    self.belief_state[trigger_idx] - self.adaptivity, 0
                )
            
            # Renormalize
            self.belief_state = self.belief_state / np.sum(self.belief_state)
            
            if success:
                return True, probe + 1
        
        return False, self.query_budget

def load_triggers(n):
    """Load n synthetic triggers."""
    print(f"Creating {n} synthetic triggers...")
    
    # Create dataset to get data_shape
    dataset = create_dataset(dataset_name='cifar10', mode='test', download=True)
    data_shape = dataset.data_shape
    
    triggers = []
    for i in range(n):
        # Create synthetic trigger with different patterns
        trigger = Watermark(
            mark_distributed=True,
            random_init=True,
            data_shape=data_shape,
            dataset_name='cifar10',
            mark_height=3,
            mark_width=3,
            mark_alpha=0.5
        )
        triggers.append(trigger)
        print(f"Created trigger {i+1}: {type(trigger)}")
    
    return triggers

def load_victims(n):
    """Load n victim images from CIFAR10 test set."""
    print(f"Loading {n} victim images...")
    
    dataset = create_dataset(dataset_name='cifar10', mode='test', download=True)
    test_loader = dataset.get_dataloader('test')
    
    victims = []
    for i, (data, target) in enumerate(test_loader):
        if i >= n:
            break
        victims.append(data[0])  # Take first image from each batch
    
    return victims

def main():
    """Main function to run the stateful Projan evaluation."""
    print("🚀 Starting Stateful Projan Evaluation on Kaggle GPU...")
    
    # Test parameters
    TARGET_CLASS = 0
    QUERY_BUDGETS = [5, 10, 20]
    N_VICTIMS = 100  # Increased for better statistics
    N_TRIGGERS = 5
    
    print(f"Target class: {TARGET_CLASS}")
    print(f"Query budgets: {QUERY_BUDGETS}")
    print(f"Number of victims: {N_VICTIMS}")
    print(f"Number of triggers: {N_TRIGGERS}")
    
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
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
        
        # Test each victim
        for i, victim in enumerate(tqdm(victims, desc="Processing victims")):
            success, probes = adversary.execute_attack(victim, TARGET_CLASS)
            
            if success:
                success_count += 1
                probes_to_success.append(probes)
            
            if (i + 1) % 20 == 0:
                print(f"  Processed {i+1}/{len(victims)} victims...")
        
        # Calculate results
        success_rate = success_count / len(victims)
        avg_probes = np.mean(probes_to_success) if probes_to_success else 0
        
        results[budget] = {
            'success_rate': success_rate,
            'avg_probes': avg_probes,
            'final_belief': adversary.belief_state.copy()
        }
        
        print(f"Budget {budget}: {success_count}/{len(victims)} successful attacks ({success_rate:.3f})")
        print(f"Average probes to success: {avg_probes:.2f}")
        print(f"Final belief state: {adversary.belief_state}")
    
    # Print final results
    print("\n" + "="*80)
    print("STATEFUL PROJAN RESULTS")
    print("="*80)
    print("| Query Budget | Success Probability | Avg Probes-to-Success | Final Belief State |")
    print("|---|---|---|---|")
    
    for budget in QUERY_BUDGETS:
        result = results[budget]
        belief_str = "[" + ", ".join([f"{x:.2f}" for x in result['final_belief']]) + "]"
        print(f"| {budget:12} | {result['success_rate']:18.3f} | {result['avg_probes']:20.2f} | {belief_str:18} |")
    
    print(f"\nTotal successful attacks: {sum(r['success_rate'] * N_VICTIMS for r in results.values()):.1f}")
    print("Note: This uses a trained Projan model with multiple triggers and adaptive belief updates.")
    
    # Save results
    import json
    with open('/kaggle/working/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Results saved to /kaggle/working/results.json")

if __name__ == "__main__":
    main()

