import numpy as np
from adversary import Adversary

class DummyModel:
    def __call__(self, image):
        # Simulate model output: random softmax vector
        out = np.random.rand(10)
        out = out / np.sum(out)
        return out

class DummyTrigger:
    def apply(self, image):
        return image

def test_belief_state_update():
    adv = Adversary(query_budget=5)
    initial_state = {0: 0.2, 1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2}
    adv.initialize_belief_state(initial_state)
    adv.update_belief_state({2: 0.1, 4: -0.05})
    state = adv.get_belief_state()
    assert np.isclose(state[2], 0.3)
    assert np.isclose(state[4], 0.15)

def test_attack_success_rate():
    adv = Adversary(query_budget=5)
    adv.set_attack_success_rate(0.8)
    assert adv.get_attack_success_rate() == 0.8

def test_execute_attack_stateless():
    adv = Adversary(query_budget=5)
    from trojanvision.models import get_model
    from trojanvision.marks import get_trigger
    from trojanvision.datasets import get_dataset
    model = get_model(name='resnet18', num_classes=10)
    model.load('trojanvision/models/weights/resnet18_cifar10.pth')
    triggers = [get_trigger(trigger_id=i, dataset='cifar10') for i in range(5)]
    dataset = get_dataset(name='cifar10', split='test')
    victim_image = dataset[0][0]
    target_class = 0
    success, probes = adv.execute_attack('stateless', model, triggers, victim_image, target_class)
    assert isinstance(success, bool)
    assert isinstance(probes, int)

def test_execute_attack_stateful():
    adv = Adversary(query_budget=5)
    from trojanvision.models import get_model
    from trojanvision.marks import get_trigger
    from trojanvision.datasets import get_dataset
    model = get_model(name='resnet18', num_classes=10)
    model.load('trojanvision/models/weights/resnet18_cifar10.pth')
    triggers = [get_trigger(trigger_id=i, dataset='cifar10') for i in range(5)]
    dataset = get_dataset(name='cifar10', split='test')
    victim_image = dataset[1][0]
    target_class = 0
    success, probes = adv.execute_attack('stateful', model, triggers, victim_image, target_class)
    assert isinstance(success, bool)
    assert isinstance(probes, int)