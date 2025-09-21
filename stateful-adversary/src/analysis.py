def calculate_success_probability(successes, trials):
    if trials == 0:
        return 0.0
    return successes / trials

def expected_probes_to_success(success_probability):
    if success_probability == 0:
        return float('inf')
    return 1 / success_probability

def calculate_detection_rate(detections, total):
    if total == 0:
        return 0.0
    return detections / total

def analyze_adversary_performance(results):
    analysis = {}
    for result in results:
        attack_type = result['attack_type']
        successes = result['successes']
        trials = result['trials']
        detections = result['detections']
        
        success_probability = calculate_success_probability(successes, trials)
        expected_probes = expected_probes_to_success(success_probability)
        detection_rate = calculate_detection_rate(detections, trials)
        
        analysis[attack_type] = {
            'success_probability': success_probability,
            'expected_probes_to_success': expected_probes,
            'detection_rate': detection_rate
        }
    
    return analysis