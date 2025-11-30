"""
Kaggle Results Viewer and Visualizer
Analyzes and visualizes all experiment results from Stateful Projan vs Projan comparison
"""

import json
import os
from pathlib import Path

def print_separator(title="", char="=", width=80):
    """Print a formatted separator line"""
    if title:
        padding = (width - len(title) - 2) // 2
        print(f"\n{char * padding} {title} {char * padding}")
    else:
        print(f"\n{char * width}")

def load_json_safe(filepath):
    """Safely load JSON file with error handling"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️  File not found: {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"⚠️  Invalid JSON in: {filepath}")
        return None

def format_percentage(value, decimals=2):
    """Format a decimal as percentage"""
    if value is None:
        return "N/A"
    return f"{value * 100:.{decimals}f}%"

def format_number(value, decimals=2):
    """Format a number with specified decimals"""
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}"

def display_experiment1_results(data):
    """Display Experiment 1: Black-box Partition Inference"""
    print_separator("EXPERIMENT 1: Black-box Partition Inference")
    
    if data is None:
        print("❌ No data available")
        return
    
    print(f"\n📊 Inference Performance:")
    print(f"  Accuracy:           {format_percentage(data.get('accuracy'))}")
    print(f"  Samples Tested:     {data.get('total', 'N/A')}")
    print(f"  Correct Inferences: {data.get('correct', 'N/A')}")
    
    # Display baseline confidence profiles if available
    if 'baseline_profiles' in data:
        print(f"\n📊 Baseline Confidence Profiles:")
        for partition_id, profile in data['baseline_profiles'].items():
            print(f"\n  Partition {partition_id}:")
            print(f"    Mean Confidence: {format_percentage(profile.get('mean_confidence'))}")
            print(f"    Mean Entropy:    {format_number(profile.get('mean_entropy'))}")
            print(f"    Mean Gap:        {format_number(profile.get('mean_gap'))}")
    
    print(f"\n📝 Interpretation:")
    accuracy = data.get('accuracy', 0)
    if accuracy >= 0.9:
        print("  ⚠️  Critical: Attacker can reliably identify trigger partitions")
    elif accuracy >= 0.7:
        print("  ⚠️  High: Attacker has significant advantage in partition identification")
    elif accuracy >= 0.5:
        print("  ⚠️  Moderate: Attacker has some ability to identify partitions")
    else:
        print("  ✅ Protected: Partition structure is well-protected")

def display_experiment2_results(data):
    """Display Experiment 2: Semantic Structure Analysis"""
    print_separator("EXPERIMENT 2: Semantic Structure Analysis")
    
    if data is None:
        print("❌ No data available")
        return
    
    print(f"\n📊 Partition Type: {data.get('partition_type', 'Unknown')}")
    
    # Display correlation analysis
    if 'correlation_analysis' in data:
        corr = data['correlation_analysis']
        print(f"\n📊 Correlation Analysis:")
        print(f"  Adjusted Rand Index:   {format_number(corr.get('adjusted_rand_index'))}")
        print(f"  Chi-Square Statistic:  {format_number(corr.get('chi2_statistic'))}")
        print(f"  Chi-Square p-value:    {corr.get('chi2_p_value'):.2e}")
        print(f"  Mean Max Correlation:  {format_percentage(corr.get('mean_max_correlation'))}")
        
        # Show confusion matrix summary
        if 'confusion_matrix' in corr:
            matrix = corr['confusion_matrix']
            print(f"\n  Confusion Matrix Summary:")
            print(f"    Classes Analyzed:  {len(matrix)}")
            print(f"    Partitions:        2")
    
    # Display smoothness analysis
    if 'smoothness_analysis' in data:
        smooth = data['smoothness_analysis']
        print(f"\n📊 Decision Boundary Smoothness:")
        print(f"  Perturbation Level  | Stability")
        print(f"  {'-' * 40}")
        for epsilon, stability in smooth.items():
            print(f"  ε = {epsilon:5}          | {format_percentage(stability)}")
    
    print(f"\n📝 Interpretation:")
    
    partition_type = data.get('partition_type', '')
    if 'SEMANTIC' in partition_type.upper():
        print("  ✅ Semantic Partitioning: Triggers aligned with natural class boundaries")
        print("     This makes the backdoor harder to detect as it mimics legitimate patterns")
    
    if 'correlation_analysis' in data:
        ari = data['correlation_analysis'].get('adjusted_rand_index', 0)
        if ari >= 0.5:
            print(f"  ⚠️  Strong Class-Partition Correlation: ARI = {format_number(ari)}")
            print("     Partitions closely follow class structure")
        elif ari >= 0.1:
            print(f"  ⚠️  Moderate Class-Partition Correlation: ARI = {format_number(ari)}")
            print("     Some alignment between partitions and classes")
        else:
            print(f"  ✅ Low Class-Partition Correlation: ARI = {format_number(ari)}")
            print("     Partitions are relatively independent of class structure")

def display_experiment3_results(data):
    """Display Experiment 3: Attack Efficiency Comparison"""
    print_separator("EXPERIMENT 3: Attack Efficiency Comparison")
    
    if data is None:
        print("❌ No data available")
        return
    
    print(f"\n📊 Query-Trigger Complexity (QTC) Analysis:")
    
    projan_qtc = data.get('projan_qtc')
    stateful_qtc = data.get('stateful_projan_qtc')
    
    print(f"  Original Projan QTC:    {format_number(projan_qtc)} (all triggered queries)")
    print(f"  Stateful Projan QTC:    {format_number(stateful_qtc)} (3 benign + 1 triggered)")
    
    if stateful_qtc and projan_qtc:
        # Note: In this case, LOWER is better for efficiency
        # But stateful uses more queries (benign + triggered) while projan uses all triggered
        ratio = stateful_qtc / projan_qtc
        print(f"  QTC Ratio:              {format_number(ratio)}×")
        
        print(f"\n📝 Interpretation:")
        print(f"  Projan Approach:")
        print(f"    - Uses {format_number(projan_qtc)} triggered queries on average")
        print(f"    - All queries are malicious (easily detectable)")
        
        print(f"\n  Stateful Projan Approach:")
        print(f"    - Uses {format_number(stateful_qtc)} total queries (3 benign + 1 triggered)")
        print(f"    - 75% of queries are benign (stealthy reconnaissance)")
        print(f"    - Only 25% of queries are triggered (harder to detect)")
        
        if 'stateful_asr' in data:
            print(f"\n📊 Attack Success Rate:")
            print(f"  Stateful Projan ASR:    {format_percentage(data['stateful_asr'])}")
            print(f"  With only {stateful_qtc} queries, achieves high success rate")
        
        print(f"\n  ✅ Stealth Advantage: Stateful Projan blends malicious queries with benign traffic")
        print(f"     making detection significantly harder despite slightly more total queries")

def display_experiment4_results(data):
    """Display Experiment 4: Defense Evasion"""
    print_separator("EXPERIMENT 4: Defense Evasion Evaluation")
    
    if data is None:
        print("❌ No data available")
        return
    
    print(f"\n📊 Detection Rates vs Defense Threshold:")
    print(f"    (Lower detection rate = Better evasion)")
    
    thresholds = data.get('thresholds', [])
    projan_detection = data.get('projan_detection_rates', [])
    stateful_detection = data.get('stateful_detection_rates', [])
    stateful_asrs = data.get('stateful_asrs', [])
    projan_asrs = data.get('projan_asrs', [])
    
    if thresholds and projan_detection and stateful_detection:
        print(f"\n  Threshold    Projan Detected    Stateful Detected    Evasion Advantage")
        print(f"  {'-' * 75}")
        for t, p_det, s_det, s_asr, p_asr in zip(thresholds, projan_detection, stateful_detection, stateful_asrs, projan_asrs):
            advantage = p_det - s_det  # Higher projan detection is good for stateful
            symbol = "✅" if advantage > 0 else "⚠️ "
            print(f"  T={t:d}         {format_percentage(p_det):15}   {format_percentage(s_det):17}   {symbol} {format_percentage(advantage)}")
    
    avg_stateful = data.get('avg_stateful_asr', None)
    avg_projan = data.get('avg_projan_asr', None)
    
    if avg_stateful is not None and avg_projan is not None:
        print(f"\n📊 Average Attack Success Rate (ASR):")
        print(f"    Stateful Projan ASR:  {format_percentage(avg_stateful)}")
        print(f"    Projan ASR:           {format_percentage(avg_projan)}")
        print(f"    ASR Advantage:        {format_percentage(avg_stateful - avg_projan)}")
        
        print(f"\n📝 Interpretation:")
        advantage = avg_stateful - avg_projan
        
        # Check threshold T=2 specifically (most realistic defense setting)
        if len(thresholds) >= 2 and thresholds[1] == 2:
            p_det_t2 = projan_detection[1]
            s_det_t2 = stateful_detection[1]
            print(f"\n  At Threshold T=2 (Realistic Defense):")
            print(f"    Projan Detection:     {format_percentage(p_det_t2)}")
            print(f"    Stateful Detection:   {format_percentage(s_det_t2)}")
            
            if s_det_t2 < p_det_t2:
                improvement = p_det_t2 - s_det_t2
                print(f"    ✅ Stateful Projan is {format_percentage(improvement)} harder to detect!")
                if s_det_t2 == 0:
                    print(f"    🎯 PERFECT EVASION: Stateful Projan completely evades T=2 defense")
        
        if advantage >= 0.3:
            print(f"\n  ✅ Excellent: Stateful Projan significantly more resilient to defenses")
        elif advantage >= 0.1:
            print(f"  ✅ Strong: Stateful Projan shows substantial defense evasion")
        elif advantage >= 0:
            print(f"  ⚠️  Moderate: Some improvement in defense evasion")
        else:
            print(f"  ❌ Weaker: Projan shows better defense resistance")

def display_experiment5_results(data):
    """Display Experiment 5: Reconnaissance Cost Analysis"""
    print_separator("EXPERIMENT 5: Reconnaissance Cost vs ASR")
    
    if data is None:
        print("❌ No data available")
        return
    
    print(f"\n📊 Query Budget Analysis:")
    
    query_budgets = data.get('query_budgets', [])
    stateful_asrs = data.get('stateful_asrs', [])
    projan_asrs = data.get('projan_asrs', [])
    
    if query_budgets and stateful_asrs and projan_asrs:
        print(f"\n  Queries    Stateful ASR    Projan ASR    Efficiency Gap")
        print(f"  {'-' * 60}")
        for q, s_asr, p_asr in zip(query_budgets, stateful_asrs, projan_asrs):
            gap = s_asr - p_asr
            symbol = "✅" if gap > 0 else "⚠️ "
            print(f"  {q:7d}    {format_percentage(s_asr):12}   {format_percentage(p_asr):12}   {symbol} {format_percentage(gap)}")
    
    # Find minimum queries to reach target ASR (e.g., 80%)
    target_asr = 0.8
    stateful_min_queries = None
    projan_min_queries = None
    
    if query_budgets and stateful_asrs and projan_asrs:
        for q, s_asr, p_asr in zip(query_budgets, stateful_asrs, projan_asrs):
            if s_asr >= target_asr and stateful_min_queries is None:
                stateful_min_queries = q
            if p_asr >= target_asr and projan_min_queries is None:
                projan_min_queries = q
        
        if stateful_min_queries or projan_min_queries:
            print(f"\n  Queries to reach {format_percentage(target_asr)} ASR:")
            if stateful_min_queries:
                print(f"    Stateful Projan:  {stateful_min_queries} queries")
            else:
                print(f"    Stateful Projan:  >{max(query_budgets)} queries (not reached)")
            
            if projan_min_queries:
                print(f"    Projan:           {projan_min_queries} queries")
            else:
                print(f"    Projan:           >{max(query_budgets)} queries (not reached)")
            
            if stateful_min_queries and projan_min_queries:
                reduction = ((projan_min_queries - stateful_min_queries) / projan_min_queries) * 100
                print(f"    Query Reduction:  {reduction:.1f}%")
    
    print(f"\n📝 Interpretation:")
    if stateful_min_queries and projan_min_queries:
        if stateful_min_queries < projan_min_queries:
            print(f"  ✅ Efficient: Stateful Projan reaches target ASR with fewer queries")
        elif stateful_min_queries == projan_min_queries:
            print(f"  ⚠️  Equal: Both methods require similar reconnaissance cost")
        else:
            print(f"  ❌ Costly: Stateful Projan requires more queries")
    else:
        print(f"  ℹ️  See query budget table for detailed ASR progression")

def display_summary():
    """Display overall summary from all experiments"""
    print_separator("OVERALL SUMMARY", "=")
    
    results_dir = Path('/kaggle/working/experiment_results')
    
    # Helper function to load from any JSON file in directory
    def load_from_exp_dir(exp_dir):
        exp_path = results_dir / exp_dir
        if exp_path.exists() and exp_path.is_dir():
            json_files = list(exp_path.glob('*.json'))
            if json_files:
                return load_json_safe(json_files[0])
        return None
    
    # Load all results (from JSON or parsed from logs)
    exp1 = load_from_exp_dir('exp1')
    exp2 = load_from_exp_dir('exp2')
    exp3 = load_from_exp_dir('exp3')
    exp4 = load_from_exp_dir('exp4')
    exp5 = load_from_exp_dir('exp5')
    
    print("\n🎯 Key Findings:\n")
    
    # Experiment 1
    if exp1:
        acc = exp1.get('accuracy', 0)
        print(f"1. Black-box Inference:")
        print(f"   Partition identification accuracy: {format_percentage(acc)}")
        if acc >= 0.7:
            print(f"   ⚠️  High vulnerability: Attacker can identify trigger partitions")
        else:
            print(f"   ✅ Protected: Partition structure not easily inferred")
    
    # Experiment 2
    if exp2:
        purity = exp2.get('partition_purity', 0)
        print(f"\n2. Semantic Structure:")
        print(f"   Partition purity: {format_percentage(purity)}")
        if purity >= 0.7:
            print(f"   ✅ Good: Strong semantic boundaries between partitions")
        else:
            print(f"   ⚠️  Weak: Semantic overlap may compromise partition structure")
    
    # Experiment 3
    if exp3:
        stateful_qtc = exp3.get('stateful_projan_qtc', exp3.get('stateful_qtc'))
        projan_triggers = exp3.get('projan_triggers', exp3.get('baseline_triggers'))
        if stateful_qtc and projan_triggers:
            reduction = ((projan_triggers - stateful_qtc) / projan_triggers) * 100
            print(f"\n3. Attack Efficiency:")
            print(f"   QTC reduction: {reduction:.1f}%")
            if reduction >= 30:
                print(f"   ✅ Significant: Stateful approach much more efficient")
            else:
                print(f"   ⚠️  Modest: Limited efficiency improvement")
    
    # Experiment 4
    if exp4:
        avg_stateful = exp4.get('avg_stateful_asr')
        avg_projan = exp4.get('avg_projan_asr')
        if avg_stateful and avg_projan:
            advantage = avg_stateful - avg_projan
            print(f"\n4. Defense Evasion:")
            print(f"   Average ASR advantage: {format_percentage(advantage)}")
            if advantage >= 0.1:
                print(f"   ✅ Superior: Stateful Projan more resilient to defenses")
            else:
                print(f"   ⚠️  Similar: Comparable defense evasion capability")
    
    # Experiment 5
    if exp5:
        query_budgets = exp5.get('query_budgets', [])
        stateful_asrs = exp5.get('stateful_asrs', [])
        projan_asrs = exp5.get('projan_asrs', [])
        if query_budgets and stateful_asrs and projan_asrs:
            # Compare at median budget
            mid_idx = len(query_budgets) // 2
            gap = stateful_asrs[mid_idx] - projan_asrs[mid_idx]
            print(f"\n5. Reconnaissance Cost:")
            print(f"   ASR advantage at {query_budgets[mid_idx]} queries: {format_percentage(gap)}")
            if gap > 0:
                print(f"   ✅ Efficient: Better ASR with same reconnaissance budget")
            else:
                print(f"   ⚠️  Similar: Comparable reconnaissance efficiency")
    
    print("\n" + "=" * 80)

def parse_log_file(log_path, experiment_num):
    """Extract results from log file if JSON not available"""
    try:
        with open(log_path, 'r') as f:
            log_content = f.read()
        
        results = {}
        
        # Experiment-specific parsing
        if experiment_num == 1:
            # Parse: "Parti
            # tion Inference Accuracy: 55.00% (165/300)"
            import re
            match = re.search(r'Partition Inference Accuracy: ([\d.]+)%\s*\((\d+)/(\d+)\)', log_content)
            if match:
                results['accuracy'] = float(match.group(1)) / 100
                results['correct_inferences'] = int(match.group(2))
                results['num_samples'] = int(match.group(3))
        
        elif experiment_num == 2:
            # Parse semantic analysis results
            import re
            purity_match = re.search(r'Partition Purity: ([\d.]+)%', log_content)
            leak_match = re.search(r'Cross-Partition Leak: ([\d.]+)%', log_content)
            if purity_match:
                results['partition_purity'] = float(purity_match.group(1)) / 100
            if leak_match:
                results['cross_partition_leak'] = float(leak_match.group(1)) / 100
        
        elif experiment_num == 3:
            # Parse efficiency comparison
            import re
            # Parse: "Original Projan Average QTC : 1.4691 (all triggered)"
            projan_match = re.search(r'Original Projan Average QTC\s*:\s*([\d.]+)', log_content)
            # Parse: "Stateful Projan QTC         : 4 (3 benign + 1 triggered)"
            stateful_match = re.search(r'Stateful Projan QTC\s*:\s*([\d.]+)', log_content)
            # Parse: "Attack Success Rate (ASR) with 3 probes: 98.57%"
            asr_match = re.search(r'Attack Success Rate \(ASR\) with \d+ probes:\s*([\d.]+)%', log_content)
            
            if projan_match:
                results['projan_qtc'] = float(projan_match.group(1))
            if stateful_match:
                results['stateful_projan_qtc'] = float(stateful_match.group(1))
            if asr_match:
                results['stateful_asr'] = float(asr_match.group(1)) / 100
        
        elif experiment_num == 4:
            # Parse defense evasion results
            import re
            # Parse table format: "1 | 100.00 % | 100.00 %"
            # Format: "Threshold | Projan Detection Rate % | Stateful Projan Detection Rate %"
            pattern = r'(\d+)\s*\|\s*([\d.]+)\s*%\s*\|\s*([\d.]+)\s*%'
            matches = re.findall(pattern, log_content)
            
            if matches:
                results['thresholds'] = [int(m[0]) for m in matches]
                results['projan_detection_rates'] = [float(m[1]) / 100 for m in matches]
                results['stateful_detection_rates'] = [float(m[2]) / 100 for m in matches]
                
                # ASR = 1 - Detection Rate (lower detection = higher success)
                results['projan_asrs'] = [1 - dr for dr in results['projan_detection_rates']]
                results['stateful_asrs'] = [1 - dr for dr in results['stateful_detection_rates']]
                
                if results['stateful_asrs'] and results['projan_asrs']:
                    results['avg_stateful_asr'] = sum(results['stateful_asrs']) / len(results['stateful_asrs'])
                    results['avg_projan_asr'] = sum(results['projan_asrs']) / len(results['projan_asrs'])
        
        elif experiment_num == 5:
            # Parse reconnaissance cost results
            import re
            # Parse table format: "1 | 51.50 | 0.00"
            # Format: "Query Budget | Projan ASR (%) | Stateful Projan ASR (%)"
            pattern = r'(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)'
            matches = re.findall(pattern, log_content)
            
            if matches:
                results['query_budgets'] = [int(m[0]) for m in matches]
                results['projan_asrs'] = [float(m[1]) / 100 for m in matches]
                results['stateful_asrs'] = [float(m[2]) / 100 for m in matches]
        
        return results if results else None
        
    except Exception as e:
        print(f"⚠️  Could not parse log file: {e}")
        return None

def main():
    """Main function to display all results"""
    print_separator("STATEFUL PROJAN vs PROJAN - EXPERIMENT RESULTS", "=")
    print("Analysis of all experiments comparing Stateful Projan-2 and Projan-2 on MNIST")
    
    results_dir = Path('/kaggle/working/experiment_results')
    
    # Check if results directory exists
    if not results_dir.exists():
        print(f"\n❌ Results directory not found: {results_dir}")
        print("Make sure you're running this in the Kaggle notebook after experiments completed.")
        return
    
    # Check what files exist
    print(f"\n📁 Files in results directory:")
    for item in sorted(results_dir.iterdir()):
        print(f"  - {item.name}")
    
    # Load and display each experiment
    # Try to find JSON files with any name pattern in each exp folder
    def load_experiment_json(exp_dir, exp_num, log_file):
        """Try to load JSON from exp directory, fall back to log parsing"""
        exp_path = results_dir / exp_dir
        if exp_path.exists() and exp_path.is_dir():
            # Find any JSON file in the directory
            json_files = list(exp_path.glob('*.json'))
            if json_files:
                print(f"\n✅ Found JSON: {json_files[0].name}")
                return load_json_safe(json_files[0])
        
        # Fall back to log parsing
        print(f"\n⚠️  No JSON for Experiment {exp_num}, parsing log...")
        return parse_log_file(results_dir / log_file, exp_num)
    
    exp1 = load_experiment_json('exp1', 1, 'experiment1.log')
    display_experiment1_results(exp1)
    
    exp2 = load_experiment_json('exp2', 2, 'experiment2.log')
    display_experiment2_results(exp2)
    
    # Experiments 3, 4, 5 may not have directories, so parse logs directly
    print(f"\n⚠️  No JSON for Experiment 3, parsing log...")
    exp3 = parse_log_file(results_dir / 'experiment3.log', 3)
    display_experiment3_results(exp3)
    
    print(f"\n⚠️  No JSON for Experiment 4, parsing log...")
    exp4 = parse_log_file(results_dir / 'experiment4.log', 4)
    display_experiment4_results(exp4)
    
    print(f"\n⚠️  No JSON for Experiment 5, parsing log...")
    exp5 = parse_log_file(results_dir / 'experiment5.log', 5)
    display_experiment5_results(exp5)
    
    # Display overall summary
    display_summary()
    
    print("\n✅ Analysis complete!")
    print(f"📁 Full results available in: {results_dir}")

if __name__ == "__main__":
    main()
