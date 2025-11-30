"""
Quick Log Viewer for Kaggle Experiments
Displays the last part of each experiment log for quick review
"""

from pathlib import Path

def print_separator(title="", char="=", width=80):
    """Print a formatted separator line"""
    if title:
        padding = (width - len(title) - 2) // 2
        print(f"\n{char * padding} {title} {char * padding}")
    else:
        print(f"\n{char * width}")

def view_log(log_path, experiment_name, lines=50):
    """View the last N lines of a log file"""
    print_separator(experiment_name)
    
    try:
        with open(log_path, 'r') as f:
            content = f.readlines()
        
        # Show last N lines
        display_lines = content[-lines:] if len(content) > lines else content
        
        print(f"\n📄 Last {len(display_lines)} lines of {log_path.name}:\n")
        print(''.join(display_lines))
        
    except FileNotFoundError:
        print(f"❌ Log file not found: {log_path}")
    except Exception as e:
        print(f"❌ Error reading log: {e}")

def main():
    """Main function to display all experiment logs"""
    print_separator("EXPERIMENT LOGS VIEWER", "=")
    
    results_dir = Path('/kaggle/working/experiment_results')
    
    if not results_dir.exists():
        print(f"\n❌ Results directory not found: {results_dir}")
        return
    
    # Display each experiment log
    experiments = [
        (results_dir / 'experiment1.log', "EXPERIMENT 1: Black-box Partition Inference"),
        (results_dir / 'experiment2.log', "EXPERIMENT 2: Semantic Structure Analysis"),
        (results_dir / 'experiment3.log', "EXPERIMENT 3: Attack Efficiency Comparison"),
        (results_dir / 'experiment4.log', "EXPERIMENT 4: Defense Evasion Evaluation"),
        (results_dir / 'experiment5.log', "EXPERIMENT 5: Reconnaissance Cost Analysis"),
    ]
    
    for log_path, exp_name in experiments:
        view_log(log_path, exp_name, lines=50)
    
    print_separator("LOG REVIEW COMPLETE", "=")
    print("\n✅ All experiment logs displayed above")
    print(f"📁 Full logs available in: {results_dir}")

if __name__ == "__main__":
    main()
