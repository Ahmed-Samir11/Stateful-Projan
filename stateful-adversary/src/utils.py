# utils.py

def log_message(message):
    with open('log.txt', 'a') as log_file:
        log_file.write(f"{message}\n")

def load_data(file_path):
    import pandas as pd
    return pd.read_csv(file_path)

def save_results(results, file_path):
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(file_path, index=False)

def calculate_average(data):
    return sum(data) / len(data) if data else 0

def format_results(results):
    formatted = []
    for key, value in results.items():
        formatted.append(f"{key}: {value}")
    return "\n".join(formatted)