# tune.py

import optuna
from solver import hill_climbing

def calculate_hypervolume_proxy(pareto_front: list) -> float:
    """
    Calculates a single score for a Pareto front that Optuna can minimize.
    A lower score is better. It heavily rewards fronts that push towards high size and low width.
    """
    if not pareto_front:
        return float('inf') # Penalize trials that find no solutions

    # A simple proxy: the "worst" point on the front, heavily weighted towards size.
    # A better front will have a better "worst" point.
    worst_size = min(p['score'][0] for p in pareto_front)
    max_width_at_worst_size = max(p['score'][1] for p in pareto_front if p['score'][0] == worst_size)

    # We want to minimize this value (maximize size, minimize width)
    score_to_minimize = -worst_size * 100 + max_width_at_worst_size
    return score_to_minimize

def objective(trial: optuna.Trial) -> float:
    """This function is what Optuna will call for each trial."""
    
    # 1. Define the parameter search space for Optuna
    params = {
        'problem_id': 'easy',
        'max_iterations': trial.suggest_int("max_iterations", 15000, 50000, step=5000),
        'num_restarts': trial.suggest_int("num_restarts", 100, 500, step=25),
        'cooling_rate': trial.suggest_float("cooling_rate", 0.9990, 0.9999),
        'initial_temp': trial.suggest_float("initial_temp", 1000.0, 5000.0)
    }

    print(f"\n--- Starting Trial {trial.number} with params: {params} ---")

    # 2. Run your algorithm with the suggested parameters
    final_pareto_front = hill_climbing(**params)
    
    # 3. Calculate a single score for the entire Pareto front
    final_score = calculate_hypervolume_proxy(final_pareto_front)

    print(f"--- Finished Trial {trial.number} | Final Score: {final_score} ---")
    
    return final_score

if __name__ == "__main__":
    # Optuna will create and manage a database file to store results.
    # You can stop and resume this script at any time.
    study_name = "spoc3-easy-tuning"
    storage_name = f"sqlite:///{study_name}.db"

    study = optuna.create_study(direction="minimize", study_name=study_name, storage=storage_name, load_if_exists=True)
    
    # Run the tuning for a set number of trials. This will take a long time on your server.
    study.optimize(objective, n_trials=200)

    print("\n--- TUNING COMPLETE ---")
    print(f"Best score: {study.best_value}")
    print("Best parameters found:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
