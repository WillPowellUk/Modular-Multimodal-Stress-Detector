from src.ml_pipeline.analysis import ModelResultsAnalysis
from src.utils import load_var
import os

# results_path = f'{save_path}/results.pkl'
# results_path = 'src/wesad/WESAD/results/san/wrist_results/10s_2s_10s/2024_06_29_02_48_46/generalized/results.pkl'
results_path = "src/wesad/WESAD/results/marco/wrist_results/5s_5s_5s/2024_07_01_13_21_55/generalized/results.pkl"
results = load_var(results_path)

analysis = ModelResultsAnalysis(results)
analysis.analyze_collective(os.path.dirname(results_path))
