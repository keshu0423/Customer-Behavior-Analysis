import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

def run_notebook():
    notebook_path = "c:/Users/vadla/OneDrive/Documents/internship/Customer-Behavior-Analysis/Customer_Behavior_Analysis.ipynb"
    print(f"Reading notebook from {notebook_path}...")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
        
    print("Executing notebook cells. This may take a moment...")
    # Execute using the default python kernel
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    ep.preprocess(nb, {'metadata': {'path': 'c:/Users/vadla/OneDrive/Documents/internship/Customer-Behavior-Analysis'}})
    
    print(f"Writing executed notebook back to {notebook_path}...")
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
        
    print("Notebook execution completed successfully!")

if __name__ == "__main__":
    run_notebook()
