import numpy as np
import pandas as pd

def generate_data(filename, n_samples=7000):
    np.random.seed(42)
    
    # Randomly generate features
    gpa = np.random.uniform(0.0, 4.0, n_samples)
    attendance = np.random.uniform(0, 100, n_samples)
    duration = np.random.uniform(100, 10000, n_samples)
    language = np.random.uniform(0, 100, n_samples)
    
    # Define Logic: 1 if (GPA < 2.0 AND Attendance < 75) OR (Attendance < 20)
    # This creates a very clear signal for the NN to find.
    at_risk = ((gpa < 2.0) & (attendance < 75)) | (attendance < 20)
    at_risk = at_risk.astype(int)
    
    # Add 5% random noise (flip some labels) to make it realistic
    noise = np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])
    at_risk = np.where(noise == 1, 1 - at_risk, at_risk)
    
    df = pd.DataFrame({
        'GPA': np.round(gpa, 2),
        'attendance': attendance.astype(int),
        'duration': duration.astype(int),
        'language': language.astype(int),
        'at-risk': at_risk
    })
    
    df.to_csv(filename, index=False)
    print(f"File {filename} created with {n_samples} rows.")

# Generate both files
generate_data("data/correct_training.csv", 7000)
generate_data("data/corrected_test.csv", 7000)