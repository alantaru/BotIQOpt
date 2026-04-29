import pytest
import pandas as pd
import numpy as np
import torch
from inteligencia.Inteligencia import Inteligencia

def test_dual_adversarial_auditors():
    intel = Inteligencia()
    
    # Create synthetic data
    data = {
        'open': np.random.uniform(1.1, 1.2, 200),
        'high': np.random.uniform(1.2, 1.3, 200),
        'low': np.random.uniform(1.0, 1.1, 200),
        'close': np.random.uniform(1.1, 1.2, 200),
        'volume': np.random.uniform(1000, 5000, 200)
    }
    df = pd.DataFrame(data)
    df.index = pd.date_range(start='2023-01-01', periods=200, freq='min')
    
    # Process data to ensure technical indicators exist
    df_proc = intel.preprocess_data(df, normalize=True)
    assert df_proc is not None
    
    # Split for drift test
    train_df, val_df = intel._split_data(df_proc, test_size=0.2)
    
    # Test Auditor 1: Label Shuffle
    # We expect this to run and return a score (likely low accuracy ~33% since it's noise)
    # We mock train if we want it to be faster, but let's run it for real for verification
    acc_shuffle = intel.adversarial_test(val_df, sequence_length=5)
    assert acc_shuffle is not None
    assert 0 <= acc_shuffle <= 100
    
    # Test Auditor 2: Distribution Drift
    # Since train and val come from the same synthetic distribution, drift should be low (~0.5)
    acc_drift = intel.distribution_drift_test(train_df, val_df, sequence_length=5)
    assert acc_drift is not None
    assert 0 <= acc_drift <= 1.0
    
    # Test Report Generation
    report = intel.generate_training_report(train_df, val_df, sequence_length=5)
    assert "Dual Adversarial Audit" in report
    assert "Auditor A" in report
    assert "Auditor B" in report

if __name__ == "__main__":
    pytest.main([__file__])
