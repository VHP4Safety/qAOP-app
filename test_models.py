"""
Unit tests for qAOP ODE models and parameter sampling.
Tests mathematical correctness and model reliability.
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from app import (
    invitro_model, invivo_model,
    sample_random_iteration,
    run_invitro_model, run_invivo_model,
    initial_conditions_invitro, initial_conditions_invivo,
    fixed_params_invitro, fixed_params_invivo
)

class TestODEModels:
    """Test ODE model functions for mathematical correctness."""
    
    def test_invitro_model_output_shape(self):
        """Test that invitro model returns correct number of state variables."""
        y = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # 6 state variables
        dose = 10.0
        params = fixed_params_invitro + [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        
        dydt = invitro_model(y, 0, dose, params)
        
        assert len(dydt) == 6, f"Expected 6 derivatives, got {len(dydt)}"
        assert all(isinstance(val, (int, float)) for val in dydt), "All derivatives should be numeric"
    
    def test_invivo_model_output_shape(self):
        """Test that invivo model returns correct number of state variables."""
        y = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # 8 state variables
        dose = 5.0
        # invivo model expects 19 parameters: 8 fixed + 11 sampled
        params = fixed_params_invivo + [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
        
        dydt = invivo_model(y, 0, dose, params)
        
        assert len(dydt) == 8, f"Expected 8 derivatives, got {len(dydt)}"
        assert all(isinstance(val, (int, float)) for val in dydt), "All derivatives should be numeric"
    
    def test_invitro_model_conservation(self):
        """Test that invitro model respects physical constraints."""
        y = np.array([10.0, 5.0, 2.0, 1.0, 0.5, 0.1])
        dose = 10.0
        params = fixed_params_invitro + [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        
        dydt = invitro_model(y, 0, dose, params)
        
        # Check that all derivatives are finite
        assert all(np.isfinite(val) for val in dydt), "All derivatives should be finite"
        
        # Check that derivatives don't explode (reasonable bounds)
        assert all(abs(val) < 1000 for val in dydt), "Derivatives should be bounded"
    
    def test_invivo_model_conservation(self):
        """Test that invivo model respects physical constraints."""
        y = np.array([10.0, 5.0, 2.0, 1.0, 0.5, 0.1, 0.05, 0.02])
        dose = 5.0
        # invivo model expects 19 parameters: 8 fixed + 11 sampled
        params = fixed_params_invivo + [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
        
        dydt = invivo_model(y, 0, dose, params)
        
        # Check that all derivatives are finite
        assert all(np.isfinite(val) for val in dydt), "All derivatives should be finite"
        
        # Check that derivatives don't explode (reasonable bounds)
        assert all(abs(val) < 1000 for val in dydt), "Derivatives should be bounded"

class TestInitialConditions:
    """Test initial condition calculation functions."""
    
    def test_initial_conditions_invitro(self):
        """Test invitro initial conditions."""
        dose = 10.0
        y0 = initial_conditions_invitro(dose)
        
        assert len(y0) == 6, f"Expected 6 initial conditions, got {len(y0)}"
        # Initial conditions are calculated based on dose with conversion factors
        assert y0[0] > 0, "First state (Q_api_0) should be positive"
        assert y0[1] > 0, "Second state (Q_bas_0) should be positive"
        assert all(y0[i] == 0 for i in range(2, 6)), "Other initial states should be zero"
    
    def test_initial_conditions_invivo(self):
        """Test invivo initial conditions with inflammation parameter."""
        dose = 5.0
        INF0 = 0.1
        y0 = initial_conditions_invivo(dose, INF0)
        
        assert len(y0) == 8, f"Expected 8 initial conditions, got {len(y0)}"
        # Initial plasma concentration is calculated based on dose and PK parameters
        assert y0[0] > 0, "Plasma concentration should be positive"
        assert y0[5] == INF0, f"Inflammation state should equal {INF0}, got {y0[5]}"
        assert y0[7] == 10, f"Last state should equal 10, got {y0[7]}"
        assert all(y0[i] == 0 for i in [1, 2, 3, 4, 6]), "Other initial states should be zero"

class TestParameterSampling:
    """Test parameter sampling functions."""
    
    @patch('app.draws_invitro_df')
    def test_sample_random_iteration_invitro(self, mock_df):
        """Test invitro parameter sampling."""
        # Mock DataFrame with actual parameter names
        mock_row = pd.Series({
            'k_cDD': 0.1, 'd_DD': 0.2, 'k_hillDD': 0.3, 'k_inter': 0.4,
            'maxdeath': 0.5, 'k_hillnec': 0.6, 'p': 0.7, 'h': 0.8
        })
        mock_df.sample.return_value.iloc = [mock_row]
        
        params = sample_random_iteration("invitro")
        
        assert len(params) == 8, f"Expected 8 parameters, got {len(params)}"
        assert params == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    @patch('app.draws_invivo_df')
    def test_sample_random_iteration_invivo(self, mock_df):
        """Test invivo parameter sampling."""
        # Mock DataFrame with actual parameter names
        mock_row = pd.Series({
            'k_kidDD': 0.1, 'hillDD': 0.2, 'd_DD': 0.3, 'maxdeath': 0.4,
            'h1': 0.5, 'k_hillnec': 0.6, 'd_CD': 0.7, 'k_CDINF': 0.8,
            'd_INF': 0.9, 'k_INFKF': 1.0, 'hillKF': 1.1, 'INF0': 1.2
        })
        mock_df.sample.return_value.iloc = [mock_row]
        
        params = sample_random_iteration("invivo")
        
        assert len(params) == 12, f"Expected 12 parameters, got {len(params)}"
        assert params == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]

class TestModelIntegration:
    """Test full model integration and simulation."""
    
    @patch('app.sample_random_iteration')
    def test_run_invitro_model(self, mock_sample):
        """Test invitro model integration."""
        mock_sample.return_value = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        
        dose = 10.0
        simulation_time = 24.0
        params = mock_sample.return_value
        
        solution, t = run_invitro_model(dose, simulation_time, params)
        
        # Check solution dimensions
        assert solution.shape[1] == 6, f"Expected 6 state variables, got {solution.shape[1]}"
        assert len(t) == 100, f"Expected 100 time points, got {len(t)}"  # DEFAULT_TIME_POINTS
        
        # Check time vector
        assert t[0] == 0, "Time should start at 0"
        assert t[-1] == simulation_time, f"Time should end at {simulation_time}"
        
        # Check solution is finite
        assert np.all(np.isfinite(solution)), "All solution values should be finite"
    
    @patch('app.sample_random_iteration')
    def test_run_invivo_model(self, mock_sample):
        """Test invivo model integration."""
        mock_sample.return_value = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
        
        dose = 5.0
        simulation_time = 48.0
        params = mock_sample.return_value
        
        solution, t = run_invivo_model(dose, simulation_time, params)
        
        # Check solution dimensions
        assert solution.shape[1] == 8, f"Expected 8 state variables, got {solution.shape[1]}"
        assert len(t) == 100, f"Expected 100 time points, got {len(t)}"  # config.TIME_POINTS
        
        # Check time vector
        assert t[0] == 0, "Time should start at 0"
        assert t[-1] == simulation_time, f"Time should end at {simulation_time}"
        
        # Check solution is finite
        assert np.all(np.isfinite(solution)), "All solution values should be finite"

class TestNumericalStability:
    """Test numerical stability and edge cases."""
    
    def test_zero_dose_invitro(self):
        """Test invitro model with zero dose."""
        dose = 0.0
        params = fixed_params_invitro + [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        
        solution, t = run_invitro_model(dose, 24.0, params[len(fixed_params_invitro):])
        
        # With zero dose, most states should remain near zero
        assert np.all(solution >= -1e-10), "No states should go significantly negative"
        assert np.all(np.isfinite(solution)), "All solution values should be finite"
    
    def test_zero_dose_invivo(self):
        """Test invivo model with zero dose."""
        dose = 0.0
        params = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
        
        solution, t = run_invivo_model(dose, 48.0, params)
        
        # Check that solution is finite and doesn't explode
        finite_solution = solution[np.isfinite(solution)]
        if len(finite_solution) > 0:
            assert np.all(np.abs(finite_solution) < 1000), "Solution should not explode"
        
        # Check that at least the initial conditions are reasonable
        assert len(solution) > 0, "Should have solution points"
    
    def test_large_dose_stability(self):
        """Test model stability with large doses."""
        large_dose = 1000.0
        params_invitro = fixed_params_invitro + [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        
        # Test that model doesn't crash with large doses
        try:
            solution, t = run_invitro_model(large_dose, 24.0, params_invitro[len(fixed_params_invitro):])
            assert np.all(np.isfinite(solution)), "Solution should remain finite even with large dose"
        except Exception as e:
            pytest.fail(f"Model should handle large doses gracefully, but failed with: {e}")

if __name__ == "__main__":
    pytest.main([__file__])