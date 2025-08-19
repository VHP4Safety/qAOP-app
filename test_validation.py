"""
Unit tests for input validation functionality.
"""
import pytest
from validation import (
    validate_model_type, validate_dose, validate_time, validate_simulation_count,
    validate_prediction_request, get_parameter_ranges, ValidationError
)

class TestModelTypeValidation:
    """Test model type validation."""
    
    def test_valid_model_types(self):
        """Test valid model types."""
        assert validate_model_type("invitro") == "invitro"
        assert validate_model_type("invivo") == "invivo"
        assert validate_model_type("INVITRO") == "invitro"  # Case insensitive
        assert validate_model_type(" invivo ") == "invivo"  # Whitespace stripped
    
    def test_invalid_model_types(self):
        """Test invalid model types."""
        with pytest.raises(ValidationError, match="Model type must be 'invitro' or 'invivo'"):
            validate_model_type("invalid")
        
        with pytest.raises(ValidationError, match="Model type must be a string"):
            validate_model_type(123)
        
        with pytest.raises(ValidationError, match="Model type must be 'invitro' or 'invivo'"):
            validate_model_type("")

class TestDoseValidation:
    """Test dose validation."""
    
    def test_valid_invitro_doses(self):
        """Test valid invitro doses."""
        assert validate_dose(10.0, "invitro") == 10.0
        assert validate_dose("50.5", "invitro") == 50.5
        assert validate_dose(0.001, "invitro") == 0.001  # Minimum
        assert validate_dose(1000.0, "invitro") == 1000.0  # Maximum
    
    def test_valid_invivo_doses(self):
        """Test valid invivo doses."""
        assert validate_dose(5.0, "invivo") == 5.0
        assert validate_dose("2.5", "invivo") == 2.5
        assert validate_dose(0.1, "invivo") == 0.1  # Minimum
        assert validate_dose(50.0, "invivo") == 50.0  # Maximum
    
    def test_invalid_dose_types(self):
        """Test invalid dose types."""
        with pytest.raises(ValidationError, match="Dose must be a numeric value"):
            validate_dose("not_a_number", "invitro")
        
        with pytest.raises(ValidationError, match="Dose must be a numeric value"):
            validate_dose(None, "invitro")
    
    def test_dose_range_violations(self):
        """Test dose range violations."""
        # Invitro too low
        with pytest.raises(ValidationError, match="Dose too low"):
            validate_dose(0.0005, "invitro")
        
        # Invitro too high
        with pytest.raises(ValidationError, match="Dose too high"):
            validate_dose(2000.0, "invitro")
        
        # Invivo too low
        with pytest.raises(ValidationError, match="Dose too low"):
            validate_dose(0.05, "invivo")
        
        # Invivo too high
        with pytest.raises(ValidationError, match="Dose too high"):
            validate_dose(100.0, "invivo")

class TestTimeValidation:
    """Test time validation."""
    
    def test_valid_invitro_times(self):
        """Test valid invitro times."""
        assert validate_time(24.0, "invitro") == 24.0
        assert validate_time("72.5", "invitro") == 72.5
        assert validate_time(0.1, "invitro") == 0.1  # Minimum
        assert validate_time(168.0, "invitro") == 168.0  # Maximum
    
    def test_valid_invivo_times(self):
        """Test valid invivo times."""
        assert validate_time(48.0, "invivo") == 48.0
        assert validate_time("100.0", "invivo") == 100.0
        assert validate_time(1.0, "invivo") == 1.0  # Minimum
        assert validate_time(2160.0, "invivo") == 2160.0  # Maximum
    
    def test_invalid_time_types(self):
        """Test invalid time types."""
        with pytest.raises(ValidationError, match="Time must be a numeric value"):
            validate_time("not_a_number", "invitro")
        
        with pytest.raises(ValidationError, match="Time must be a numeric value"):
            validate_time([], "invitro")
    
    def test_time_range_violations(self):
        """Test time range violations."""
        # Zero or negative time
        with pytest.raises(ValidationError, match="Time must be positive"):
            validate_time(0, "invitro")
        
        with pytest.raises(ValidationError, match="Time must be positive"):
            validate_time(-5, "invitro")
        
        # Invitro too short
        with pytest.raises(ValidationError, match="Time too short"):
            validate_time(0.05, "invitro")
        
        # Invitro too long
        with pytest.raises(ValidationError, match="Time too long"):
            validate_time(200.0, "invitro")
        
        # Invivo too short
        with pytest.raises(ValidationError, match="Time too short"):
            validate_time(0.5, "invivo")
        
        # Invivo too long
        with pytest.raises(ValidationError, match="Time too long"):
            validate_time(3000.0, "invivo")

class TestSimulationCountValidation:
    """Test simulation count validation."""
    
    def test_valid_simulation_counts(self):
        """Test valid simulation counts."""
        assert validate_simulation_count(10) == 10
        assert validate_simulation_count("250") == 250
        assert validate_simulation_count(1) == 1  # Minimum
        assert validate_simulation_count(1000) == 1000  # Maximum
    
    def test_invalid_simulation_count_types(self):
        """Test invalid simulation count types."""
        with pytest.raises(ValidationError, match="Simulation count must be an integer"):
            validate_simulation_count("not_a_number")
        
        # Note: Python int() can convert floats, so 10.5 becomes 10
        # Test with something that can't be converted to int
        with pytest.raises(ValidationError, match="Simulation count must be an integer"):
            validate_simulation_count(None)
    
    def test_simulation_count_range_violations(self):
        """Test simulation count range violations."""
        with pytest.raises(ValidationError, match="Simulation count too low"):
            validate_simulation_count(0)
        
        with pytest.raises(ValidationError, match="Simulation count too high"):
            validate_simulation_count(2000)

class TestPredictionRequestValidation:
    """Test complete prediction request validation."""
    
    def test_valid_invitro_request(self):
        """Test valid invitro request."""
        request_data = {
            "model_type": "invitro",
            "dose": 10.0,
            "time": 24.0,
            "simulation_count": 100
        }
        
        validated = validate_prediction_request(request_data)
        
        assert validated["model_type"] == "invitro"
        assert validated["dose"] == 10.0
        assert validated["time"] == 24.0
        assert validated["simulation_count"] == 100
    
    def test_valid_invivo_request(self):
        """Test valid invivo request."""
        request_data = {
            "model_type": "invivo",
            "dose": 5.0,
            "time": 48.0
        }
        
        validated = validate_prediction_request(request_data)
        
        assert validated["model_type"] == "invivo"
        assert validated["dose"] == 5.0
        assert validated["time"] == 48.0
        assert validated["simulation_count"] == 250  # Default
    
    def test_missing_required_fields(self):
        """Test missing required fields."""
        request_data = {
            "model_type": "invitro",
            "dose": 10.0
            # Missing "time"
        }
        
        with pytest.raises(ValidationError, match="Missing required fields: time"):
            validate_prediction_request(request_data)
    
    def test_invalid_request_type(self):
        """Test invalid request data type."""
        with pytest.raises(ValidationError, match="Request data must be a JSON object"):
            validate_prediction_request("not_a_dict")

class TestParameterRanges:
    """Test parameter range retrieval."""
    
    def test_get_all_ranges(self):
        """Test getting all parameter ranges."""
        ranges = get_parameter_ranges()
        
        assert "models" in ranges
        assert "simulation" in ranges
        assert "invitro" in ranges["models"]
        assert "invivo" in ranges["models"]
    
    def test_get_invitro_ranges(self):
        """Test getting invitro parameter ranges."""
        ranges = get_parameter_ranges("invitro")
        
        assert ranges["model"] == "invitro"
        assert "parameters" in ranges
        assert "dose" in ranges["parameters"]
        assert "time" in ranges["parameters"]
    
    def test_get_invivo_ranges(self):
        """Test getting invivo parameter ranges."""
        ranges = get_parameter_ranges("invivo")
        
        assert ranges["model"] == "invivo"
        assert "parameters" in ranges
        assert "dose" in ranges["parameters"]
        assert "time" in ranges["parameters"]
    
    def test_invalid_model_type_for_ranges(self):
        """Test invalid model type for ranges."""
        with pytest.raises(ValidationError):
            get_parameter_ranges("invalid_model")

if __name__ == "__main__":
    pytest.main([__file__])