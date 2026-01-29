"""
Input validation for qAOP model parameters.
Defines scientifically reasonable ranges for dose and time parameters.
"""
from typing import Dict, Tuple, Any
import logging

logger = logging.getLogger("qAOP.validation")

# Parameter ranges based on experimental data and scientific literature
PARAMETER_RANGES = {
    "invitro": {
        "dose": {
            "min": 0.001,      # μM - minimum detectable concentration
            "max": 150.0,      # μM - upper limit for cell viability
            "unit": "μM",
            "description": "Cisplatin concentration in cell culture medium"
        },
        "time": {
            "min": 0.1,        # hours - minimum meaningful exposure time
            "max": 168.0,      # hours - 7 days maximum for cell culture
            "unit": "hours",
            "description": "Exposure duration in cell culture"
        }
    },
    "invivo": {
        "dose": {
            "min": 0.1,        # mg/kg - minimum therapeutic dose
            "max": 10.0,       # mg/kg - maximum tolerable dose
            "unit": "mg/kg",
            "description": "Cisplatin dose per body weight"
        },
        "time": {
            "min": 1.0,        # hours - minimum observation time
            "max": 2160.0,     # hours - 90 days maximum observation
            "unit": "hours",
            "description": "Observation time after dosing"
        }
    }
}

SIMULATION_LIMITS = {
    "simulation_count": {
        "min": 1,
        "max": 1000,
        "default": 250,
        "description": "Number of Monte Carlo simulations"
    }
}

class ValidationError(ValueError):
    """Custom exception for validation errors."""
    pass

def validate_model_type(model_type: str) -> str:
    """
    Validate model type parameter.
    
    Args:
        model_type: Model type to validate
        
    Returns:
        Validated model type
        
    Raises:
        ValidationError: If model type is invalid
    """
    if not isinstance(model_type, str):
        raise ValidationError(f"Model type must be a string, got {type(model_type).__name__}")
    
    model_type = model_type.lower().strip()
    
    if model_type not in ["invitro", "invivo"]:
        raise ValidationError(f"Model type must be 'invitro' or 'invivo', got '{model_type}'")
    
    return model_type

def validate_dose(dose: Any, model_type: str) -> float:
    """
    Validate dose parameter against model-specific ranges.
    
    Args:
        dose: Dose value to validate
        model_type: Model type (invitro or invivo)
        
    Returns:
        Validated dose as float
        
    Raises:
        ValidationError: If dose is invalid
    """
    # Type validation
    try:
        dose_float = float(dose)
    except (TypeError, ValueError) as e:
        raise ValidationError(f"Dose must be a numeric value, got {type(dose).__name__}") from e
    
    # Range validation
    dose_range = PARAMETER_RANGES[model_type]["dose"]
    
    if dose_float < dose_range["min"]:
        raise ValidationError(
            f"Dose too low: {dose_float} {dose_range['unit']} "
            f"(minimum: {dose_range['min']} {dose_range['unit']})"
        )
    
    if dose_float > dose_range["max"]:
        raise ValidationError(
            f"Dose too high: {dose_float} {dose_range['unit']} "
            f"(maximum: {dose_range['max']} {dose_range['unit']})"
        )
    
    # Special validation rules
    if dose_float == 0.0:
        logger.warning(f"Zero dose specified for {model_type} model - results may not be meaningful")
    
    return dose_float

def validate_time(time: Any, model_type: str) -> float:
    """
    Validate time parameter against model-specific ranges.
    
    Args:
        time: Time value to validate
        model_type: Model type (invitro or invivo)
        
    Returns:
        Validated time as float
        
    Raises:
        ValidationError: If time is invalid
    """
    # Type validation
    try:
        time_float = float(time)
    except (TypeError, ValueError) as e:
        raise ValidationError(f"Time must be a numeric value, got {type(time).__name__}") from e
    
    # Range validation
    time_range = PARAMETER_RANGES[model_type]["time"]
    
    if time_float <= 0:
        raise ValidationError(f"Time must be positive, got {time_float}")
    
    if time_float < time_range["min"]:
        raise ValidationError(
            f"Time too short: {time_float} {time_range['unit']} "
            f"(minimum: {time_range['min']} {time_range['unit']})"
        )
    
    if time_float > time_range["max"]:
        raise ValidationError(
            f"Time too long: {time_float} {time_range['unit']} "
            f"(maximum: {time_range['max']} {time_range['unit']})"
        )
    
    return time_float

def validate_simulation_count(simulation_count: Any) -> int:
    """
    Validate simulation count parameter.
    
    Args:
        simulation_count: Simulation count to validate
        
    Returns:
        Validated simulation count as int
        
    Raises:
        ValidationError: If simulation count is invalid
    """
    # Type validation
    try:
        count_int = int(simulation_count)
    except (TypeError, ValueError) as e:
        raise ValidationError(f"Simulation count must be an integer, got {type(simulation_count).__name__}") from e
    
    # Range validation
    limits = SIMULATION_LIMITS["simulation_count"]
    
    if count_int < limits["min"]:
        raise ValidationError(f"Simulation count too low: {count_int} (minimum: {limits['min']})")
    
    if count_int > limits["max"]:
        raise ValidationError(f"Simulation count too high: {count_int} (maximum: {limits['max']})")
    
    return count_int

def validate_prediction_request(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate complete prediction request.
    
    Args:
        data: Request data dictionary
        
    Returns:
        Validated and normalized request data
        
    Raises:
        ValidationError: If any parameter is invalid
    """
    if not isinstance(data, dict):
        raise ValidationError("Request data must be a JSON object")
    
    # Required fields
    required_fields = ["model_type", "dose", "time"]
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        raise ValidationError(f"Missing required fields: {', '.join(missing_fields)}")
    
    # Validate each parameter
    validated_data = {}
    
    validated_data["model_type"] = validate_model_type(data["model_type"])
    validated_data["dose"] = validate_dose(data["dose"], validated_data["model_type"])
    validated_data["time"] = validate_time(data["time"], validated_data["model_type"])
    
    # Optional simulation count
    if "simulation_count" in data:
        validated_data["simulation_count"] = validate_simulation_count(data["simulation_count"])
    else:
        validated_data["simulation_count"] = SIMULATION_LIMITS["simulation_count"]["default"]
    
    return validated_data

def get_parameter_ranges(model_type: str = None) -> Dict[str, Any]:
    """
    Get parameter ranges for documentation or validation.
    
    Args:
        model_type: Optional model type to get ranges for
        
    Returns:
        Parameter ranges dictionary
    """
    if model_type:
        validate_model_type(model_type)
        return {
            "model": model_type,
            "parameters": PARAMETER_RANGES[model_type],
            "simulation": SIMULATION_LIMITS
        }
    else:
        return {
            "models": PARAMETER_RANGES,
            "simulation": SIMULATION_LIMITS
        }