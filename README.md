# qAOP Predictor Application

A quantitative Adverse Outcome Pathway (qAOP) simulation tool that provides predictions of cisplatin-induced nephrotoxicity using ODE-based mathematical models.

## Features

- **In Vitro Model**: 6-compartment RPTEC/TERT1 cell culture model
- **In Vivo Model**: 8-compartment rat kidney model  
- **Web Interface**: User-friendly form-based interface
- **REST API**: JSON-based API for programmatic access
- **Input Validation**: Scientific parameter range validation
- **Error Handling**: Comprehensive logging and error management
- **Configuration**: Environment-based configuration management
- **Testing**: Unit tests for mathematical model validation

## Model Outputs

### In Vitro Model
- **DNA Damage**: Relative DNA damage levels
- **Necrosis**: Percentage cell death

### In Vivo Model
- **DNA Damage**: Relative DNA damage levels  
- **Cell Death**: Percentage cell death
- **Inflammation**: Inflammatory response
- **Kidney Failure**: Kidney function impairment

## Quick Start

### Option 1: Python Virtual Environment

```bash
# Clone repository
git clone https://github.com/VHP4Safety/qAOP-predictor.git
cd qAOP-predictor

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run application
python app.py
```

The app will be available at: http://localhost:5000

### Option 2: Docker

```bash
# Build Docker image
docker build -t qaop-predictor .

# Run container
docker run -p 5000:5000 qaop-predictor
```

The app will be available at: http://localhost:5000

## Configuration

The application supports environment-based configuration:

### Environment Variables

```bash
# Flask Configuration
FLASK_ENV=development          # development, production, testing
FLASK_DEBUG=true              # true/false
FLASK_HOST=0.0.0.0            # Host binding
FLASK_PORT=5000               # Port number

# Simulation Parameters
SIMULATION_COUNT=250          # Number of Monte Carlo simulations
TIME_POINTS=100               # Time resolution for ODE solver
```

### Using .env Files

Create a `.env` file in the project root:

```bash
cp .env.example .env
# Edit .env with your preferred settings
```

## API Documentation

The application provides a REST API for programmatic access:

### Base URL
```
http://localhost:5000/api
```

### Endpoints

#### Health Check
```http
GET /api/health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "models": ["invitro", "invivo"],
  "checks": {
    "data": {
      "invitro_params": true,
      "invivo_params": true,
      "pk_params": true
    },
    "models": {
      "invitro_sampling": true,
      "invivo_sampling": true
    }
  }
}
```

#### Model Information
```http
GET /api/models
```

**Response:**
```json
{
  "models": {
    "invitro": {
      "description": "6-compartment model for RPTEC/TERT1 cell culture",
      "dose_unit": "μM",
      "typical_dose_range": "1-100 μM",
      "time_unit": "hours",
      "typical_time_range": "1-72 hours",
      "endpoints": ["dna_damage", "necrosis"]
    },
    "invivo": {
      "description": "8-compartment rat kidney model",
      "dose_unit": "mg/kg",
      "typical_dose_range": "1-20 mg/kg",
      "time_unit": "hours",
      "typical_time_range": "1-700 hours",
      "endpoints": ["dna_damage", "necrosis", "inflammation", "kidney_failure"]
    }
  }
}
```

#### Parameter Ranges
```http
GET /api/ranges
GET /api/ranges?model_type=invitro
```

**Response:**
```json
{
  "models": {
    "invitro": {
      "dose": {
        "min": 0.001,
        "max": 1000.0,
        "unit": "μM",
        "description": "Cisplatin concentration in cell culture medium"
      },
      "time": {
        "min": 0.1,
        "max": 168.0,
        "unit": "hours",
        "description": "Exposure duration in cell culture"
      }
    }
  },
  "simulation": {
    "simulation_count": {
      "min": 1,
      "max": 1000,
      "default": 250
    }
  }
}
```

#### Model Prediction
```http
POST /api/predict
Content-Type: application/json
```

**Request Body:**
```json
{
  "model_type": "invitro",
  "dose": 10.0,
  "time": 24.0,
  "simulation_count": 100
}
```

**Response:**
```json
{
  "model_type": "invitro",
  "dose": 10.0,
  "dose_unit": "μM",
  "time": 24.0,
  "time_unit": "hours",
  "simulation_count": 100,
  "endpoints": {
    "dna_damage": {
      "final_value": {
        "mean": 2.89,
        "std": 0.11,
        "median": 2.97,
        "min": 2.74,
        "max": 2.99,
        "percentile_25": 2.76,
        "percentile_75": 2.98
      },
      "time_series": {
        "time_points": [0, 0.24, 0.48, ...],
        "mean": [0, 0.002, 0.011, ...],
        "std": [0, 0.0004, 0.002, ...]
      }
    },
    "necrosis": {
      "final_value": { /* similar structure */ },
      "time_series": { /* similar structure */ }
    }
  }
}
```

### API Usage Examples

#### Python
```python
import requests

# Make prediction request
response = requests.post('http://localhost:5000/api/predict', json={
    "model_type": "invitro",
    "dose": 10.0,
    "time": 24.0,
    "simulation_count": 100
})

data = response.json()
dna_damage_mean = data['endpoints']['dna_damage']['final_value']['mean']
print(f"DNA Damage (mean): {dna_damage_mean:.4f}")
```

#### cURL
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "invivo",
    "dose": 5.0,
    "time": 48.0
  }'
```

## Testing

Run the test suite to validate model functionality:

```bash
# Run all tests
python -m pytest

# Run specific test files
python -m pytest test_models.py -v
python -m pytest test_validation.py -v

# Test API endpoints (requires running server)
python test_api.py
```

## Project Structure

```
qAOP-app/
├── app.py                 # Main Flask application
├── config.py             # Configuration management
├── validation.py         # Input validation logic
├── logger.py             # Structured logging
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker configuration
├── pytest.ini           # Testing configuration
├── .env.example         # Environment variables template
├── data/                # Model parameter files
│   ├── draws_cisDDNEC2024-08-05_11-21-18.csv
│   ├── draws_cisvivo2024-08-07_10-34-00.csv
│   └── fit_pk_summary.csv
├── templates/           # HTML templates
│   └── index.html
├── static/             # Static assets
│   ├── styles.css
│   └── img/
└── tests/              # Test files
    ├── test_models.py
    ├── test_validation.py
    └── test_api.py
```

## Parameter Validation

The application validates input parameters against scientifically reasonable ranges:

### In Vitro Model
- **Dose**: 0.001 - 1000 μM (cisplatin concentration)
- **Time**: 0.1 - 168 hours (up to 7 days)

### In Vivo Model
- **Dose**: 0.1 - 50 mg/kg (therapeutic to maximum tolerable)
- **Time**: 1 - 2160 hours (up to 90 days observation)

### Simulation Parameters
- **Simulation Count**: 1 - 1000 Monte Carlo iterations

## Model Details

### Mathematical Framework
- **ODE Integration**: scipy.integrate.odeint
- **Parameter Uncertainty**: Bayesian posterior distributions
- **Monte Carlo Simulation**: 250 iterations (default)
- **Statistical Analysis**: Mean, std, percentiles, time series

### Model Calibration
- **In Vitro**: Calibrated against RPTEC/TERT1 cell culture data
- **In Vivo**: Calibrated against rat nephrotoxicity studies  
- **Parameters**: Pre-computed Bayesian parameter posteriors

## Deployment

### Production Configuration
```bash
export FLASK_ENV=production
export FLASK_DEBUG=false
export FLASK_HOST=0.0.0.0
export FLASK_PORT=8080
export SIMULATION_COUNT=250
```

### Docker Production
```bash
docker run -p 8080:8080 -e FLASK_PORT=8080 -e FLASK_ENV=production qaop-predictor
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`python -m pytest`)
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## License

This project is part of the VHP4Safety consortium. Please refer to the project license for usage terms.

## Acknowledgments

- VHP4Safety consortium for project funding and support
- qAOP modeling framework development team
- RPTEC/TERT1 cell culture data providers
- Rat nephrotoxicity study contributors