# qAOP-app

This tool runs a quantitative Adverse Outcome Pathway (qAOP) simulation using an ODE-based model. It provides predictions of **DNA Damage (EGs)** and **Necrosis (% cell death)** for a given dose and time point.

---

## 🚀 How to Run the App

### Clone this repository

```bash
git clone https://github.com/VHP4Safety/qAOP-predictor.git
cd qAOP-predictor
```

### Run with Python
We recommend using a virtual environment or `conda`.

Using pip:
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Then, run the Flask app
```bash
python app.py
```
The app will be available at:
http://localhost:5000

---

### Run with Docker

If you prefer not to install Python locally, you can run the app via Docker.

Build the Docker image.

```bash
docker build -t qaop-predictor .
```

Then, run the container

```bash
docker run -p 5000:5000 qaop-predictor
```
The app will be available at:
http://localhost:5000