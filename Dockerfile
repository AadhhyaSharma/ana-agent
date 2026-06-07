FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy ANA source files
COPY mini_100_neuron_agent.py .
COPY core_infra.py .
COPY neural_engine.py .
COPY ana_launcher.py .

# Expose web UI port
EXPOSE 8080

# Run ANA
CMD ["python", "ana_launcher.py", "--port", "8080", "--no-browser"]
