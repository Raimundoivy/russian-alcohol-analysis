# Start from a specific Python version to ensure consistency
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker's build cache
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your analysis library and the main notebook into the container
COPY ./analysis_lib ./analysis_lib
COPY ./russian_alcohol_consumption_analysis.ipynb .
COPY ./*.csv .

# Jupyter runs on port 8888, so we expose it
EXPOSE 8888

# The command to run when the container starts.
# It starts the Jupyter Notebook server, allowing connections from any IP,
# without a token, and allowing it to run as the root user.
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]