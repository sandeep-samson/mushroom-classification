# Use an official Python 3.x image as the base
FROM python:3.12-slim

# Set the working directory to /app
WORKDIR /app

# Create a virtual environment
RUN python -m venv venv
ENV VIRTUAL_ENV=/app/venv
ENV PATH=$VIRTUAL_ENV/bin:$PATH

# Activate the virtual environment (using Bash)
SHELL ["/bin/bash", "-c"]
RUN source venv/bin/activate && pip install --upgrade pip

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies in the virtual environment
RUN pip install -r requirements.txt

# Copy the application code into the container
COPY . .

# Copy the dataset into the container (assuming it's stored in a directory named "data")
COPY data /app/data/

# Expose the port that your Streamlit app will use (default is 8501)
EXPOSE 8501

# Set environment variables for streamlit
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_RUNTIME_PREFIX=/app

# Run the command to start the Streamlit app when the container launches
CMD ["streamlit", "run", "app.py"]
