#goal: i wanna make an image that will make my project run, this docker file is recipe for that image

#pul a base image which gives all required tool and libraries
FROM python:3.11-slim

#create a folder where the app code will be stored
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY src/app/requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/app /app
COPY src/vector_twin /app/vector_twin

EXPOSE 8501

#run the application #cmd can be overwritten
CMD ["streamlit", "run", "main.py"]

#ENTRYPOINT: bada bhai of CMD which cannot be overwritten