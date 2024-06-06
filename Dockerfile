FROM python:3.9.10

# Install system-level packages required by your Python libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    libgl1-mesa-dev

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

ENV PYTHONUNBUFFERED True

ENV APP_HOME /back-end
WORKDIR $APP_HOME
COPY . ./

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Use an environment variable for the port number
ENV PORT 8080

# Expose the port
EXPOSE $PORT

# Use the environment variable in the gunicorn command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]