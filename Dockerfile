# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file to the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the container
COPY . .

# Set the environment variable to indicate that the container will run in production mode
ENV FLASK_ENV=production

# Expose port 5000 for the Flask app to run on
EXPOSE 5000

# Start the Flask app
CMD ["flask", "run", "--host=0.0.0.0"]