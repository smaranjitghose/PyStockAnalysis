# Get the base image
FROM python:3.8-slim-buster

#Expose port 8080
# EXPOSE 8080

#Copy Requirements.txt file into app directory
COPY requirements.txt app/requirements.txt

#install all requirements in requirements.txt
RUN pip install -r app/requirements.txt

#Copy all files in current directory into app directory
COPY . /app

#Change Working Directory to app directory
WORKDIR /app

# Run the application on port 8080 for Local Host or GCP
# ENTRYPOINT ["streamlit", "run", "app.py", "--server.port 8080", "--server.address 0.0.0.0"]

# Heroku Startup Command
CMD streamlit run app.py --server.port $PORT