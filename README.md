# Project Title

Doubleit Model

## Description

This is a FastAPI-based project designed to illustrate a robust structure for Continuous Integration/Continuous Deployment (CI/CD) pipelines. 

It hosts a simple machine learning model that doubles numerical values. 

# Getting Started

This section will walk you through the necessary steps to get DoubleitModel up and running 
on your local machine for development and testing purposes. 

Here's a breakdown of the steps you'll be following:

## Prerequisites

Ensure you have the following installed on your local machine:

- [Python](https://www.python.org/downloads/) (3.7 or higher)
- [Docker](https://www.docker.com/products/docker-desktop)
- [Docker Compose](https://docs.docker.com/compose/install/)

## Cloning the Repository

Clone the repository to your local machine:

```bash
git clone https://github.com/gerardgonzalez-ai/doubleit_model
```
## Setting Up the Environment

It's advisable to create a virtual environment to manage dependencies for the project:

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

install the project dependencies:

```bash
pip install -r requirements.txt
```
## Building the Docker Image

```bash
docker build -t DoubleitModel .
```

## Running the Application

```bash
docker-compose -f compose.yaml up
```

Once the application is running, navigate to http://localhost:8000 in your web browser to access the FastAPI autogenerated interactive API documentation.

## Making Predictions

Use the interactive documentation or a tool like curl to send a POST request to the /predict endpoint with a list of numbers. Here's an example using curl:

```bash
curl -X POST "http://localhost:8000/predict/" -H "accept: application/json" -H "Content-Type: application/json" -d '{"numbers":[1,2,3,4]}'
```
