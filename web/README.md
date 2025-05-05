# FastAPI Web Application

This repository contains a FastAPI web application that serves as a backend for the causal discovery algorithms in the [group-causation library](https://github.com/JoaquinMateosBarroso/group-causation). The application is containerized using Docker and can be easily run using Docker Compose.

## Features

- **Causal Discovery**: 
  - Discover causal relationships within individual time series.
  - Discover causal relationships across groups of time series.
  
- **Benchmarking**: 
  - Benchmark causal discovery algorithms on individual time series.
  - Benchmark causal discovery algorithms on groups of time series.
  
- **Dataset Creation**: 
  - Generate synthetic datasets for individual time series.
  - Generate synthetic datasets for groups of time series.

- **Easy Deployment**: 
  - Built using FastAPI for efficient API management.
  - Docker Compose for easy setup and management of dependencies.

## Prerequisites

Before you begin, ensure you have the following installed:

- Docker: [Install Docker](https://www.docker.com/get-started)
- Docker Compose: [Install Docker Compose](https://docs.docker.com/compose/install/)

## Getting Started

Follow these steps to get the project up and running on your local machine:

### 1. Clone the repository

```bash
git clone https://github.com/JoaquinMateosBarroso/Causal-Inference
cd Causal-Inference/web
```

### 2. Build and run the application with Docker Compose
To start the application, run the following command:

```bash
docker-compose up --build
```
This will:

- Build the Docker containers for the FastAPI app.

- Start the FastAPI app and all required services in the background.

3. Access the application
Once the application is up and running, you can access the FastAPI web interface by navigating to:

```
http://localhost:8000
```

For Swagger UI (API documentation), go to:

```bash
http://localhost:8000/docs
```

4. Stopping the application
To stop the application, run:

```bash
docker-compose down
```

This will stop and remove the containers but leave your data intact.

## Configuration
- You can configure the application by editing the .env file or passing environment variables for:

- Dataset generation parameters (e.g., length of time series, number of series).

- Causal discovery algorithm options (e.g., settings for different methods).

- Benchmarking configuration (e.g., comparison metrics, validation parameters).

## Troubleshooting
- Ensure that Docker is installed and running correctly.

- Make sure the required ports (e.g., 8000) are not being used by other services on your machine.

- If you encounter issues during docker-compose up, check the logs by running docker-compose logs for more information.
