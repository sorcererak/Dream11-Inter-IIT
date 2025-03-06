# Dream11 Team Prediction App - Backend

This is the backend service for the Dream11 Team Prediction App. Built with **FastAPI**, it provides an API for managing and predicting Dream11 fantasy sports teams. This backend is designed to handle user management, team generation, player statistics, and prediction algorithms.

## Features

- **User Management**: Authentication and user profile management.
- **Player Database**: Store and retrieve player statistics.
- **Team Prediction**: Generate recommended teams based on player statistics and user preferences.
- **API Endpoints**: RESTful API for easy integration with the frontend.
- **Dockerized**: Easily deployable using Docker.

## Tech Stack

- **Backend Framework**: FastAPI
- **Database**: PostgreSQL
- **Deployment**: Docker, Docker Compose
- **Testing**: Pytest
- **Others**: Pydantic for data validation, SQLAlchemy for ORM

## Getting Started

### Prerequisites

- Python 3.8+
- Docker and Docker Compose (optional, for containerized setup)
- PostgreSQL (for production)

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/dream11-prediction-backend.git
   cd dream11-prediction-backend
   uvicorn main:app --reload

