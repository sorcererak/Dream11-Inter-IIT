import logging
import colorlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.db.session import setup_db
from app.api.routes import main as routes
from contextlib import asynccontextmanager

# Create a colorized formatter
formatter = colorlog.ColoredFormatter(
    "%(log_color)s%(levelname)s - %(message)s",
    log_colors={
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold_red",
    },
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Set up handler and root logger
handler = logging.StreamHandler()
handler.setFormatter(formatter)

logging.basicConfig(
    level=logging.INFO,  # Set the default logging level for the root logger
    handlers=[handler]   # Add the colorized handler to the root logger
)

# Reduce verbosity for specific libraries
logging.getLogger("sqlalchemy.engine").setLevel(logging.INFO)  # Log SQL queries
logging.getLogger("uvicorn").setLevel(logging.WARNING)  # Reduce verbosity for Uvicorn

# Create a custom logger for your application
logger = logging.getLogger("app_logger")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup event
    logger.info("Starting application...")
    setup_db()  # Ensure DB setup is logged
    logger.info("Database setup complete!")
    yield
    # Shutdown event
    logger.info("Shutting down application...")

# Create the FastAPI app instance
app = FastAPI(
    title="APP_NAME",
    lifespan=lifespan
)

# Add middleware for handling CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
    allow_credentials=True
)

# Include the routes for the app
app.include_router(routes.api_router)
