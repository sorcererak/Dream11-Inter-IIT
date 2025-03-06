#!/bin/bash

# Set the PostgreSQL password environment variable
export PGPASSWORD="Ak@123"

# Function to log messages with timestamps
log() {
    echo "$(date +'%Y-%m-%d %H:%M:%S') [ENTRYPOINT] $1"
}

# Wait for the PostgreSQL container to be ready
log "Waiting for the PostgreSQL container to be ready..."
until psql -h db -U postgres -d dream11 -c '\q' 2>/dev/null; do
  log "Database not ready yet. Retrying in 2 seconds..."
  sleep 2
done
log "Database is ready."

# Check if the database is empty
log "Checking if the database contains tables..."
result=$(psql -h db -U postgres -d dream11 -t -c "SELECT COUNT(*) FROM pg_catalog.pg_tables WHERE schemaname != 'pg_catalog' AND schemaname != 'information_schema';" | xargs)

if [[ "$result" -eq 0 ]]; then
    log "Database is empty. Importing SQL dump files..."
    # Run the SQL dump files from the /app/sql folder
    for file in /app/sql/*.sql; do
        if [[ -f "$file" ]]; then
            log "Executing $file..."
            psql -h db -U postgres -d dream11 -f "$file"
        else
            log "No SQL files found in /app/sql. Skipping import."
            break
        fi
    done
else
    log "Database already contains data. Skipping SQL dump."
fi

# Start the FastAPI application
log "Starting the FastAPI application..."
exec uvicorn main:app --host 0.0.0.0 --port 80
