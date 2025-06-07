#!/usr/bin/env bash

set -eEuo pipefail

handle_failure() {
    echo "An error occurred. Logs from /app/logs/responsible_ai.log:"
    tail -n 50 "/app/logs/responsible_ai.log" || true
    echo "Keeping the container alive for debugging..."
    tail -f /dev/null
}

trap 'handle_failure' ERR INT TERM

log() {
    echo "[$(date +'%F %T')] $*"
}

# Start Gunicorn server
log "Starting Gunicorn server..."
gunicorn --config gunicorn_conf.py app.wsgi:application &
GUNICORN_PID=$!

# Start Streamlit dashboard if enabled
DASHBOARD_ENABLED=$(grep -A10 'DASHBOARD:' config/templates/app_config.yaml | grep 'ENABLED:' | head -n1 | cut -d ':' -f 2 | tr -d ' \t')
if [[ "$DASHBOARD_ENABLED" == "true" ]]; then
    # Get dashboard configuration from app_config
    DASHBOARD_HOST=$(grep -A10 'DASHBOARD:' config/templates/app_config.yaml | grep 'HOST:' | head -n1 | cut -d ':' -f 2 | tr -d ' \t"')
    DASHBOARD_PORT=$(grep -A10 'DASHBOARD:' config/templates/app_config.yaml | grep 'PORT:' | head -n1 | cut -d ':' -f 2 | tr -d ' \t"')
    
    log "Starting Streamlit dashboard on ${DASHBOARD_HOST}:${DASHBOARD_PORT}..."
    streamlit run dashboard/dashboard_app.py \
        --server.address="${DASHBOARD_HOST}" \
        --server.port="${DASHBOARD_PORT}" \
        --server.enableCORS=false \
        --server.enableXsrfProtection=false &
    STREAMLIT_PID=$!
    
    log "Monitoring processes: Gunicorn(${GUNICORN_PID}), Streamlit(${STREAMLIT_PID})"
    
    # Wait for any process to exit
    wait -n "$GUNICORN_PID" "$STREAMLIT_PID"
else
    log "Dashboard is disabled. Only starting API server."
    log "Monitoring processes: Gunicorn(${GUNICORN_PID})"
    
    # Wait for gunicorn to exit
    wait -n "$GUNICORN_PID"
fi

EXIT_CODE=$?
log "A service exited with code ${EXIT_CODE}"

# Determine which process exited
if ! ps -p "$GUNICORN_PID" > /dev/null 2>&1; then
    log "Gunicorn (PID ${GUNICORN_PID}) was the process that exited."
    EXITED_SERVICE="Gunicorn"
elif [[ "$DASHBOARD_ENABLED" == "true" ]] && ! ps -p "$STREAMLIT_PID" > /dev/null 2>&1; then
    log "Streamlit (PID ${STREAMLIT_PID}) was the process that exited."
    EXITED_SERVICE="Streamlit"
else
    log "Could not determine which process exited."
    EXITED_SERVICE="Unknown"
fi

log "Service ${EXITED_SERVICE} exited with code ${EXIT_CODE}"

# Only handle failure for non-zero exit codes
if [ "$EXIT_CODE" -ne 0 ]; then
    handle_failure
fi

# If we reach here, all services are running fine
log "All services are running fine."

# Wait indefinitely
wait

# This point should never be reached, but if it is, log the exit
log "Exited unexpectedly"
handle_failure
