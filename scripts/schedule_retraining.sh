#!/bin/bash

# Automated model retraining scheduler
# This script can be run via cron job for periodic model updates

# Configuration
PROJECT_DIR="/home/alerman/projects/bena_projects/flood_pred"
WEATHER_DATA="$PROJECT_DIR/data/sample_weather.csv"
FLOOD_DATA="$PROJECT_DIR/data/sample_flood.csv"
TERRAIN_DATA="$PROJECT_DIR/data/sample_terrain.tif"
FLOOD_MASK="$PROJECT_DIR/data/sample_flood_mask.tif"
LOG_FILE="$PROJECT_DIR/logs/retrain.log"

# Create log directory
mkdir -p "$PROJECT_DIR/logs"

# Activate virtual environment
source "$PROJECT_DIR/venv/bin/activate"

# Change to project directory
cd "$PROJECT_DIR"

echo "$(date): Starting automated model retraining..." >> "$LOG_FILE"

# Check if data files exist
if [ -f "$WEATHER_DATA" ] && [ -f "$FLOOD_DATA" ]; then
    echo "$(date): Running GBM retraining..." >> "$LOG_FILE"
    python -c "
from scripts.retrain_models import retrain_gbm
try:
    retrain_gbm('$WEATHER_DATA', '$FLOOD_DATA')
    print('GBM retraining successful')
except Exception as e:
    print(f'GBM retraining failed: {e}')
    " >> "$LOG_FILE" 2>&1
else
    echo "$(date): Weather or flood data not found, skipping GBM retraining" >> "$LOG_FILE"
fi

if [ -f "$TERRAIN_DATA" ] && [ -f "$FLOOD_MASK" ]; then
    echo "$(date): Running U-Net retraining..." >> "$LOG_FILE"
    python -c "
from scripts.retrain_models import retrain_unet
try:
    retrain_unet('$TERRAIN_DATA', '$FLOOD_MASK')
    print('U-Net retraining successful')
except Exception as e:
    print(f'U-Net retraining failed: {e}')
    " >> "$LOG_FILE" 2>&1
else
    echo "$(date): Terrain or flood mask data not found, skipping U-Net retraining" >> "$LOG_FILE"
fi

echo "$(date): Automated retraining completed." >> "$LOG_FILE"