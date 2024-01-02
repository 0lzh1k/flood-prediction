# Flood Prediction Project

This project uses AI to predict flood events based on weather forecasts and terrain data. It includes:
- Data ingestion and preprocessing
- Machine learning models (GBM, U-Net)
- Streamlit UI for simulation and visualization
- Map visualization using Folium/Kepler.gl
 - Live weather integration with toggle vs simulation
 - Configurable spatial flood risk point generation & legend
 - Automated retraining scripts & scheduling examples

## Folder Structure
- app.py: Streamlit application entry point
- data/: Raw and generated sample data (CSV/GeoTIFF)
- models/: Saved models/artifacts (gbm_model.pkl, unet_model.pth) and model code
- scripts/: Data ingestion, feature engineering, retraining, scheduling, sample generation
- logs/: Retraining logs
- requirements.txt: Pinned dependencies for reproducibility

## Installation
### 1. Create & Activate a Virtual Environment
Recommended to isolate dependencies.

Linux / macOS:
```bash
python -m venv venv
source venv/bin/activate
```

Windows (PowerShell):
```powershell
python -m venv venv
./venv/Scripts/Activate.ps1
```

Confirm activation (shell prompt prefixed by (venv)) and correct Python:
```bash
python -V
which python  # (where python on Windows)
```

Upgrade pip (optional but recommended):
```bash
python -m pip install --upgrade pip
```

### 2. Install dependencies
   ```bash
   pip install streamlit pandas numpy scikit-learn rasterio folium streamlit-folium torch torchvision keplergl streamlit-keplergl joblib python-dotenv
   ```

### 3. (Optional) Set up environment variables for API keys in a `.env` file
   ```bash
   echo "OPENWEATHER_API_KEY=your_key_here" > .env
   ```
   You can also paste the key directly into the app sidebar.

## Data Sources
For real data, you can obtain:

- **Weather Data**: 
  - Kazakhstan Meteorological Service
  - OpenWeatherMap API (https://openweathermap.org/api)
  - Global weather APIs

- **Terrain Data**:
  - Kazakhstan State Land Cadastre
  - SRTM DEM data (https://www.usgs.gov/centers/eros/science/usgs-eros-archive-digital-elevation-shuttle-radar-topography-mission-srtm-1-arc?qt-science_center_objects=0#qt-science_center_objects)
  - Open Elevation API

- **Historical Flood Data**:
  - Kazakhstan Emergency Situations Ministry
  - Local government flood records
  - Global Flood Database (https://www.globalfloods.eu/)

The app includes a "Use Sample Data" option to generate synthetic data for testing (centered on Astana, Kazakhstan).

## Usage
1. Prepare data (optional - app can generate sample data):
   - Weather data CSV with columns: time, rainfall, temperature
   - Terrain data as GeoTIFF
   - Historical flood data CSV with flood occurrences

2. Train models (optional, or use pre-trained):
   ```bash
   python scripts/retrain_models.py
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

4. (Important) After generating sample data for the first time, train models to create local artifacts:
   ```bash
   python scripts/retrain_models.py
   ```
   This produces/updates files under `models/` (e.g., `gbm_model.pkl`). If you skip this, the app will fall back to an in-memory dummy model each run.

5. In the app:
   - Check "Use Sample Data" and click "Generate Sample Data" for instant demo data (Astana region)
   - Or upload your own weather CSV, terrain GeoTIFF, flood history CSV, and optional flood mask GeoTIFF
   - Use sliders to simulate rainfall & temperature
   - (Optional) Enter your OpenWeatherMap API key and click "Fetch Live Weather" for current conditions
   - Toggle "Use Live Weather for Prediction" to switch between real-time and simulation inputs
   - Use "Clear Live Weather" to revert and free the state
   - Adjust "Number of map points" and "Point spread (radius)" to control density & spatial spread of generated risk markers
   - Enable the Kepler.gl advanced map checkbox for richer exploration (sized & colored by risk)
   - Expand "Debug Info" (if present) to inspect internal state while developing

## Testing
Run the unit tests with pytest (after activating your virtualenv):

```bash
pytest -q
```

Tips:
- Run a single file: `pytest tests/test_models.py -q`
- Run a single test: `pytest tests/test_models.py::test_unet_forward_and_gradients -q`
- If pytest is missing, install dependencies first: `pip install -r requirements.txt`

## Model Details
- **GBM**: Predicts flood risk probability from tabular weather data
- **U-Net**: Performs pixel-level flood mapping on terrain data

### Feature Engineering
The `scripts/feature_engineering.py` module includes utilities for:
- Outlier clipping (IQR)
- Missing value imputation
- Normalization / scaling pipelines
- Categorical encoding (placeholder / extensible)
- Temporal aggregation (e.g., daily resampling)

Integrate into a training workflow or enhance `retrain_models.py` to apply these transforms prior to model fitting.

## Retraining Models
Run `scripts/retrain_models.py` after:
- Generating new sample data
- Adding or updating real weather / flood / terrain datasets
- Adjusting feature engineering logic

Example:
```bash
python scripts/retrain_models.py --epochs 10  # (extend script to accept args if needed)
```

### Automated Scheduling
Use the provided shell script to automate periodic retraining:
```bash
./scripts/schedule_retraining.sh
```
This script logs output to `logs/retrain.log`. See `scripts/cron_setup.txt` for example cron entries:
```cron
# Example (run every day at 02:30)
30 2 * * * /path/to/venv/python /path/to/project/scripts/retrain_models.py >> /path/to/project/logs/retrain.log 2>&1
```

### Flood Mask Support
If you provide or generate a flood mask GeoTIFF (`sample_flood_mask.tif` is auto-created with sample data), you can extend U-Net training for more realistic segmentation.

## Live Weather Integration
- Powered by OpenWeatherMap current weather endpoint.
- Input your API key in the sidebar, then click the fetch button.
- A toggle lets you switch between live conditions and manual simulation without losing either.
- The "Clear Live Weather" button removes cached weather from session state.

## Map Visualization Enhancements
- Folium map shows anchor (core) high-risk locations plus deterministically jittered surrounding points.
- Point count & spread are user-controlled, enabling coarse vs dense scenario visualization.
- Dynamic color gradient (yellow → red) reflects relative risk; anchors slightly larger.
- Legend overlay clarifies risk categories.
- Kepler.gl layer (optional) adds interactive point exploration with risk-based size & color.

## Configuration Summary
| Aspect | Location / Control | Notes |
|--------|--------------------|-------|
| Sample Data Generation | Sidebar checkbox & button | Creates synthetic weather/terrain/flood/mask files |
| Live Weather Fetch | Sidebar (API key + button) | Uses OpenWeatherMap; stores in session state |
| Live vs Simulation Toggle | Appears after fetch | Switch between modes instantly |
| Clear Live Weather | Button (after fetch) | Removes live data and reverts to sliders |
| Map Point Density | Slider (5–150) | Controls number of risk markers |
| Point Spread | Select box | Approximate max radius in meters around anchor zones |
| Advanced Map | Checkbox | Displays Kepler.gl visualization |

## Troubleshooting
| Issue | Cause | Resolution |
|-------|-------|------------|
| No map points / only one marker | Low risk or earlier logic | Current version always generates points; adjust count slider |
| AttributeError: experimental_rerun | Newer Streamlit removed experimental API | Code now uses `st.rerun()` fallback logic |
| Kepler.gl not rendering | Missing dependency | Install: `pip install keplergl streamlit-keplergl` |
| Live weather not updating | API key invalid or rate limit | Verify key validity; check sidebar log message |
| Retraining script not executable | Missing permission | `chmod +x scripts/schedule_retraining.sh` |

## Security & API Usage
Avoid committing API keys. Use environment variables or `.env` (excluded via `.gitignore` recommendation). Free OpenWeatherMap tiers have rate limits; cache or throttle production calls.

## Future Improvements
- Integrate real-time weather APIs
- Add more features (soil moisture, wind speed)
- Improve U-Net training with larger datasets
- Deploy to cloud for real-time predictions
 - Add rainfall radar & river gauge ingestion
 - Persist historical prediction snapshots (time-series dashboard)
 - Implement probabilistic ensembles for uncertainty bounds
 - Add geospatial clustering & risk contour generation
