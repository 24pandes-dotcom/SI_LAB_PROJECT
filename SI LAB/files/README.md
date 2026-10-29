# Daily Temperature Patterns & Weather Analysis – Streamlit Dashboard

## Folder layout
```
weather_app/
├── app.py
├── requirements.txt
├── daily_temperature_patterns.csv   <-- CSV must be in the SAME folder as app.py
└── .streamlit/config.toml           <-- theme colours
```

## Install & run (Windows / macOS / Linux)
```bash
cd weather_app
python -m venv venv
venv\Scripts\activate            # Windows      |  source venv/bin/activate   (macOS/Linux)
pip install -r requirements.txt
streamlit run app.py
```
The site opens at http://localhost:8501. (If the CSV is missing, the sidebar lets you upload it.)

## Pre-presentation checklist
- [ ] `streamlit run app.py` starts with no red error boxes
- [ ] Home: 5 metric cards show; change date range and seasons – numbers update
- [ ] Open every one of the 10 sections from the sidebar once
- [ ] Sampling: change sample size; four methods appear in the comparison table
- [ ] CLT: n = 10, 30, 50 histograms get narrower; SE table matches σ/√n
- [ ] Bootstrap and Confidence Intervals: change the seed – results change; same seed – same results
- [ ] Predictive: metrics table shown; submit the prediction form; note the result is an estimate
- [ ] Try the chronological split and be ready to explain why it scores poorly (one year of data)
- [ ] Reset filters to "all seasons" and the full date range before presenting
