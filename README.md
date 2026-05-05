# QFA Prime Finance Platform — Streamlit v4.5 QuantStats + Strategy Education Fix

## Fixes

- QuantStats report generation now returns clear status.
- App no longer shows only "QuantStats unavailable" without explanation.
- If QuantStats fails, QFA Internal Tearsheet HTML is generated as fallback.
- Added strategy-specific report downloads.
- Added Portfolio Strategy Education tab.
- All prior v4.4 features preserved.

## Important

QuantStats can fail on Streamlit Cloud due to rendering/dependency issues. v4.5 guarantees a downloadable internal institutional tearsheet even when QuantStats fails.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```
