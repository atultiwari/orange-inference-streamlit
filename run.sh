#!/bin/bash
export QT_QPA_PLATFORM='offscreen'
export DISPLAY=:99
streamlit run app.py --server.port=8501 --server.address=0.0.0.0
