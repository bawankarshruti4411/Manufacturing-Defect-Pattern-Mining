# Manufacturing Defect Pattern Mining
Flask API + HTML/CSS/vanilla-JS dashboard on a scikit-learn model.

```bash
pip install -r requirements.txt
python scripts/train_model.py     # trains 4 models, saves model/*.joblib + metadata
python backend/app.py             # then open http://127.0.0.1:5000
```
Folders: data/ (CSV) · scripts/ (training) · model/ (saved artifact) · backend/app.py (API) · frontend/ (index.html, style.css, script.js).
