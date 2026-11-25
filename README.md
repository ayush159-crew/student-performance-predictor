# Student Performance Lab v2
Run `pip install -r requirements.txt` then `streamlit run app.py`.
Deploy by replacing app.py and requirements.txt in your existing GitHub repo, adding engine.py, and retaining student_performance.csv. Commit and reboot the Streamlit app if necessary.
Synthetic demo is bundled. UCI Mathematics data is fetched on demand from https://archive.ics.uci.edu/dataset/320/student+performance ; if unavailable download student-mat.csv and upload manually. UCI G3 is on a 0–20 scale. No real-world accuracy is claimed for synthetic data.
Models use a common holdout and five-fold training-only CV; lowest CV MAE wins. Preprocessing is fitted inside each fold. Exclude leakage and IDs manually. Random splits assume independent rows. Feature importance is not causal. Uploaded datasets need at least 30 valid rows, numeric target and at least one feature. No permanent storage of uploads is implemented.
