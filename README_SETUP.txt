RACC dashboard update: K-Means, Decision Tree, and barangay concentrations

1. Put these updated files in the SAME folder as your existing part_2_data.py and part_4_results.py:
   racc_streamlit_app.py
   part_3_models.py
   part_5_clustering.py
   requirements.txt
2. Back up your existing files before replacing them.
3. In that folder run:
   python -m pip install -r requirements.txt
   python -m streamlit run racc_streamlit_app.py
4. Upload your RACC CSV. In the sidebar, select:
   3. Classification and prediction -> Run classification (Decision Tree metrics appear)
   5. K-Means clustering -> Run K-Means (Silhouette Score and clusters appear)
   6. Accident hotspot analysis -> Analyze barangay counts

Important: This is an update pack, not a complete standalone project. Keep the original part_2_data.py and part_4_results.py. Barangay concentrations are descriptive, not verified road-segment hotspots. Model results depend on the CSV and must be validated before thesis reporting.
