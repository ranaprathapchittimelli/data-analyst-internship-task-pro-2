HR Analytics – Employee Attrition Prediction
A data analytics and machine learning project that explores employee attrition patterns and builds classification models to help understand factors associated with employees leaving an organization.
Project status: EDA, Logistic Regression, Decision Tree, SHAP plots, and Power BI visuals have been completed according to the project progress notes. Add verified model metrics and screenshots before publishing.

Objectives
- Explore employee attrition patterns across departments, job roles, salary levels, and other employee attributes.
- Clean and prepare the HR dataset for analysis.
- Train and compare Logistic Regression and Decision Tree classification models.
- Evaluate models using appropriate classification metrics and confusion matrices.
- Interpret model behaviour using SHAP visualizations.
- Present key findings through a Power BI dashboard.
- Develop data-informed recommendations that may help an organization investigate and address attrition.
Tools and Technologies
- Python
- Pandas – data preparation and analysis
- Matplotlib / Seaborn – exploratory visualizations
- Scikit-learn – preprocessing, classification, and evaluation
- SHAP – model explanation
- Power BI – interactive dashboard
- VS Code – development environment
Dataset
This project uses the IBM HR Employee Attrition dataset, commonly distributed as WA_Fn-UseC_-HR-Employee-Attrition.csv.

The target column is Attrition (typically Yes or No). The dataset contains employee attributes such as department, job role, age, monthly income, job satisfaction, years at the company, and other work-related factors.

Dataset note: Review the dataset's source, license, and attribution requirements. Do not include confidential employee records or other private information in a public repository. If redistribution is not permitted, publish instructions for obtaining the dataset instead of committing the CSV.
Repository Structure
Organize the repository approximately like this, adjusting filenames to match the files you actually have:
HR-Analytics-Employee-Attrition/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── src/
│   ├── hr_analytics.py
│   └── hr_analytics_model.py
├── outputs/
│   ├── model_comparison.csv
│   ├── logistic_regression_confusion_matrix.png
│   ├── decision_tree_confusion_matrix.png
│   ├── shap_feature_importance.png
│   └── shap_summary_plot.png
├── powerbi/
│   └── HR_Analytics_Dashboard.pbix
├── screenshots/
│   └── dashboard_preview.png
└── reports/
    └── attrition_recommendations.pdf
Keep only files that exist. If your scripts currently live in the project root, you can either move them into src/ and update file paths, or leave them in the root and change the structure shown here. Do not commit empty placeholder files.
Project Workflow
1. Data loading and quality checks – inspect dimensions, column names, data types, missing values, and duplicates.
2. Exploratory Data Analysis (EDA) – analyze attrition across department, job role, salary bands, promotions, tenure, and other available attributes.
3. Preprocessing and feature preparation – encode categorical columns and prepare training and test data.
4. Model training – train Logistic Regression and Decision Tree classifiers.
5. Model evaluation – compare accuracy, precision, recall, F1-score, and confusion matrices. Because attrition classes can be imbalanced, do not rely on accuracy alone.
6. Model interpretation – use SHAP plots to examine how input features contribute to model predictions.
7. Power BI reporting – present high-level KPIs and attrition breakdowns with appropriate filters.
8. Recommendations – use the observed patterns as hypotheses for further HR investigation, not as proof of causation.
How to Run
1. Clone the repository
git clone YOUR_GITHUB_REPOSITORY_URL
cd HR-Analytics-Employee-Attrition
2. Create a virtual environment (recommended)
python -m venv .venv
On Windows:
.venv\Scripts\activate
3. Install dependencies
python -m pip install -r requirements.txt
4. Place the dataset
Put the CSV in the location expected by your scripts. For example, if your code reads from data/WA_Fn-UseC_-HR-Employee-Attrition.csv, save it there and make sure the path in the script matches.
5. Run the analysis
Use the filenames and paths in your repository. If the scripts are in src/, for example:
python src/hr_analytics.py
python src/hr_analytics_model.py
The scripts should save charts and evaluation outputs to outputs/. Confirm that your code creates the output directory before saving files.
Model Evaluation
Fill in this table using the actual output from your latest run. Do not estimate or invent results.
Metric	Logistic Regression	Decision Tree
Accuracy	Add actual value	Add actual value
Precision (Attrition = Yes)	Add actual value	Add actual value
Recall (Attrition = Yes)	Add actual value	Add actual value
F1-score (Attrition = Yes)	Add actual value	Add actual value


Include the confusion matrices and briefly explain the trade-off between detecting employees who leave (recall for Attrition = Yes) and false alarms (precision).
Power BI Dashboard
Add a screenshot of the completed dashboard to screenshots/ and save the editable Power BI report in powerbi/ if you have the .pbix file.

Suggested dashboard content, where supported by the data:
- Total employee count
- Attrition count and attrition rate
- Attrition by department and job role
- Attrition by age or salary band
- Attrition by years at company
- Filters for department, gender, job role, and other relevant fields
Only describe visuals that are actually present in your dashboard.
SHAP Interpretation
Include the generated plots, such as shap_summary_plot.png and shap_feature_importance.png, if these are the actual filenames in your project. Explain the top features based on your plots and model output. SHAP explanations describe the model's learned patterns; they do not prove that a feature causes attrition.
Findings and Recommendations
Add 3–5 evidence-based findings after reviewing your actual EDA, model results, and SHAP plots. For each recommendation, state the supporting result and any limitation. Avoid presenting correlations or model explanations as causal conclusions.
Limitations
- The dataset may not represent every organization or workforce.
- Model performance can be affected by class imbalance, data leakage, feature choices, and the train/test split.
- Feature importance and SHAP values explain model behaviour, not causation.
- Predictions should support careful review and should not be used as the sole basis for decisions about individual employees.
Future Improvements
- Tune model hyperparameters and compare additional suitable classifiers.
- Validate performance with cross-validation and class-specific metrics.
- Review fairness and potential bias across relevant employee groups.
- Improve dashboard usability and document data definitions.
- Validate recommendations with HR stakeholders and additional data.
