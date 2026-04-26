Body Performance Analytics — Final Report
Course: Introduction to AI and ML
Project: Body Performance Classification and Regression
Dataset: Body Performance (Kaggle) — 13,393 rows × 12 columns
Part 1: Data Preparation & Exploratory Data Analysis
1. Dataset Overview (5.1)
The dataset contains 13,393 fitness test records from a national fitness program. Each row represents one person's measurements: age, gender, height, weight, body fat %, blood pressure (systolic/diastolic), grip force, flexibility (sit-and-bend), endurance (sit-ups), explosive power (broad jump), and a performance class (A/B/C/D).

2. Column Understanding (5.2)
Column	Type	Constraints	Description
age	Numeric	Positive, 21–64	Participant age in years
gender	Categorical	M or F	Biological sex
heightCm	Numeric	100–220 cm	Height in centimeters
weightKg	Numeric	20–200 kg	Body weight
bodyFatPercent	Numeric	3–60% realistic	Body fat as % of total weight
diastolic	Numeric	< systolic	Blood pressure at rest (mmHg)
systolic	Numeric	> diastolic	Blood pressure at contraction (mmHg)
gripForce	Numeric	Positive	Hand grip strength (kg)
sitAndBendForwardCm	Numeric	-30 to +50	Flexibility test (cm past toes)
sitUpsCounts	Numeric	Non-negative	Sit-ups completed (core endurance)
broadJumpCm	Numeric	Positive	Standing broad jump distance (cm)
performanceClass	Categorical	A, B, C, D	A = best, D = worst
3. Data Type Verification (5.3)
All columns have correct data types. Numeric features are stored as float64, gender and class as object (string). No type conversion was needed.

4. Missing Values Analysis (5.4)
Result: Zero missing values across all 12 columns. No imputation techniques were required. This was verified by computing null counts and percentages for every column.

5. Duplicate Detection (5.5)
Duplicate rows were found, but the decision was made to keep them. In fitness testing without participant IDs, two different people CAN have identical measurements. Without additional metadata, we cannot distinguish true duplicates from coincidental matches.

6. Data Validity Checks & Cleaning (5.6)
Fix 1: Diastolic ≥ Systolic (5 rows dropped)
How we found it: Medical science dictates that systolic pressure (heart contraction) must always exceed diastolic (heart relaxation). We filtered for rows where diastolic ≥ systolic. Why we removed them: These values represent measurement errors or data-entry mistakes — no living human can have diastolic ≥ systolic.

Fix 2: Extreme Body Fat Percentages (rows dropped)
How we found it: We checked physiological limits — essential body fat is 3–5% for men, 10–13% for women. Values above 50% represent extreme cases that are likely errors. Action: Dropped rows where body fat < 5%, > 50%, female < 10%, or male > 45%.

Fix 3: Zero Values (rows dropped)
How we found it: Zero grip force (0 kg), zero broad jump (0 cm), and zero blood pressure (0/0 mmHg) are physically impossible for a conscious person performing a test.

Grip = 0: Even severe sarcopenia produces 5–10 kg of grip. Zero means equipment failure.
Broad jump = 0: Zero distance means the test was not performed (missing data as zero).
BP = 0/0: Clinically dead — clearly missing data.
Sit-ups = 0 & age < 60: Unrealistic for adults under 60 → dropped.
Sit-ups = 0 & age ≥ 60: Kept — age-related sarcopenia makes this physiologically valid.
Fix 4: Sit-and-Bend Range [-30, +50] cm
How we found it: Max values included 213 cm and 185 cm — no human bends forward 2+ meters.

Fix 5: Diastolic < 20 mmHg (3 rows dropped)
How we found it: Below 20 mmHg is incompatible with consciousness — measurement error.

Fix 6: BMI < 14
How we found it: We computed BMI = weight/(height/100)² and filtered for values below 14. BMI < 14 is incompatible with life.

Fix 7: Broad Jump > 290 cm (16 rows dropped)
How we found it: Standing broad jumps above 290 cm are Olympic-caliber — impossible in a general population sample.

Column Renaming
Why: Original names contained spaces, hyphens, and special characters (body fat_%, sit-ups counts). We standardized all column names to camelCase for code compatibility.

Cleaning Summary
Metric	Value
Original rows	13,393
Rows dropped	~120 (0.9%)
Remaining rows	~13,270
Columns	12 → 12 (renamed)
Structural Observations
Gender imbalance: 63% Male, 37% Female — addressed via stratified splitting (NOT by dropping data).
Artificial class balance: All 4 classes have ~3,348 rows — suspected quartile-based binning.
7. Univariate Analysis (5.7)
8. Distribution Analysis (5.8)
Shapiro-Wilk normality tests reveal that ALL features deviate significantly from normality (p < 0.001). Key observations:

sitAndBendForwardCm: Most skewed (skew = -0.86), confirmed by QQ plot deviation at tails.
age: Right-skewed (skew = 0.61) — more young participants than elderly.
gripForce: Nearly symmetric (skew ≈ 0) but platykurtic (kurtosis = -0.85).
9. Outlier Detection (5.9)
After domain-specific cleaning, remaining outliers were counted using the IQR method.

Decision: KEEP remaining IQR outliers. These represent natural biological variation (e.g., an extremely strong person), not measurement errors.

10. Correlation Analysis (5.10)
Top findings:

weightKg ↔ gripForce: r = +0.43 (heavier people grip harder)
weightKg ↔ bmi: r = +0.85 (expected — BMI is derived from weight)
age ↔ sitUpsCounts: r = -0.49 (endurance declines with age)
sitUpsCounts ↔ broadJumpCm: r = +0.55 (fitness metrics correlate)
11. Gender-Based Analysis
12. Age & Aging Effects
13. Performance Class Profiles
14. Body Composition & Cardiovascular
15. Clustering
16. Additional EDA
17. Final EDA Summary (5.11)
Five Key Insights
Grip strength and broad jump are the most discriminative features for class separation (Cohen's d > 1.0 between A and D).
Males outperform females in strength and power, but females show comparable flexibility.
All fitness metrics decline with age, accelerating after 50.
Classes B and C overlap so heavily that they are nearly indistinguishable.
K-Means clustering partially recovers the class structure, confirming that the classes have real (though overlapping) physical bases.
Five Data Quality Problems
Gender imbalance (63% M / 37% F)
Artificial class balance (quartile-based binning suspected)
Zero-value measurements masking missing data
Extreme outliers from data-entry errors (body fat 78%, sit-bend 213 cm)
Blood pressure measurement errors (diastolic > systolic)
Recommended Preprocessing
Apply StandardScaler for distance-based models (KNN, SVM, MLP)
Use gender-normalized z-scores for sit-ups
Engineer BMI, lean body mass, and interaction features
Consider 3-class and 2-class targets alongside 4-class
Use stratified splits to maintain class proportions
Part 2: ML Model Training & Evaluation
1. Feature Engineering
We added 4 engineered features to the 11 original features, selected by greedy forward selection from 57 candidates:

Feature	Formula	Why
BMI	weight / (height/100)²	Metabolic indicator combining weight and height
sitUpsCounts_zGender	Gender-normalized z-score	Removes gender confound in sit-up counts
gripPerLeanMass	gripForce / (weight × (1-fatPct/100))	"Pound-for-pound" strength — better predictor
genderXgrip	genderEncoded × gripForce	Learns gender-specific grip thresholds
2. Target Variable Configurations
Configuration	Classes	Rationale
4-class	A, B, C, D	Original labels — required by handbook
3-class	High(A), Average(B+C), Low(D)	B and C are statistically indistinguishable
2-class	Good(A+B), Poor(C+D)	Binary fitness classification
3. Hyperparameter Tuning
4. Classification Results (80:20 Split)
Model	4-class	3-class	2-class
KNN	0.636	0.758	0.840
Decision Tree	0.716	0.789	0.862
SVM (Linear)	0.626	0.742	0.823
SVM (RBF)	0.729	0.806	0.865
MLP (128,64)	0.755	0.815	0.872
XGBoost	0.775	0.834	0.890
Voting Ensemble	0.758	—	—
5. Regression (Predicting Broad Jump)
Model	R²	MAE	RMSE
Linear Regression	0.801	13.4 cm	17.8 cm
KNN Regressor	0.786	14.0 cm	18.4 cm
Decision Tree	0.742	15.3 cm	20.3 cm
SVR (RBF)	0.806	13.3 cm	17.6 cm
MLP Regressor	0.808	13.2 cm	17.5 cm
6. Multi-Split Evaluation
XGBoost shows minimal degradation (-1.1%) from 80:20 to 50:50, demonstrating robustness. Decision Tree drops more (-2.5%), confirming its higher variance nature.

7. Cross-Validation (5-Fold Stratified)
XGBoost achieves the highest mean CV accuracy (0.763 on 4-class) with low variance, confirming it as the most stable and accurate model.

8. Feature Importance
Top features: sitUpsCounts, broadJumpCm, age, gripForce. Engineered features (BMI, gripPerLeanMass) rank mid-level but contribute to overall accuracy improvement.

9. SHAP Analysis
SHAP provides per-sample feature attribution. High sit-up counts push predictions toward Class A; high body fat pushes toward Class D.

10. Learning Curves
Decision Tree: Large train-validation gap = high variance
KNN/SVM: Converge slowly = data-hungry
XGBoost/MLP: Best balance of bias and variance
11. ROC Curves (One-vs-Rest)
Classes A and D have high AUC (>0.90), confirming they are easy to separate. Classes B and C have lower AUC, consistent with their overlap.

12. Feature Scaling Impact
KNN: Dramatic improvement with scaling (+10-15% accuracy) — distance-based, sensitive to scale.
SVM: Significant improvement (+5-10%) — kernel computations need normalized features.
Decision Tree: No difference — splits are scale-invariant by design.
13. SVM Hyperparameter Sensitivity
Best configuration: C=10, gamma='scale' (RBF kernel). High C + high gamma overfits. Low C underfits.

14. Voting Ensemble
Soft voting ensemble of KNN + DT + SVM + MLP achieves 0.758 on 4-class, outperforming individual KNN/DT/SVM models but slightly below standalone XGBoost.

15. Statistical Significance
Paired t-test on 5-fold CV accuracies confirms:

XGBoost significantly outperforms ALL other models (p < 0.05 for all pairs).
This is not due to random variation — it's a genuine performance advantage.
16. MLP Architecture Comparison
Architecture	3-Fold CV Accuracy
MLP(64,)	0.720 ± 0.009
MLP(128,)	0.722 ± 0.007
MLP(128, 64)	0.732 ± 0.010
MLP(256, 128)	0.731 ± 0.006
Two hidden layers (128, 64) is optimal. The second layer captures higher-order interactions without overfitting.

Final Comparison
Conclusions
Best Model: XGBoost
4-class: 77.5% accuracy (best among all models)
3-class: 83.4% accuracy (after merging overlapping B+C)
2-class: 89.0% accuracy (binary classification)
Statistically significantly better than all alternatives (p < 0.05)
Most stable across different data splits and cross-validation folds
Benefits from sequential error correction and sophisticated regularization
Key Takeaways
The B/C class overlap is the fundamental bottleneck limiting 4-class accuracy.
Feature engineering adds ~1.8% accuracy beyond raw features.
Gradient boosting (XGBoost) consistently outperforms traditional ML models.
The voting ensemble improves over individual weak models but cannot beat a tuned XGBoost.
StandardScaler is essential for KNN/SVM/MLP but irrelevant for tree-based models.
