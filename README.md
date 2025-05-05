📌 Explanation of Each Step
1. Load and prepare dataset
A 2D synthetic dataset is generated using make_classification for binary classification.

Using 2 features helps in visualizing the decision boundary.

2. Train/Test Split
Data is split into training and testing sets (80% training, 20% testing).

3. Standardize Features
SVM is sensitive to feature scale.

We normalize using StandardScaler.

4. Train SVM with Linear Kernel
A linear hyperplane is fitted using kernel='linear'.

5. Train SVM with RBF Kernel
RBF kernel (non-linear) maps data into higher dimension to separate it better.

gamma controls the influence of a single training example.

C controls the trade-off between margin size and misclassification.

6. Evaluation
Confusion matrix shows true positives/negatives and false ones.

Classification report includes precision, recall, F1-score, and accuracy.

7. Cross-validation
cross_val_score runs the model on different parts of the data to validate generalization.

We take the mean of 5-fold CV.

8. Decision Boundary
Helps visualize how SVM is separating classes in the feature space.

Especially useful for RBF (non-linear) models.
