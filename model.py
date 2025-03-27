from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV

def create_and_tune_model(X_train, y_train):
    print("Tuning hyperparameters with RandomizedSearchCV...")
    param_dist = {
        'hidden_layer_sizes': [(128, 64), (256, 128), (128, 128, 64)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'sgd'],
        'learning_rate_init': [0.001, 0.01, 0.1]
    }

    random_search = RandomizedSearchCV(
        MLPClassifier(max_iter=50, random_state=42, verbose=True), 
        param_distributions=param_dist, 
        n_iter=10, 
        cv=3, 
        n_jobs=-1, 
        random_state=42,
        verbose=2
    )
    random_search.fit(X_train, y_train)
    print(f"Best parameters: {random_search.best_params_}")
    return random_search.best_estimator_
