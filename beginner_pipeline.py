#!/usr/bin/env python3

import kfp
from kfp import dsl
from kfp.dsl import component, pipeline, Input, Output, Dataset, Model, Metrics
from typing import NamedTuple

@component(
    base_image="python:3.9",
    packages_to_install=["pandas", "scikit-learn", "numpy"]
)
def load_and_preprocess_data(
    output_dataset: Output[Dataset],
    dataset_size: int = 1000,
    test_size: float = 0.2,
    random_state: int = 42
) -> NamedTuple('DataInfo', [('num_samples', int), ('num_features', int)]):
    """Load and preprocess sample data for training."""
    import pandas as pd
    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import pickle
    
    # Generate sample classification dataset
    X, y = make_classification(
        n_samples=dataset_size,
        n_features=20,
        n_informative=10,
        n_redundant=10,
        n_clusters_per_class=1,
        random_state=random_state
    )
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the processed data
    data_dict = {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler
    }
    
    with open(output_dataset.path, 'wb') as f:
        pickle.dump(data_dict, f)
    
    from collections import namedtuple
    DataInfo = namedtuple('DataInfo', ['num_samples', 'num_features'])
    return DataInfo(num_samples=len(X_train), num_features=X_train.shape[1])

@component(
    base_image="python:3.9",
    packages_to_install=["pandas", "scikit-learn", "numpy"]
)
def train_model(
    input_dataset: Input[Dataset],
    output_model: Output[Model],
    model_type: str = "random_forest",
    n_estimators: int = 100,
    random_state: int = 42
) -> NamedTuple('ModelInfo', [('model_type', str), ('n_estimators', int)]):
    """Train a machine learning model."""
    import pickle
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    
    # Load the data
    with open(input_dataset.path, 'rb') as f:
        data_dict = pickle.load(f)
    
    X_train = data_dict['X_train']
    y_train = data_dict['y_train']
    
    # Choose and train the model
    if model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    elif model_type == "logistic_regression":
        model = LogisticRegression(random_state=random_state, max_iter=1000)
    elif model_type == "svm":
        model = SVC(random_state=random_state, probability=True)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Save the model
    with open(output_model.path, 'wb') as f:
        pickle.dump(model, f)
    
    from collections import namedtuple
    ModelInfo = namedtuple('ModelInfo', ['model_type', 'n_estimators'])
    return ModelInfo(model_type=model_type, n_estimators=n_estimators)

@component(
    base_image="python:3.9",
    packages_to_install=["pandas", "scikit-learn", "numpy"]
)
def evaluate_model(
    input_dataset: Input[Dataset],
    input_model: Input[Model],
    output_metrics: Output[Metrics]
) -> NamedTuple('EvaluationResults', [('accuracy', float), ('precision', float), ('recall', float), ('f1_score', float)]):
    """Evaluate the trained model."""
    import pickle
    import json
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
    
    # Load the data and model
    with open(input_dataset.path, 'rb') as f:
        data_dict = pickle.load(f)
    
    with open(input_model.path, 'rb') as f:
        model = pickle.load(f)
    
    X_test = data_dict['X_test']
    y_test = data_dict['y_test']
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Create metrics dictionary
    metrics_dict = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'classification_report': classification_report(y_test, y_pred)
    }
    
    # Save metrics
    with open(output_metrics.path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    
    print(f"Model Evaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    from collections import namedtuple
    EvaluationResults = namedtuple('EvaluationResults', ['accuracy', 'precision', 'recall', 'f1_score'])
    return EvaluationResults(accuracy=accuracy, precision=precision, recall=recall, f1_score=f1)

@component(
    base_image="python:3.9",
    packages_to_install=["pandas", "scikit-learn", "numpy"]
)
def validate_model(
    accuracy: float,
    min_accuracy: float = 0.7
) -> str:
    """Validate if the model meets minimum performance criteria."""
    if accuracy >= min_accuracy:
        result = f"✅ Model PASSED validation with accuracy {accuracy:.4f} (minimum: {min_accuracy})"
        print(result)
        return "PASSED"
    else:
        result = f"❌ Model FAILED validation with accuracy {accuracy:.4f} (minimum: {min_accuracy})"
        print(result)
        return "FAILED"

@pipeline(
    name="beginner-ml-pipeline",
    description="A beginner-friendly ML pipeline demonstrating data processing, training, and evaluation",
    version="1.0.0"
)
def beginner_ml_pipeline(
    dataset_size: int = 1000,
    test_size: float = 0.2,
    model_type: str = "random_forest",
    n_estimators: int = 100,
    min_accuracy: float = 0.7,
    random_state: int = 42
):
    """
    A complete ML pipeline for beginners that includes:
    1. Data loading and preprocessing
    2. Model training
    3. Model evaluation
    4. Model validation
    """
    
    # Step 1: Load and preprocess data
    data_task = load_and_preprocess_data(
        dataset_size=dataset_size,
        test_size=test_size,
        random_state=random_state
    )
    
    # Step 2: Train the model
    train_task = train_model(
        input_dataset=data_task.outputs['output_dataset'],
        model_type=model_type,
        n_estimators=n_estimators,
        random_state=random_state
    )
    
    # Step 3: Evaluate the model
    eval_task = evaluate_model(
        input_dataset=data_task.outputs['output_dataset'],
        input_model=train_task.outputs['output_model']
    )
    
    # Step 4: Validate the model
    validation_task = validate_model(
        accuracy=eval_task.outputs['accuracy'],
        min_accuracy=min_accuracy
    )
    
    # Set task display names for better visualization
    data_task.set_display_name("Data Processing")
    train_task.set_display_name("Model Training")
    eval_task.set_display_name("Model Evaluation")
    validation_task.set_display_name("Model Validation")

if __name__ == "__main__":
    # Compile the pipeline
    kfp.compiler.Compiler().compile(
        pipeline_func=beginner_ml_pipeline,
        package_path="beginner_ml_pipeline.yaml"
    )
    print("Pipeline compiled successfully to 'beginner_ml_pipeline.yaml'")