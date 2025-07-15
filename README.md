# Kubeflow Pipeline Beginner Example

This repository contains a beginner-friendly Kubeflow Pipeline example that demonstrates a complete machine learning workflow including data processing, model training, evaluation, and validation.

## 🎯 What This Pipeline Does

The pipeline implements a complete ML workflow:

1. **Data Processing**: Generates synthetic classification data and preprocesses it
2. **Model Training**: Trains a machine learning model (Random Forest, Logistic Regression, or SVM)
3. **Model Evaluation**: Evaluates the model performance using standard metrics
4. **Model Validation**: Validates if the model meets minimum performance criteria

## 📁 Project Structure

```
kubeflow-pipeline-2/
├── beginner_pipeline.py      # Main pipeline definition
├── run_pipeline.py           # Pipeline runner script
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Compile the Pipeline

```bash
python run_pipeline.py --action compile
```

This creates `beginner_ml_pipeline.yaml` which can be uploaded to Kubeflow.

### 3. Run on Kubeflow Cluster

```bash
python run_pipeline.py --action run --host <KUBEFLOW_HOST>
```

Replace `<KUBEFLOW_HOST>` with your Kubeflow cluster endpoint (e.g., `http://localhost:8080`).

## 🔧 Pipeline Components

### Data Processing Component
- Generates synthetic classification dataset
- Splits data into training and testing sets
- Applies standard scaling to features
- Saves preprocessed data for downstream components

### Model Training Component
- Supports multiple model types: Random Forest, Logistic Regression, SVM
- Configurable hyperparameters
- Saves trained model for evaluation

### Model Evaluation Component
- Calculates accuracy, precision, recall, and F1-score
- Generates detailed classification report
- Saves metrics for validation

### Model Validation Component
- Checks if model meets minimum accuracy threshold
- Provides clear pass/fail feedback

## ⚙️ Configuration Options

You can customize the pipeline with these parameters:

```bash
python run_pipeline.py --action run \
  --host <KUBEFLOW_HOST> \
  --dataset-size 2000 \
  --model-type logistic_regression \
  --n-estimators 200 \
  --min-accuracy 0.8 \
  --experiment my-experiment \
  --run-name my-run
```

### Available Parameters:
- `--dataset-size`: Number of samples in synthetic dataset (default: 1000)
- `--model-type`: Model to train (`random_forest`, `logistic_regression`, `svm`)
- `--n-estimators`: Number of trees for Random Forest (default: 100)
- `--min-accuracy`: Minimum accuracy threshold for validation (default: 0.7)
- `--experiment`: Kubeflow experiment name (default: "beginner-ml-experiment")
- `--run-name`: Pipeline run name (default: "beginner-ml-run")

## 📊 Expected Results

The pipeline will:
1. Process 1000 samples with 20 features
2. Train a Random Forest model with 100 estimators
3. Achieve accuracy typically between 0.85-0.95
4. Pass validation with default 0.7 accuracy threshold

## 🛠️ Troubleshooting

### Common Issues:

1. **Connection Error**: Ensure your Kubeflow cluster is accessible and the host URL is correct
2. **Permission Error**: Make sure you have proper credentials configured for your Kubeflow cluster
3. **Component Failures**: Check the Kubeflow UI for detailed error logs

### Pipeline Monitoring:

After running the pipeline, you can monitor progress in the Kubeflow UI:
- Navigate to `<KUBEFLOW_HOST>/#/experiments`
- Find your experiment and click on the run
- Monitor individual component execution and logs

## 🎓 Learning Resources

This example demonstrates:
- ✅ Kubeflow Pipeline component creation
- ✅ Data passing between components
- ✅ Pipeline parameters and configuration
- ✅ Model training and evaluation patterns
- ✅ Error handling and validation

## 📈 Next Steps

After mastering this example, consider:
1. Adding more complex data processing steps
2. Implementing hyperparameter tuning
3. Adding model comparison components
4. Integrating with external data sources
5. Adding model deployment components

Happy learning with Kubeflow Pipelines! 🚀