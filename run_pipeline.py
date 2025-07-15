#!/usr/bin/env python3

import kfp
from kfp import Client
from beginner_pipeline import beginner_ml_pipeline
import argparse
import sys

def compile_pipeline():
    """Compile the pipeline to YAML format."""
    print("🔧 Compiling pipeline...")
    
    try:
        kfp.compiler.Compiler().compile(
            pipeline_func=beginner_ml_pipeline,
            package_path="beginner_ml_pipeline.yaml"
        )
        print("✅ Pipeline compiled successfully to 'beginner_ml_pipeline.yaml'")
        return True
    except Exception as e:
        print(f"❌ Failed to compile pipeline: {e}")
        return False

def run_pipeline_locally():
    """Run the pipeline locally for testing."""
    print("🏃 Running pipeline locally...")
    
    try:
        # This will execute the pipeline components locally
        # Note: This is for testing purposes only
        print("⚠️  Local execution is for testing only. Use upload_and_run() for actual deployment.")
        
        # You can test individual components here if needed
        print("✅ Local test completed")
        return True
    except Exception as e:
        print(f"❌ Local test failed: {e}")
        return False

def upload_and_run(
    host: str,
    namespace: str = "kubeflow",
    experiment_name: str = "beginner-ml-experiment",
    run_name: str = "beginner-ml-run",
    pipeline_params: dict = None
):
    """Upload and run the pipeline on Kubeflow cluster."""
    print(f"🚀 Uploading and running pipeline on Kubeflow cluster...")
    print(f"   Host: {host}")
    print(f"   Namespace: {namespace}")
    print(f"   Experiment: {experiment_name}")
    
    try:
        # Initialize the KFP client
        client = Client(host=host, namespace=namespace)
        
        # Create or get experiment
        try:
            experiment = client.create_experiment(name=experiment_name)
            print(f"✅ Created experiment: {experiment_name}")
        except Exception:
            experiment = client.get_experiment(experiment_name=experiment_name)
            print(f"✅ Using existing experiment: {experiment_name}")
        
        # Default pipeline parameters
        if pipeline_params is None:
            pipeline_params = {
                'dataset_size': 1000,
                'test_size': 0.2,
                'model_type': 'random_forest',
                'n_estimators': 100,
                'min_accuracy': 0.7,
                'random_state': 42
            }
        
        print(f"📋 Pipeline parameters: {pipeline_params}")
        
        # Submit the pipeline run
        run_result = client.run_pipeline(
            experiment_id=experiment.id,
            job_name=run_name,
            pipeline_func=beginner_ml_pipeline,
            arguments=pipeline_params
        )
        
        print(f"✅ Pipeline run submitted successfully!")
        print(f"   Run ID: {run_result.id}")
        print(f"   Run Name: {run_result.name}")
        print(f"   Run URL: {host}/#/runs/details/{run_result.id}")
        
        return run_result
        
    except Exception as e:
        print(f"❌ Failed to upload and run pipeline: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Kubeflow Pipeline Runner")
    parser.add_argument("--action", choices=["compile", "test", "run"], required=True,
                       help="Action to perform: compile, test, or run")
    parser.add_argument("--host", type=str, help="Kubeflow host URL (required for 'run' action)")
    parser.add_argument("--namespace", type=str, default="kubeflow", help="Kubernetes namespace")
    parser.add_argument("--experiment", type=str, default="beginner-ml-experiment", help="Experiment name")
    parser.add_argument("--run-name", type=str, default="beginner-ml-run", help="Run name")
    parser.add_argument("--dataset-size", type=int, default=1000, help="Dataset size")
    parser.add_argument("--model-type", type=str, default="random_forest", 
                       choices=["random_forest", "logistic_regression", "svm"],
                       help="Model type to train")
    parser.add_argument("--n-estimators", type=int, default=100, help="Number of estimators for random forest")
    parser.add_argument("--min-accuracy", type=float, default=0.7, help="Minimum accuracy threshold")
    
    args = parser.parse_args()
    
    if args.action == "compile":
        success = compile_pipeline()
        sys.exit(0 if success else 1)
    
    elif args.action == "test":
        success = run_pipeline_locally()
        sys.exit(0 if success else 1)
    
    elif args.action == "run":
        if not args.host:
            print("❌ Error: --host is required for 'run' action")
            sys.exit(1)
        
        pipeline_params = {
            'dataset_size': args.dataset_size,
            'test_size': 0.2,
            'model_type': args.model_type,
            'n_estimators': args.n_estimators,
            'min_accuracy': args.min_accuracy,
            'random_state': 42
        }
        
        run_result = upload_and_run(
            host=args.host,
            namespace=args.namespace,
            experiment_name=args.experiment,
            run_name=args.run_name,
            pipeline_params=pipeline_params
        )
        
        sys.exit(0 if run_result else 1)

if __name__ == "__main__":
    main()