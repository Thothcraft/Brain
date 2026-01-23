"""Example Usage of the Modular FL System.

This file demonstrates how to use the FL module for various scenarios.
Run these examples to test the FL pipeline.
"""

import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def example_quick_test():
    """Quick test with minimal configuration."""
    from server.fl import (
        create_experiment,
        FLExperimentRunner,
        generate_experiment_report,
        format_report_as_text,
    )
    
    print("\n" + "="*60)
    print("EXAMPLE 1: Quick Test")
    print("="*60)
    
    # Create a quick test experiment
    experiment = create_experiment(
        name="Quick-Test-CIFAR10",
        algorithm="fedavg",
        model="cnn",
        dataset="cifar10",
        num_partitions=3,
        num_rounds=5,
        local_epochs=1,
        num_runs=1,
    )
    
    # Run the experiment
    runner = FLExperimentRunner()
    result = await runner.run(experiment)
    
    # Generate report
    report = generate_experiment_report(result)
    print(format_report_as_text(report))
    
    return result


async def example_algorithm_comparison():
    """Compare different FL algorithms."""
    from server.fl import (
        create_experiment,
        FLExperimentRunner,
        generate_comparative_report,
    )
    
    print("\n" + "="*60)
    print("EXAMPLE 2: Algorithm Comparison")
    print("="*60)
    
    # Create experiments for different algorithms
    experiments = [
        create_experiment(
            name="FedAvg",
            algorithm="fedavg",
            model="cnn",
            dataset="cifar10",
            num_partitions=5,
            num_rounds=20,
            num_runs=2,
        ),
        create_experiment(
            name="FedProx",
            algorithm="fedprox",
            model="cnn",
            dataset="cifar10",
            num_partitions=5,
            num_rounds=20,
            num_runs=2,
            proximal_mu=0.01,
        ),
        create_experiment(
            name="FedAdam",
            algorithm="fedadam",
            model="cnn",
            dataset="cifar10",
            num_partitions=5,
            num_rounds=20,
            num_runs=2,
            server_learning_rate=0.1,
        ),
    ]
    
    # Run all experiments
    runner = FLExperimentRunner()
    results = await runner.run_queue(experiments)
    
    # Generate comparative report
    report = generate_comparative_report(results)
    
    print("\n--- Comparative Results ---")
    for exp in report["summary"]:
        print(f"  {exp['name']}: {exp['mean_accuracy']:.4f} ± {exp['std_accuracy']:.4f}")
    
    print(f"\nBest: {report['best_performer']['name']}")
    
    return results


async def example_noniid_comparison():
    """Compare IID vs Non-IID data distributions."""
    from server.fl import (
        create_experiment,
        FLExperimentRunner,
        plot_sample_distribution,
        get_all_label_distributions,
    )
    
    print("\n" + "="*60)
    print("EXAMPLE 3: IID vs Non-IID Comparison")
    print("="*60)
    
    experiments = [
        create_experiment(
            name="IID",
            algorithm="fedavg",
            model="cnn",
            dataset="cifar10",
            num_partitions=5,
            num_rounds=20,
            partition_strategy="iid",
            num_runs=2,
        ),
        create_experiment(
            name="Non-IID (α=0.5)",
            algorithm="fedavg",
            model="cnn",
            dataset="cifar10",
            num_partitions=5,
            num_rounds=20,
            partition_strategy="non_iid_dirichlet",
            dirichlet_alpha=0.5,
            num_runs=2,
        ),
        create_experiment(
            name="Non-IID (α=0.1)",
            algorithm="fedavg",
            model="cnn",
            dataset="cifar10",
            num_partitions=5,
            num_rounds=20,
            partition_strategy="non_iid_dirichlet",
            dirichlet_alpha=0.1,
            num_runs=2,
        ),
    ]
    
    runner = FLExperimentRunner()
    results = await runner.run_queue(experiments)
    
    # Show sample distribution for non-IID case
    distributions = get_all_label_distributions(
        num_partitions=5,
        dataset="cifar10",
        partition_strategy="non_iid_dirichlet",
        dirichlet_alpha=0.1,
    )
    
    print("\n--- Label Distribution (α=0.1) ---")
    for client_id, dist in distributions.items():
        labels = sorted(dist.keys())
        counts = [dist.get(l, 0) for l in labels]
        print(f"  Client {client_id}: {counts}")
    
    # Plot if matplotlib available
    fig = plot_sample_distribution(distributions, num_classes=10, title="Non-IID Distribution (α=0.1)")
    if fig:
        fig.savefig("noniid_distribution.png")
        print("\nSaved distribution plot to noniid_distribution.png")
    
    return results


async def example_model_comparison():
    """Compare different model architectures."""
    from server.fl import (
        create_experiment,
        FLExperimentRunner,
        plot_accuracy_curves,
    )
    
    print("\n" + "="*60)
    print("EXAMPLE 4: Model Architecture Comparison")
    print("="*60)
    
    experiments = [
        create_experiment(
            name="SimpleCNN",
            algorithm="fedavg",
            model="cnn",
            dataset="cifar10",
            num_partitions=5,
            num_rounds=30,
            num_runs=2,
        ),
        create_experiment(
            name="ResNet18",
            algorithm="fedavg",
            model="resnet18",
            dataset="cifar10",
            num_partitions=5,
            num_rounds=30,
            num_runs=2,
        ),
    ]
    
    runner = FLExperimentRunner()
    results = await runner.run_queue(experiments)
    
    # Prepare data for plotting
    plot_data = []
    for result in results:
        curves = []
        for run in result.runs:
            if run.round_metrics:
                curves.append([m.accuracy for m in run.round_metrics])
        plot_data.append({
            "name": result.config.name,
            "curves": curves,
        })
    
    # Plot accuracy curves
    fig = plot_accuracy_curves(plot_data, title="Model Comparison on CIFAR-10")
    if fig:
        fig.savefig("model_comparison.png")
        print("\nSaved accuracy curves to model_comparison.png")
    
    return results


async def example_using_pipelines():
    """Use pre-defined pipelines."""
    from server.fl import (
        DEFAULT_PIPELINES,
        list_pipelines,
        pipeline_to_experiments,
        FLExperimentRunner,
    )
    
    print("\n" + "="*60)
    print("EXAMPLE 5: Using Default Pipelines")
    print("="*60)
    
    # List available pipelines
    print("\nAvailable Pipelines:")
    for pipeline in list_pipelines():
        print(f"  - {pipeline['id']}: {pipeline['name']}")
        print(f"    {pipeline['description']}")
    
    # Use the quick_test pipeline
    experiments = pipeline_to_experiments("quick_test")
    
    runner = FLExperimentRunner()
    results = await runner.run_queue(experiments)
    
    print(f"\nPipeline completed: {results[0].mean_accuracy:.4f} accuracy")
    
    return results


async def example_knowledge_distillation():
    """Example with knowledge distillation for heterogeneous models."""
    from server.fl import (
        create_experiment,
        FLExperimentRunner,
    )
    
    print("\n" + "="*60)
    print("EXAMPLE 6: Knowledge Distillation (Heterogeneous Models)")
    print("="*60)
    
    # FedDF allows different model architectures per client
    experiment = create_experiment(
        name="FedDF-Heterogeneous",
        algorithm="feddf",
        model="cnn",  # Default model, but clients can have different
        dataset="cifar10",
        num_partitions=5,
        num_rounds=20,
        temperature=3.0,
        distillation_weight=0.5,
        num_runs=1,
    )
    
    runner = FLExperimentRunner()
    result = await runner.run(experiment)
    
    print(f"\nFedDF Result: {result.mean_accuracy:.4f} accuracy")
    
    return result


async def example_session_manager():
    """Example using the session manager for real-time tracking."""
    from server.fl import (
        FLSessionManager,
        create_experiment,
        SessionStatus,
    )
    
    print("\n" + "="*60)
    print("EXAMPLE 7: Session Manager")
    print("="*60)
    
    manager = FLSessionManager()
    
    # Create a session
    config = create_experiment(
        name="Session-Test",
        algorithm="fedavg",
        model="cnn",
        dataset="cifar10",
        num_partitions=3,
        num_rounds=10,
    )
    
    session = manager.create_session(config)
    print(f"Created session: {session.session_id}")
    
    # Run the session
    await manager.run_session(session.session_id)
    
    # Get metrics
    metrics = manager.get_session_metrics(session.session_id)
    print(f"\nSession completed:")
    print(f"  Best accuracy: {metrics['best_accuracy']:.4f}")
    print(f"  Best round: {metrics['best_round']}")
    print(f"  Final accuracy: {metrics['accuracy_curve'][-1]:.4f}")
    
    # List all sessions
    print(f"\nAll sessions: {len(manager.list_sessions())}")
    
    return session


def example_list_algorithms():
    """List all available FL algorithms."""
    from server.fl import list_algorithms, ALGORITHM_REGISTRY
    
    print("\n" + "="*60)
    print("Available FL Algorithms")
    print("="*60)
    
    for algo in list_algorithms():
        print(f"\n{algo['name']} ({algo['id']})")
        print(f"  {algo['description']}")
        print(f"  Heterogeneous models: {'Yes' if algo['supports_heterogeneous_models'] else 'No'}")


def example_list_models():
    """List all available model architectures."""
    from server.fl import ModelRegistry
    
    print("\n" + "="*60)
    print("Available Model Architectures")
    print("="*60)
    
    for model in ModelRegistry.list_models():
        print(f"  - {model}")


async def run_all_examples():
    """Run all examples (for testing)."""
    print("\n" + "#"*60)
    print("# FL MODULE EXAMPLES")
    print("#"*60)
    
    # Non-async examples
    example_list_algorithms()
    example_list_models()
    
    # Async examples (run quick ones only for testing)
    await example_quick_test()
    await example_using_pipelines()
    
    print("\n" + "#"*60)
    print("# ALL EXAMPLES COMPLETED")
    print("#"*60)


if __name__ == "__main__":
    # Run all examples
    asyncio.run(run_all_examples())
