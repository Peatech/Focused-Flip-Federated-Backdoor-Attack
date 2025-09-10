#!/usr/bin/env python3
"""
Comprehensive experiment runner for federated learning with backdoor attacks and defenses.
"""

import subprocess
import sys
import time
import argparse
from datetime import datetime

def run_experiment(defense, attack, model="simple", config="cifar", epochs=10):
    """Run a single experiment."""
    print(f"\n🧪 Running Experiment: {defense} vs {attack}")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Defense: {defense}")
    print(f"Attack: {attack}")
    print(f"Model: {model}")
    print(f"Config: {config}")
    print(f"Epochs: {epochs}")
    print("=" * 60)
    
    cmd = [
        "python", "Bases.py",
        "--defense", defense,
        "--config", config,
        "--backdoor", attack,
        "--model", model
    ]
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ Experiment completed successfully in {duration:.1f}s")
            
            # Extract key metrics from output
            lines = result.stdout.split('\n')
            for line in lines:
                if 'FedSPECTRE-Hybrid metrics:' in line or 'FedAvgCKA metrics:' in line:
                    print(f"📊 {line}")
                elif 'Selected:' in line or 'Excluded:' in line:
                    print(f"📊 {line}")
                elif 'backdoor:False metric:' in line or 'backdoor:True metric:' in line:
                    print(f"📊 {line}")
            
            return True, duration
        else:
            print(f"❌ Experiment failed after {duration:.1f}s")
            print("Error output:")
            print(result.stderr[-500:])  # Last 500 chars of error
            return False, duration
            
    except subprocess.TimeoutExpired:
        print("⏰ Experiment timed out after 10 minutes")
        return False, 600
    except Exception as e:
        print(f"💥 Experiment error: {e}")
        return False, 0

def main():
    """Main experiment runner."""
    parser = argparse.ArgumentParser(description='Run federated learning experiments')
    parser.add_argument('--defense', choices=['fedspectre', 'fedavgcka', 'none'], 
                       default='fedspectre', help='Defense method')
    parser.add_argument('--attack', choices=['ff', 'neurotoxin', 'dba', 'naive', 'baseline'],
                       default='ff', help='Attack method')
    parser.add_argument('--model', choices=['simple', 'resnet18'], 
                       default='simple', help='Model architecture')
    parser.add_argument('--config', choices=['cifar', 'imagenet'], 
                       default='cifar', help='Dataset configuration')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--all', action='store_true', help='Run all attack/defense combinations')
    
    args = parser.parse_args()
    
    print("🚀 Federated Learning Experiment Runner")
    print("=" * 60)
    
    if args.all:
        # Run comprehensive test suite
        defenses = ['fedspectre', 'fedavgcka', 'none']
        attacks = ['ff', 'neurotoxin', 'baseline']
        
        results = []
        
        for defense in defenses:
            for attack in attacks:
                success, duration = run_experiment(defense, attack, args.model, args.config, args.epochs)
                results.append((defense, attack, success, duration))
                time.sleep(2)  # Brief pause between experiments
        
        # Summary
        print("\n📊 Experiment Summary")
        print("=" * 60)
        for defense, attack, success, duration in results:
            status = "✅" if success else "❌"
            print(f"{status} {defense:12} vs {attack:10} - {duration:6.1f}s")
            
    else:
        # Run single experiment
        defense = args.defense if args.defense != 'none' else ''
        success, duration = run_experiment(defense, args.attack, args.model, args.config, args.epochs)
        
        if success:
            print(f"\n🎉 Experiment completed successfully!")
        else:
            print(f"\n💥 Experiment failed!")
            sys.exit(1)

if __name__ == "__main__":
    main()
