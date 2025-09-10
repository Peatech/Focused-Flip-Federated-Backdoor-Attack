#!/usr/bin/env python3
"""
Quick test script for FedSPECTRE-Hybrid defense.
Run this to verify the defense is working correctly.
"""

import subprocess
import sys
import time

def run_test(attack, model="simple", epochs=2):
    """Run a quick test with specified parameters."""
    print(f"\n🧪 Testing FedSPECTRE-Hybrid with {attack} attack...")
    print("=" * 60)
    
    cmd = [
        "python", "Bases.py",
        "--defense", "fedspectre",
        "--config", "cifar", 
        "--backdoor", attack,
        "--model", model
    ]
    
    try:
        # Run the test
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("✅ Test completed successfully!")
            print("\n📊 Results:")
            print(result.stdout)
        else:
            print("❌ Test failed!")
            print("Error output:")
            print(result.stderr)
            
    except subprocess.TimeoutExpired:
        print("⏰ Test timed out after 2 minutes")
    except Exception as e:
        print(f"💥 Test error: {e}")

def main():
    """Main test function."""
    print("🚀 FedSPECTRE-Hybrid Defense Quick Test")
    print("=" * 60)
    
    # Test with FF attack (CPU-friendly)
    run_test("ff")
    
    # Test with baseline (no attack)
    run_test("baseline")
    
    print("\n🎉 Quick test completed!")
    print("\nFor full testing with GPU, run:")
    print("python Bases.py --defense fedspectre --config cifar --backdoor neurotoxin --model simple")

if __name__ == "__main__":
    main()
