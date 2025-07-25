#!/usr/bin/env python3
"""
StockLab Dependency Fixer
=========================
Automatically installs missing dependencies for StockLab
"""

import subprocess
import sys
import importlib

def check_and_install_package(package_name, pip_name=None):
    """Check if package is installed, install if missing"""
    if pip_name is None:
        pip_name = package_name
    
    try:
        importlib.import_module(package_name)
        print(f"✅ {package_name} is already installed")
        return True
    except ImportError:
        print(f"❌ {package_name} is missing. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
            print(f"✅ {package_name} installed successfully")
            return True
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package_name}")
            return False

def main():
    """Fix common dependency issues"""
    print("🔧 StockLab Dependency Fixer")
    print("=" * 40)
    
    # Core dependencies that might be missing
    dependencies = [
        ("hmmlearn", "hmmlearn>=0.3.0"),
        ("scipy", "scipy>=1.12.0"),
        ("statsmodels", "statsmodels>=0.15.0"),
        ("streamlit", "streamlit>=1.47.0"),
        ("plotly", "plotly>=5.24.1"),
        ("yfinance", "yfinance>=0.2.64"),
        ("pandas", "pandas>=2.3.0"),
        ("numpy", "numpy>=1.26.4"),
        ("sklearn", "scikit-learn>=1.7.0"),
    ]
    
    all_installed = True
    for package, pip_name in dependencies:
        if not check_and_install_package(package, pip_name):
            all_installed = False
    
    print("\n" + "=" * 40)
    if all_installed:
        print("✅ All dependencies are now installed!")
        print("🚀 You can now run: py run_app.py")
    else:
        print("⚠️  Some dependencies failed to install.")
        print("   Please check the error messages above.")
    
    # Test import
    print("\n🧪 Testing imports...")
    try:
        from Components.TickerData import TickerData
        from Components.BackTesting import CustomBacktestingEngine
        from Components.RuleBasedAnalysis import get_analyzer
        print("✅ All StockLab components imported successfully!")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Please check the error and install missing dependencies manually.")

if __name__ == "__main__":
    main() 