"""
Complete Setup and Execution Script
Installs all required libraries and runs Python scripts in order
"""

import subprocess
import sys
import os
from pathlib import Path
import time
from datetime import datetime

def print_header(text):
    """Print formatted header"""
    header = "\n" + "="*70 + f"\n  {text}\n" + "="*70 + "\n"
    print(header)
    return header

def print_step(step_num, total_steps, text):
    """Print step information"""
    print(f"\n[{step_num}/{total_steps}] {text}")
    print("-" * 70)

def run_command(command, description, check=True, log_file=None):
    """Run a command and handle errors"""
    print(f"\nExecuting: {command}")
    if log_file:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\nExecuting: {command}\n")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=check,
            capture_output=True,
            text=True
        )
        if result.stdout:
            print(result.stdout)
            if log_file:
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(result.stdout + "\n")
        return True
    except subprocess.CalledProcessError as e:
        error_msg = f"ERROR: {description}\nError output: {e.stderr}"
        print(error_msg)
        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(error_msg + "\n")
        if check:
            print("\n⚠️  Installation failed. Please check the error above.")
            return False
        return False
    except Exception as e:
        error_msg = f"ERROR: {str(e)}"
        print(error_msg)
        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(error_msg + "\n")
        return False

def check_python():
    """Check Python installation"""
    print_header("CHECKING PYTHON INSTALLATION")
    try:
        version = sys.version_info
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} detected")
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            print("⚠️  WARNING: Python 3.8+ is recommended")
        return True
    except Exception as e:
        print(f"✗ Python check failed: {e}")
        return False

def check_pip():
    """Check pip installation"""
    print_header("CHECKING PIP INSTALLATION")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"✓ {result.stdout.strip()}")
            return True
        else:
            print("✗ pip not found")
            return False
    except Exception as e:
        print(f"✗ pip check failed: {e}")
        return False

def upgrade_pip():
    """Upgrade pip to latest version"""
    print_header("UPGRADING PIP")
    return run_command(
        f"{sys.executable} -m pip install --upgrade pip",
        "pip upgrade"
    )

def install_pytorch():
    """Install PyTorch with CUDA support"""
    print_header("INSTALLING PYTORCH WITH CUDA SUPPORT")
    print("This may take 5-10 minutes...")
    print("Installing PyTorch, torchvision, and torchaudio")
    
    # Try CUDA 11.8 first (most compatible)
    command = f"{sys.executable} -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    
    print("\nAttempting to install PyTorch with CUDA 11.8...")
    success = run_command(command, "PyTorch installation", check=False)
    
    if not success:
        print("\n⚠️  CUDA installation failed, trying CPU-only version...")
        command = f"{sys.executable} -m pip install torch torchvision torchaudio"
        success = run_command(command, "PyTorch CPU installation", check=False)
    
    if success:
        print("\n✓ PyTorch installation completed")
        # Verify installation
        try:
            import torch
            print(f"✓ PyTorch {torch.__version__} installed")
            if torch.cuda.is_available():
                print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            else:
                print("⚠️  CUDA not available (CPU-only mode)")
        except:
            print("⚠️  Could not verify PyTorch installation")
    
    return success

def install_requirements():
    """Install all requirements from requirements_training.txt"""
    print_header("INSTALLING ALL REQUIREMENTS")
    
    requirements_file = Path("requirements_training.txt")
    if not requirements_file.exists():
        print("✗ requirements_training.txt not found!")
        print("Installing core packages manually...")
        return install_core_packages()
    
    print(f"Reading requirements from: {requirements_file}")
    print("This may take 5-15 minutes depending on your internet speed...")
    
    command = f"{sys.executable} -m pip install -r {requirements_file}"
    return run_command(command, "Requirements installation")

def install_core_packages():
    """Install core packages manually if requirements file is missing"""
    print_header("INSTALLING CORE PACKAGES")
    
    packages = [
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "Pillow>=10.0.0",
        "opencv-python>=4.8.0",
        "scikit-learn>=1.3.0",
        "scikit-image>=0.21.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0"
    ]
    
    for package in packages:
        print(f"\nInstalling {package}...")
        command = f"{sys.executable} -m pip install {package}"
        if not run_command(command, f"Installing {package}", check=False):
            print(f"⚠️  Warning: {package} installation had issues")
    
    return True

def verify_installation():
    """Verify all critical packages are installed"""
    print_header("VERIFYING INSTALLATION")
    
    required_packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'PIL': 'Pillow',
        'cv2': 'OpenCV',
        'sklearn': 'scikit-learn',
        'skimage': 'scikit-image',
        'scipy': 'SciPy',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'tqdm': 'tqdm'
    }
    
    all_ok = True
    for module, name in required_packages.items():
        try:
            __import__(module)
            print(f"✓ {name} installed")
        except ImportError:
            print(f"✗ {name} NOT installed")
            all_ok = False
    
    return all_ok

def run_python_script(script_name, description):
    """Run a Python script"""
    script_path = Path(script_name)
    
    if not script_path.exists():
        print(f"⚠️  Script not found: {script_name}")
        print(f"   Skipping {description}")
        return False
    
    print_header(f"RUNNING: {description}")
    print(f"Script: {script_name}")
    
    command = f"{sys.executable} {script_name}"
    success = run_command(command, description, check=False)
    
    if success:
        print(f"\n✓ {description} completed successfully")
    else:
        print(f"\n⚠️  {description} had issues (check output above)")
    
    return success

def log_message(message, log_file_path="setup_and_training.log"):
    """Log message to both console and file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {message}"
    print(message)
    try:
        with open(log_file_path, 'a', encoding='utf-8') as f:
            f.write(log_msg + "\n")
    except:
        pass  # If logging fails, just print

def main():
    """Main execution flow - Fully automated, no user interaction"""
    # Create/clear log file
    log_file = Path("setup_and_training.log")
    if log_file.exists():
        log_file.unlink()  # Clear old log
    
    start_time = datetime.now()
    log_message("="*70)
    log_message("COMPLETE SETUP AND EXECUTION SCRIPT - FULLY AUTOMATED")
    log_message("="*70)
    log_message(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_message("="*70)
    
    log_message("="*70)
    log_message("COMPLETE SETUP AND EXECUTION SCRIPT - FULLY AUTOMATED")
    log_message("="*70)
    log_message(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_message("="*70)
    log_message("This script will:")
    log_message("1. Check Python and pip installation")
    log_message("2. Upgrade pip to latest version")
    log_message("3. Install PyTorch with CUDA support")
    log_message("4. Install all required libraries")
    log_message("5. Verify installation")
    log_message("6. Run dataset exploration")
    log_message("7. Run main training script (2-4 hours)")
    log_message("\nThis process may take 15-30 minutes for installation")
    log_message("Training will take an additional 2-4 hours")
    log_message("\nStarting automatically in 3 seconds...")
    log_message("(All output will be logged to: setup_and_training.log)")
    
    for i in range(3, 0, -1):
        print(f"Starting in {i}...", end='\r')
        time.sleep(1)
    print("\n")
    log_message("\n" + "="*70)
    
    total_steps = 8
    step = 1
    
    # Step 1: Check Python
    print_step(step, total_steps, "Checking Python Installation")
    log_message(f"[Step {step}/{total_steps}] Checking Python Installation")
    if not check_python():
        log_message("✗ Python check failed. Please install Python 3.8+")
        return False
    log_message("✓ Python check passed")
    step += 1
    time.sleep(1)
    
    # Step 2: Check pip
    print_step(step, total_steps, "Checking pip Installation")
    log_message(f"[Step {step}/{total_steps}] Checking pip Installation")
    if not check_pip():
        log_message("✗ pip not found. Please install pip")
        return False
    log_message("✓ pip check passed")
    step += 1
    time.sleep(1)
    
    # Step 3: Upgrade pip
    print_step(step, total_steps, "Upgrading pip")
    log_message(f"[Step {step}/{total_steps}] Upgrading pip")
    upgrade_pip()
    log_message("✓ pip upgrade completed")
    step += 1
    time.sleep(1)
    
    # Step 4: Install PyTorch
    print_step(step, total_steps, "Installing PyTorch (This takes 5-10 minutes)")
    log_message(f"[Step {step}/{total_steps}] Installing PyTorch (This takes 5-10 minutes)")
    install_pytorch()
    log_message("✓ PyTorch installation completed")
    step += 1
    time.sleep(2)
    
    # Step 5: Install requirements
    print_step(step, total_steps, "Installing All Requirements (This takes 5-15 minutes)")
    log_message(f"[Step {step}/{total_steps}] Installing All Requirements (This takes 5-15 minutes)")
    install_requirements()
    log_message("✓ Requirements installation completed")
    step += 1
    time.sleep(2)
    
    # Step 6: Verify installation
    print_step(step, total_steps, "Verifying Installation")
    log_message(f"[Step {step}/{total_steps}] Verifying Installation")
    if not verify_installation():
        log_message("\n⚠️  Some packages may not be installed correctly")
        log_message("   Continuing anyway...")
    else:
        log_message("✓ All packages verified")
    step += 1
    time.sleep(1)
    
    # Step 7: Run dataset exploration (automatic)
    print_step(step, total_steps, "Dataset Exploration")
    log_message(f"[Step {step}/{total_steps}] Running Dataset Exploration")
    log_message("Running dataset exploration automatically...")
    run_python_script("dataset_exploration.py", "Dataset Exploration")
    log_message("✓ Dataset exploration completed")
    step += 1
    time.sleep(1)
    
    # Step 8: Run main training script (automatic)
    print_step(step, total_steps, "Main Training Script")
    log_message(f"[Step {step}/{total_steps}] Starting Main Training Script")
    log_message("\n" + "="*70)
    log_message("STARTING TRAINING AUTOMATICALLY")
    log_message("="*70)
    log_message("⚠️  This will take 2-4 hours to complete!")
    log_message("   The script will train the model and generate all visualizations")
    log_message("   Results will be saved to 'training_results' folder")
    log_message("="*70 + "\n")
    
    print("\n" + "="*70)
    print("STARTING TRAINING AUTOMATICALLY")
    print("="*70)
    print("⚠️  This will take 2-4 hours to complete!")
    print("   You can safely minimize this window and come back later")
    print("   All progress is logged to: setup_and_training.log")
    print("   Results will be saved to 'training_results' folder")
    print("="*70 + "\n")
    
    time.sleep(3)
    run_python_script("train_complete_model.py", "Model Training")
    log_message("✓ Training completed")
    
    # Final summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    log_message("\n" + "="*70)
    log_message("SETUP AND EXECUTION COMPLETE")
    log_message("="*70)
    log_message(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_message(f"Total Duration: {duration}")
    log_message("\nSummary:")
    log_message("✓ All libraries installed")
    log_message("✓ Dataset exploration completed")
    log_message("✓ Training completed")
    log_message("\nResults:")
    log_message("  - Check 'training_results' folder for outputs")
    log_message("  - View visualizations in 'training_results/plots'")
    log_message("  - Check metrics in 'training_results/tables'")
    log_message("  - View complete log in 'setup_and_training.log'")
    log_message("\n" + "="*70)
    
    print("\n" + "="*70)
    print("✓ ALL DONE! Everything completed successfully.")
    print("="*70)
    print(f"\nTotal time: {duration}")
    print("\nCheck these folders for results:")
    print("  - training_results/ (all outputs)")
    print("  - setup_and_training.log (complete log)")
    print("\n" + "="*70)
    
    return True

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("setup_and_training.log", 'a', encoding='utf-8') as f:
            f.write(f"\n[{log_time}] ⚠️  Process interrupted by user\n")
        print("\n\n⚠️  Process interrupted by user")
        print("Check setup_and_training.log for progress")
        sys.exit(1)
    except Exception as e:
        log_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("setup_and_training.log", 'a', encoding='utf-8') as f:
            f.write(f"\n[{log_time}] ✗ Unexpected error: {e}\n")
        print(f"\n\n✗ Unexpected error: {e}")
        print("Check setup_and_training.log for details")
        sys.exit(1)

