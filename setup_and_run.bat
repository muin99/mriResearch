@echo off
REM Complete Setup and Execution Script for Windows
REM Installs all required libraries and runs Python scripts

echo ========================================================================
echo   COMPLETE SETUP AND EXECUTION SCRIPT
echo ========================================================================
echo.
echo This script will:
echo   1. Check Python installation
echo   2. Install all required libraries
echo   3. Run dataset exploration (optional)
echo   4. Run main training script
echo.
echo This process may take 15-30 minutes for installation
echo Training will take an additional 2-4 hours
echo.
pause

REM Check if Python is installed
echo.
echo [1/4] Checking Python installation...
python --version
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/downloads/
    pause
    exit /b 1
)
echo ✓ Python found

REM Check if pip is installed
echo.
echo [2/4] Checking pip installation...
python -m pip --version
if errorlevel 1 (
    echo ERROR: pip is not installed
    pause
    exit /b 1
)
echo ✓ pip found

REM Upgrade pip
echo.
echo [3/4] Upgrading pip...
python -m pip install --upgrade pip --quiet
echo ✓ pip upgraded

REM Install PyTorch with CUDA
echo.
echo [4/4] Installing PyTorch with CUDA support...
echo This may take 5-10 minutes...
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
if errorlevel 1 (
    echo.
    echo WARNING: CUDA installation failed, trying CPU-only version...
    python -m pip install torch torchvision torchaudio
)

REM Install all requirements
echo.
echo Installing all required libraries...
echo This may take 5-15 minutes...
if exist requirements_training.txt (
    python -m pip install -r requirements_training.txt
) else (
    echo requirements_training.txt not found, installing core packages...
    python -m pip install numpy pandas Pillow opencv-python scikit-learn scikit-image scipy matplotlib seaborn tqdm
)

REM Verify installation
echo.
echo Verifying installation...
python -c "import torch; import numpy; import pandas; import matplotlib; import seaborn; print('✓ All core packages installed')"
if errorlevel 1 (
    echo WARNING: Some packages may not be installed correctly
)

REM Run dataset exploration automatically
echo.
echo ========================================================================
echo   DATASET EXPLORATION (Automatic)
echo ========================================================================
echo Running dataset exploration automatically...
python dataset_exploration.py

REM Run training automatically
echo.
echo ========================================================================
echo   MAIN TRAINING SCRIPT (Automatic)
echo ========================================================================
echo.
echo IMPORTANT: Training will take 2-4 hours to complete!
echo Starting automatically...
echo.
echo ========================================================================
echo   STARTING TRAINING - This will take 2-4 hours
echo ========================================================================
echo Results will be saved to 'training_results' folder
echo You can minimize this window and come back later
echo.
python train_complete_model.py

echo.
echo ========================================================================
echo   SETUP COMPLETE
echo ========================================================================
echo.
echo Check 'training_results' folder for all outputs
echo.
pause

