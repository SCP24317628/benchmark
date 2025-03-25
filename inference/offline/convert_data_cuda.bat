@echo off
setlocal EnableDelayedExpansion

:: Default values for CUDA/TRT-LLM environment
set "DEFAULT_DEVICE=cuda"
set "DEFAULT_GPU_NAME=A100"
set "DEFAULT_BACKEND=cuda"
set "DEFAULT_BACKEND_VERSION=12.8"
set "DEFAULT_ENGINE=trt-llm"
set "DEFAULT_ENGINE_VERSION=0.17.0.post"
set "DEFAULT_DRIVER=NVIDIA_Linux_x86_64"
set "DEFAULT_DRIVER_VERSION=535.161.08"

:: Initialize variables
set "INPUT_FILE="
set "OUTPUT_FILE="
set "MODEL_NAME="

:: Parse command line arguments
:parse_args
if "%~1"=="" goto :validate_args
if /i "%~1"=="-i" (
    set "INPUT_FILE=%~2"
    shift & shift
    goto :parse_args
)
if /i "%~1"=="--input" (
    set "INPUT_FILE=%~2"
    shift & shift
    goto :parse_args
)
if /i "%~1"=="-m" (
    set "MODEL_NAME=%~2"
    shift & shift
    goto :parse_args
)
if /i "%~1"=="--model" (
    set "MODEL_NAME=%~2"
    shift & shift
    goto :parse_args
)
if /i "%~1"=="-o" (
    set "OUTPUT_FILE=%~2"
    shift & shift
    goto :parse_args
)
if /i "%~1"=="--output" (
    set "OUTPUT_FILE=%~2"
    shift & shift
    goto :parse_args
)
if /i "%~1"=="-h" (
    goto :usage
)
if /i "%~1"=="--help" (
    goto :usage
)
echo Unknown argument: %~1
goto :usage

:validate_args
:: Check if input file is provided
if "%INPUT_FILE%"=="" (
    echo Error: Input file is required
    goto :usage
)

:: Check if model name is provided
if "%MODEL_NAME%"=="" (
    echo Error: Model name is required
    goto :usage
)

:: Check if input file exists
if not exist "%INPUT_FILE%" (
    echo Error: Input file '%INPUT_FILE%' does not exist
    exit /b 1
)

:: Construct python command
set "CMD=python convert_data.py --input "%INPUT_FILE%" --device %DEFAULT_DEVICE% --gpu-name "%DEFAULT_GPU_NAME%" --model-name "%MODEL_NAME%" --backend %DEFAULT_BACKEND% --backend-version %DEFAULT_BACKEND_VERSION% --engine %DEFAULT_ENGINE% --engine-version "%DEFAULT_ENGINE_VERSION%" --driver "%DEFAULT_DRIVER%" --driver-version "%DEFAULT_DRIVER_VERSION%""

:: Add output file if specified
if not "%OUTPUT_FILE%"=="" (
    set "CMD=!CMD! --output "!OUTPUT_FILE!""
)

:: Execute the command
echo Running conversion...
%CMD%

if errorlevel 1 (
    echo Error: Conversion failed
    exit /b 1
) else (
    echo Conversion completed successfully
)
exit /b 0

:usage
echo Usage: %~n0 -i ^<input_csv^> -m ^<model_name^> [-o ^<output_json^>]
echo.
echo Convert benchmark data with default settings for CUDA/TRT-LLM
echo.
echo Required arguments:
echo   -i, --input      Input CSV file path
echo   -m, --model      Model name (e.g., 'Llama-2-7b', 'Llama-2-13b')
echo.
echo Optional arguments:
echo   -o, --output     Output JSON file path
echo                    (default: results/model_name-gpu_name-device-timestamp.json)
echo   -h, --help       Show this help message
echo.
echo Default values used:
echo   Device: %DEFAULT_DEVICE%
echo   GPU: %DEFAULT_GPU_NAME%
echo   Backend: %DEFAULT_BACKEND%
echo   Backend Version: %DEFAULT_BACKEND_VERSION%
echo   Engine: %DEFAULT_ENGINE%
echo   Engine Version: %DEFAULT_ENGINE_VERSION%
echo   Driver: %DEFAULT_DRIVER%
echo   Driver Version: %DEFAULT_DRIVER_VERSION%
exit /b 1
