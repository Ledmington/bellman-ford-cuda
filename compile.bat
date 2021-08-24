@echo off

if not exist bin mkdir bin

if [%2] == [] (
    echo Please type the name of the file, without extension, to be compiled
    echo or type "all" to compile all .cu files.
    echo Then type the CUDA architecture you want to compile for.
    exit /b 1
)

if "%1" == "all" (
	for /f tokens^=* %%A in ('where "src:*.cu"') do (
		if exist bin\%%~nA.exe del bin\%%~nA*
		nvcc -Wno-deprecated-gpu-targets -arch=%2 -rdc=true %%A -o bin\%%~nA.exe
	)
	exit /b 1
)

if exist bin\%1.exe del bin\%1*
nvcc -Wno-deprecated-gpu-targets -arch=%2 -rdc=true src\%1.cu -o bin\%1.exe
exit /b 1