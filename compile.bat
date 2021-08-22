@echo off

if not exist bin mkdir bin
del /Q bin\*

if [%1] == [] (
    echo Please type the name of the file, without extension, to be compiled
    echo or type "all" to compile all .cu files.
    exit /b 1
)

if "%1" == "all" (
	for /f tokens^=* %%A in ('where "src:*.cu"') do (
		nvcc -Wno-deprecated-gpu-targets -arch=compute_50 %%A -o bin\%%~nA.exe
	)
	exit /b 1
)

nvcc -arch=compute_50 src\%1.cu -o bin\%1.exe
exit /b 1