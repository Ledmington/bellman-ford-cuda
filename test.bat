@echo off

if not exist bin mkdir bin

if [%1] == [] (
    echo Please type the name of the file, without extension, to be tested
    echo or type "all" to compile all .cu files.
    exit /b 1
)

if "%1" == "all" (
	for /f tokens^=* %%A in ('where "bin:*.exe"') do (
		echo =======================================
		echo TESTING %%~nA
		bin\%%~nA.exe < test\graph-rome.in > rome.out
		fc rome.out test\graph-rome.sol
	)
	exit /b 1
)

bin\%1.exe < test\graph-rome.in > rome.out
fc rome.out test\graph-rome.sol
exit /b 1