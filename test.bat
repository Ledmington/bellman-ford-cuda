@echo off

if not exist bin mkdir bin

if [%1] == [] (
    echo Please type the name of the file, without extension, to be tested
    echo or type "all" to test all .exe files.
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

if not exist bin\%1.exe (
	echo The file bin\%1.exe does not exist, compile the source code first
	exit /b 1
)
bin\%1.exe < test\graph-rome.in > rome.out
fc rome.out test\graph-rome.sol
exit /b 1