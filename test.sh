if [ ! -d bin ] ; then mkdir bin ; fi

if [ $# -lt 2 ] ; then
    echo Please type the name of the file, without extension, to be tested
    echo or type "all" to test all .exe files.
    echo Then type the name of the test file, without extension, to be used.
    exit 1
fi

if [ "$1" == "all" ] ; then
	for F in `ls bin/*.exe` ; do 
		echo =======================================;
		echo TESTING ${F};
		${F} < $2.in > result.out;
		diff --strip-trailing-cr -q result.out $2.sol;
		if [[ $? == 0 ]] ; then echo CHECK OK ; fi
	done
	exit 1
fi

if [ ! -f bin/$1.exe ] ; then
	echo The file bin/$1.exe does not exist, compile the source code first
	exit 1
fi
bin/$1.exe < $2.in > result.out
diff --strip-trailing-cr -q result.out $2.sol
if [[ $? == 0 ]] ; then echo CHECK OK ; fi
exit 1