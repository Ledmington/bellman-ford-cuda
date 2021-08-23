if [ ! -d bin ] ; then mkdir bin ; fi

if [ $# -eq 0 ] ; then
    echo Please type the name of the file, without extension, to be tested
    echo or type "all" to test all .exe files.
    exit 1
fi

if [ "$1" == "all" ] ; then
	for F in `ls bin/*.exe` ; do 
		echo =======================================;
		echo TESTING ${F};
		${F} < test/graph-rome.in > rome.out;
		cmp rome.out test/graph-rome.sol;
	done
	exit 1
fi

if [ ! -f bin/$1.exe ] ; then
	echo The file bin/$1.exe does not exist, compile the source code first
	exit 1
fi
bin/$1.exe < test/graph-rome.in > rome.out
cmp rome.out test/graph-rome.sol
exit 1