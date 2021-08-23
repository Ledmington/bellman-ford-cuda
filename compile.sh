if [ ! -d bin ] ; then mkdir bin ; fi

if [ $# -eq 0 ] ; then
    echo Please type the name of the file, without extension, to be compiled
    echo or type "all" to compile all .cu files.
    exit 1
fi

if [ "$1" == "all" ] ; then
	for F in `ls src/*.cu` ; do
		F=${F##*/}
		if [ -f bin/${F%.*}.exe ] ; then rm bin/${F%.*}* ; fi
		nvcc -Wno-deprecated-gpu-targets -arch=compute_50 src/${F} -o bin/${F%.*}.exe ;
	done
	exit 1
fi

if [ -f bin/$1.exe ] ; then rm bin/$1* ; fi
nvcc -Wno-deprecated-gpu-targets -arch=compute_50 src/$1.cu -o bin/$1.exe
exit 1