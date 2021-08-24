if [ ! -d bin ] ; then mkdir bin ; fi

if [ $# -lt 2 ] ; then
    echo Please type the name of the file, without extension, to be compiled
    echo or type "all" to compile all .cu files.
    echo Then type the CUDA architecture you want to compile for.
    exit 1
fi

if [ "$1" == "all" ] ; then
	for F in `ls src/*.cu` ; do
		F=${F##*/}
		echo Compiling ${F}...
		if [ -f bin/${F%.*}.exe ] ; then rm bin/${F%.*}* ; fi
		nvcc -Wno-deprecated-gpu-targets -arch=$2 -rdc=true src/${F} -o bin/${F%.*}.exe ;
		if [[ $? == 0 ]] ; then echo Compilation successful ; fi
	done
	exit 1
fi

if [ -f bin/$1.exe ] ; then rm bin/$1* ; fi
nvcc -Wno-deprecated-gpu-targets -arch=$2 -rdc=true src/$1.cu -o bin/$1.exe
if [[ $? == 0 ]] ; then echo Compilation successful ; fi
exit 1