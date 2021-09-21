# CUDA implementation of the Bellman-Ford algorithm
This project is the thesis work of Filippo Barbari for academic year 2020/2021.

This program is distributed under the GNU GPL v3 license. Please check the COPYING file.

## Compilation
All source files are contained inside the `src` folder. The executable files will be created in the `bin` folder.

### Compile on Windows
You can use the file `compile.bat` to compile `.cu` files. Pass the name of the file, without extension, as first parameter to compile it. Otherwise, you can type `compile.bat all` to compile all of them. Then pass as second parameter the CUDA architecture you want to compile for.
For each file, it runs `nvcc -arch=<my-architecture> -Wno-deprecated-gpu-targets src\<my-file>.cu -o bin\<my-file>.exe`.

### Compile on Linux
You can use the file `compile.sh` to compile `.cu` files. Pass the name of the file, without extension, as first parameter to compile it. Otherwise, you can type `./compile.sh all` to compile all of them. Then pass as second parameter the CUDA architecture you want to compile for.
For each file, it runs `nvcc -arch=<my-architecture> -Wno-deprecated-gpu-targets src/<my-file>.cu -o bin/<my-file>.exe`.

## Test
All test files are contained inside the `test` folder. Those files contain the graph representation of the road map of Rome (Italy) and some countries of the USA: for this reason, they are planar graphs.
You can generate a new test file using the program `graphgen.c` inside the `src` folder: check the comments inside it to learn how to use it. This program generates a random graph that is highly improbable to be planar.
The actual test file is the `.in` one, the `.sol` file is the solution.

### Testing on Windows
You can use the file `test.bat` to test `.exe` files. Pass the name of the file, without extension, as first parameter to test it. Otherwise, you can type `test.bat all` to test all of them. Then pass as second parameter the test file, without extension, you want to use.
For each file, it runs `bin\<my-file>.exe < test\<my-test-file>.in > result.out` and then checks if the result is correct with `fc result.out test\<my-test-file>.sol`.

### Testing on Linux
You can use the file `test.sh` to test `.exe` files. Pass the name of the file, without extension, as first parameter to test it. Otherwise, you can type `./test.sh all` to test all of them. Then pass as second parameter the test file, without extension, you want to use.
For each file, it runs `bin/<my-file>.exe < test/<my-test-file>.in > result.out` and then checks if the result is correct with `diff --strip-trailing-cr -q result.out test/<my-test-file>.sol`.

## Document
The main document of this project is a `tex` file inside the `doc` folder.
You can create the `.pdf` file running
```bash
biber "tesi"
pdflatex.exe -synctex=1 -interaction=nonstopmode "tesi".tex
```
