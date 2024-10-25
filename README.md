# CUDA implementation of the Bellman-Ford algorithm
This project is the thesis work of Filippo Barbari for academic year 2020/2021.

This is the modern version of the project (work in progress), for the original code, take a look at the [`old` branch](https://github.com/Ledmington/bellman-ford-cuda/tree/old).

## Compilation
In order to compile this project, you need:
 - a CUDA-capable GPU
 - CMake
 - a working C99 compiler
 - a working CUDA compiler

```bash
cmake -S . -B build
cmake --build build
```

## Testing
After compilation, you can run the test suite to check that everything is set up correctly.

```bash
cd build
ctest
```

## Document
The main document of this project is a `tex` file inside the `doc` folder.
You can create the `.pdf` file running
```bash
biber "tesi"
pdflatex -synctex=1 -interaction=nonstopmode "tesi".tex
```

## License
This program is distributed under the GNU GPL v3 license. Please check the COPYING file.
