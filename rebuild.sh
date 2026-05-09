rm -r build
mkdir build
cd build
cmake ..
make
cpack -G DEB