rm -rf build
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=/usr/local
cmake --build build -j$(nproc)
cpack --config build/CPackConfig.cmake -G DEB
