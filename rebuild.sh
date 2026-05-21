rm -rf build
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=/usr/local
cmake --build build -j$(nproc)
rm -rf _CPack_Packages
cd build || exit 1
cpack --config CPackConfig.cmake -G DEB
