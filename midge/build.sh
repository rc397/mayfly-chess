#!/bin/sh
# build.sh: builds Midge with plain g++ or clang++ on Linux, and on Windows under MSYS2 (17 Nov 2026).
# Run it from the engine folder (the one that holds src/).
#   ./build.sh                       for any x86-64 CPU with AVX2 and BMI2 (x86-64-v3), 256 wide
#   H=512 ./build.sh                 the 512-wide net
#   H=256 KB=4 ./build.sh            the king-bucket net
#   ARCH=native ./build.sh           tuned to the machine it is built on (not for a release that moves machines)
#   ARCH=x86-64 ./build.sh           no AVX2: the plain C++ path of the network (slower)
#   CXX=clang++ ./build.sh           another compiler
#   VERSION=0.2 ./build.sh           the version shown in "id name" (default: dev)
#   DEV=1 ./build.sh                 a test build: the search test switches are listed as UCI options
#   STATIC=0 ./build.sh              Linux only: link libc from the system instead of into the file
#   OUT=name ./build.sh              the output file name inside build/ (default: midge)
#   NET=../nets/midge_v5h512.bin ./build.sh     that net is BUILT INTO the binary (it needs no net file), unless
#                                               EMBED=0; it is also copied to build/midge_net.bin, which the engine
#                                               loads by itself at start (the net must match H and KB)
# Output: build/midge (build/midge.exe on Windows) and build/perft.
# Tested: Windows (MSYS2 g++ 14.2) and Linux (Ubuntu 24.04, g++ 13.3). After a build, "build/midge bench 12"
# must print the node count the notes give for that net.
set -e
CXX=${CXX:-g++}
ARCH=${ARCH:-x86-64-v3}
H=${H:-256}
KB=${KB:-1}
OUT=${OUT:-midge}
FLAGS="-O3 -std=c++20 -Wall -DNDEBUG -march=$ARCH -DMIDGE_H=$H -DMIDGE_KB=$KB -pthread"
if [ -n "$VERSION" ]; then FLAGS="$FLAGS -DMIDGE_VERSION=\"$VERSION\""; fi
mkdir -p build
if [ -n "$NET" ] && [ "$EMBED" != "0" ]; then cp "$NET" build/embed_net.bin; FLAGS="$FLAGS -DMIDGE_EMBED"; fi
if [ "$DEV" = "1" ]; then FLAGS="$FLAGS -DMIDGE_DEV"; fi
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) FLAGS="$FLAGS -static"; EXE=.exe ;;       # one file, no MSYS DLLs needed
  *) if [ "$STATIC" = "0" ]; then FLAGS="$FLAGS -static-libstdc++ -static-libgcc"; else FLAGS="$FLAGS -static"; fi
     EXE= ;;                                                      # Linux: one file that needs nothing installed
esac
mkdir -p build
echo "$CXX $FLAGS"
$CXX $FLAGS -o build/$OUT$EXE src/main.cpp
$CXX -O3 -std=c++20 -march=$ARCH -o build/perft$EXE src/perft_main.cpp
if [ -n "$NET" ]; then cp "$NET" build/midge_net.bin; fi
echo "built build/$OUT$EXE (H=$H KB=$KB ARCH=$ARCH VERSION=${VERSION:-dev})"
