
# cpp
# https://github.com/devcontainers/images/blob/fca5ffa753e7baf6043e2731c830c65af05213e7/src/cpp/.devcontainer/Dockerfile
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install build-essential cmake cppcheck valgrind clang lldb llvm gdb \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# emscripten
# https://github.com/matt9ucci/vscode-devcontainers/blob/b280b5a2547ae2bd08ea232d73d5f24a6022f0f6/devcontainers/emscripten/Dockerfile
EMSDK=$HOME/emsdk
EMSDK_VERSION=3.1.26
git clone --depth 1 https://github.com/emscripten-core/emsdk.git $EMSDK
$EMSDK/emsdk install $EMSDK_VERSION
$EMSDK/emsdk activate $EMSDK_VERSION
echo "source $(realpath $EMSDK/emsdk_env.sh) > /dev/null 2>&1" | sudo tee -a /etc/bash.bashrc > /dev/null