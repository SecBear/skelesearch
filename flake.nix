{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    crane.url = "github:ipetkov/crane";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      crane,
      flake-utils,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };
        # crane v2: mkLib replaced the old crane.lib.${system} attribute
        craneLib = crane.mkLib pkgs;
        src = craneLib.cleanCargoSource ./.;
        # We only consume the shared runtime library, not the Python wheel.
        baseOnnxruntime = pkgs.onnxruntime.override {
          pythonSupport = false;
        };
        # nixpkgs builds onnxruntime with FETCHCONTENT_FULLY_DISCONNECTED, so
        # enabling CoreML requires vendoring the extra dependency trees that
        # upstream CMake would otherwise fetch dynamically.
        coremltoolsSource =
          pkgs.applyPatches
            {
              name = "coremltools-7.1-patched";
              src = pkgs.fetchzip {
                url = "https://github.com/apple/coremltools/archive/refs/tags/7.1.zip";
                hash = "sha256-kajQFHpl+4UK6fp+rM8TP0GiqIFYXPVFc2x1p19rBSw=";
              };
              patches = [
                "${pkgs.onnxruntime.src}/cmake/patches/coremltools/crossplatformbuild.patch"
              ];
            };
        fp16Source =
          pkgs.applyPatches
            {
              name = "fp16-cmake4-compatible";
              src = pkgs.fetchzip {
                url = "https://github.com/Maratyszcza/FP16/archive/0a92994d729ff76a58f692d3028ca1b64b145d91.zip";
                hash = "sha256-m2d9bqZoGWzuUPGkd29MsrdscnJRtuIkLIMp3fMmtRY=";
              };
              postPatch = ''
                substituteInPlace CMakeLists.txt \
                  --replace-fail \
                    'CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12 FATAL_ERROR)' \
                    'CMAKE_MINIMUM_REQUIRED(VERSION 3.5 FATAL_ERROR)'
              '';
            };
        psimdSource =
          pkgs.applyPatches
            {
              name = "psimd-cmake4-compatible";
              src = pkgs.fetchzip {
                url = "https://github.com/Maratyszcza/psimd/archive/072586a71b55b7f8c584153d223e95687148a900.zip";
                hash = "sha256-lV+VZi2b4SQlRYrhKx9Dxc6HlDEFz3newvcBjTekupo=";
              };
              postPatch = ''
                substituteInPlace CMakeLists.txt \
                  --replace-fail \
                    'CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12 FATAL_ERROR)' \
                    'CMAKE_MINIMUM_REQUIRED(VERSION 3.5 FATAL_ERROR)'
              '';
            };
        onnxruntimePackage =
          if pkgs.stdenv.isDarwin then
            baseOnnxruntime.overrideAttrs (old: {
              cmakeFlags = (old.cmakeFlags or [ ]) ++ [
                (pkgs.lib.cmakeBool "onnxruntime_USE_COREML" true)
                (pkgs.lib.cmakeFeature "FETCHCONTENT_SOURCE_DIR_COREMLTOOLS" "${coremltoolsSource}")
                (pkgs.lib.cmakeFeature "FETCHCONTENT_SOURCE_DIR_FP16" "${fp16Source}")
                (pkgs.lib.cmakeFeature "FETCHCONTENT_SOURCE_DIR_PSIMD" "${psimdSource}")
              ];
              # ONNX Runtime's upstream test suite is not needed for the local
              # embedding dependency and materially increases build time/heat.
              doCheck = false;
            })
          else
            baseOnnxruntime;
        darwinRustRpathFlags = pkgs.lib.optionalString pkgs.stdenv.isDarwin
          "-C link-arg=-Wl,-rpath,${pkgs.lib.getLib onnxruntimePackage}/lib";

        # cmake/pkg-config are needed by ort; onnxruntime satisfies ort-sys
        # without network downloads so Nix builds stay reproducible/offline.
        commonArgs = {
          inherit src;
          nativeBuildInputs = [
            pkgs.cmake
            pkgs.pkg-config
            pkgs.protobuf # lance-encoding protos require protoc
          ];
          buildInputs = [
            onnxruntimePackage
          ];
          ORT_STRATEGY = "system";
          ORT_LIB_LOCATION = "${pkgs.lib.getLib onnxruntimePackage}/lib";
          ORT_PREFER_DYNAMIC_LINK = "1";
          RUSTFLAGS = darwinRustRpathFlags;
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            (pkgs.lib.getLib onnxruntimePackage)
          ];
          DYLD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            (pkgs.lib.getLib onnxruntimePackage)
          ];
        };
      in
      {
        packages = {
          # Nix packages build without storage feature flags — CompositeBackend is always on.
          skelesearch-cli = craneLib.buildPackage (
            commonArgs
            // {
              pname = "skelesearch-cli";
              cargoExtraArgs = "-p skelesearch-cli";
            }
          );

          skelesearch-mcp = craneLib.buildPackage (
            commonArgs
            // {
              pname = "skelesearch-mcp";
              cargoExtraArgs = "-p skelesearch-mcp";
            }
          );

          skelesearch-daemon = craneLib.buildPackage (
            commonArgs
            // {
              pname = "skelesearch-daemon";
              cargoExtraArgs = "-p skelesearch-daemon";
            }
          );

          onnxruntime-lib = pkgs.lib.getLib onnxruntimePackage;

          # Alias expected by Claude plugin hook: hooks/session-start calls
          # `nix run .#mcp-server`.
          mcp-server = self.packages.${system}.skelesearch-mcp;

          default = self.packages.${system}.skelesearch-cli;
        };

        devShells.default = pkgs.mkShell {
          # clang provides C++ runtime needed by lance; cmake/pkg-config mirror
          # the release nativeBuildInputs so `cargo build` works locally.
          nativeBuildInputs = [
            pkgs.cmake
            pkgs.pkg-config
            pkgs.protobuf # lance-encoding protos require protoc
          ];
          buildInputs = [
            pkgs.rustc
            pkgs.cargo
            pkgs.clippy
            pkgs.rustfmt
            pkgs.clang
            # libc++ runtime/headers for Darwin links that pass -lc++
            pkgs.llvmPackages.libcxx
            # bindgen needs libclang for C/C++ build scripts
            pkgs.llvmPackages.libclang.lib
            # Native TLS (reqwest -> openssl-sys)
            pkgs.openssl
            # ONNX Runtime for fastembed / local rerankers without network downloads
            onnxruntimePackage
            # Benchmark scripts (TypeScript)
            pkgs.bun
            # ContextBench adapter (Python); uv also provides hf CLI
            # for model downloads: uv tool run --from huggingface_hub hf download ...
            pkgs.python312
            pkgs.uv
            # Repo cloning for benchmarks
            pkgs.git
          ];
          # numpy and other Python C-extensions need libstdc++.so.6 on NixOS.
          # stdenv.cc.cc.lib provides it; LD_LIBRARY_PATH makes it discoverable.
          LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
          ORT_STRATEGY = "system";
          ORT_LIB_LOCATION = "${pkgs.lib.getLib onnxruntimePackage}/lib";
          ORT_PREFER_DYNAMIC_LINK = "1";
          RUSTFLAGS = darwinRustRpathFlags;
          # Force clang for C++ deps (lance, ort) on NixOS where g++ lacks system headers
          CXX = "clang++";
          CC = "clang";
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            pkgs.stdenv.cc.cc.lib
            (pkgs.lib.getLib onnxruntimePackage)
          ];
          DYLD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            (pkgs.lib.getLib onnxruntimePackage)
          ];
        };
      }
    );
}
