{
    description = "Python development environment";
    inputs = {
        nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
        flake-utils.url = "github:numtide/flake-utils";
    };
    outputs = { self, nixpkgs, flake-utils }:
        flake-utils.lib.eachDefaultSystem (system:
            let
                pkgs = import nixpkgs {
                    inherit system;
                    config.allowUnfree = true;
                };
                
                pythonOverrides = self: super: {
                    timm = super.timm.overridePythonAttrs (old: {
                        buildInputs = [];
                        nativeBuildInputs = [];
                        propagatedBuildInputs = [ self.torch ];
                    });
                };
                
                python = pkgs.python3.override {
                    packageOverrides = pythonOverrides;
                };
            in
            {
                devShells.default = pkgs.mkShell {
                    buildInputs = [
                        (python.withPackages (ps: with ps; [
                            torchWithCuda
                            timm
                            torchvision
                            transformers
                            accelerate
                            pytest-playwright
                            playwright
                            pandas
                            pytz
                            sentencepiece
                            attrdict
                            einops
                            gradio
                            gradio-client
                            pymupdf
                            janus
                            (pkgs.python3Packages.buildPythonPackage {
                                pname = "mdtex2html";
                                version = "1.2.0";
                                src = pkgs.python3Packages.fetchPypi {
                                    pname = "mdtex2html";
                                    version = "1.3.1";
                                    sha256 = "sha256-+ktWgxE1sQD7DDd7HgXsEcL3NBoW/kCZJFhZpKQ7NCw=";
                                };
                                propagatedBuildInputs = [ pkgs.python3Packages.markdown ];
                                doCheck = false;
                            })
                            pypinyin
                            tiktoken
                            tqdm
                            colorama
                            pygments
                            markdown
                            sentencepiece
                        ]))
                        python.pkgs.black
                        pkgs.cudatoolkit
                        pkgs.cudaPackages.cudnn
                    ];
                    
                    shellHook = ''
                      export CUDA_PATH=${pkgs.cudatoolkit}
                      export CUDNN_PATH=${pkgs.cudaPackages.cudnn}
                      export LD_LIBRARY_PATH=${pkgs.cudatoolkit}/lib:${pkgs.cudaPackages.cudnn}/lib:$LD_LIBRARY_PATH
                    '';
                };
            }
        );
}

