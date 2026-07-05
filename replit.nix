# Dependencias de sistema para Replit (canal Nix stable-24_05).
# gcc/gfortran habilitan la compilacion de PyMC/pytensor; stdenv.cc.cc.lib
# aporta libstdc++ que necesitan numba/llvmlite y pyarrow.
{ pkgs }: {
  deps = [
    pkgs.python311
    pkgs.python311Packages.pip
    pkgs.gcc
    pkgs.gfortran
    pkgs.stdenv.cc.cc.lib
    pkgs.glibcLocales
  ];
  env = {
    LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
      pkgs.stdenv.cc.cc.lib
      pkgs.zlib
    ];
  };
}
