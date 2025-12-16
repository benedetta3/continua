from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import subprocess
import numpy as np
import glob
import os

gruppo='gruppo6'

class CustomBuildExt(build_ext):
    def run(self):
        # Compila file NASM prima di build C
        for arch in ['32', '64', '64omp']:
            folder = f"src/{arch}"
            nasm_files = glob.glob(os.path.join(folder, "*.nasm"))
            for nasm_file in nasm_files:
                subprocess.run([
                    'nasm',
                    '-f', 'elf64',
                    '-DPIC',
                    '-I', folder,
                    nasm_file
                ], check=True)

        # Aggiunge i file .o dinamicamente
        for ext in self.extensions:
            if '32' in ext.name:
                ext.extra_objects = glob.glob('src/32/*.o')
            elif '64omp' in ext.name:
                ext.extra_objects = glob.glob('src/64omp/*.o')
            elif '64' in ext.name:
                ext.extra_objects = glob.glob('src/64/*.o')

        super().run()

# ---- FLAGS (minime e sicure) ----
# Se vuoi usare -march=native solo sul tuo PC:
#   export NATIVE=1
use_native = os.environ.get("NATIVE", "0") == "1"

base_cflags = ['-O3', '-DNDEBUG', '-fPIC']
if use_native:
    base_cflags += ['-march=native']  # opzionale, non metterlo se temi differenze sulla macchina del prof

module32 = Extension(
    f"{gruppo}.quantpivot32._quantpivot32",
    sources=['src/32/quantpivot32_py.c'],
    include_dirs=[np.get_include()],
    extra_compile_args=base_cflags + ['-msse'],
    extra_link_args=['-z', 'noexecstack']
)

module64 = Extension(
    f"{gruppo}.quantpivot64._quantpivot64",
    sources=['src/64/quantpivot64_py.c'],
    include_dirs=[np.get_include()],
    extra_compile_args=base_cflags + ['-msse', '-mavx'],
    extra_link_args=['-z', 'noexecstack']
)

module64omp = Extension(
    f"{gruppo}.quantpivot64omp._quantpivot64omp",
    sources=['src/64omp/quantpivot64omp_py.c'],
    include_dirs=[np.get_include()],
    extra_compile_args=base_cflags + ['-msse', '-mavx', '-fopenmp'],
    extra_link_args=['-z', 'noexecstack', '-fopenmp']
)

setup(
    name=gruppo,
    version='1.0',
    author="LISTA COMPONENTI GRUPPO",
    packages=find_packages(),
    ext_modules=[module32, module64, module64omp],
    cmdclass={'build_ext': CustomBuildExt},
    install_requires=['numpy'],
    zip_safe=False
)