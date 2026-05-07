import subprocess
import sys


def main():
    peclets = [0.25, 0.5, 1.0, 5.0, 10.0, 20.0]
    bcs = ["dirichlet", "periodic", "zero_flux"]
    for bc in bcs:
        for pe in peclets:
            cmd = [
                sys.executable,
                "scripts/generate_cn_reference.py",
                "--bc",
                bc,
                "--pe",
                str(pe),
                "--T",
                "10",
                "--nx",
                "1000",
                "--nt",
                "1000",
            ]
            print(" ".join(cmd))
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

