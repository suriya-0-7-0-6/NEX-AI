import subprocess, sys, torch

def pip_install(pkgs):
    print(f"📦 Installing: {' '.join(pkgs)}")
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + pkgs)

def install_torch():
    if not torch.cuda.is_available():
        print("⚠️ No GPU detected → installing CPU-only PyTorch")
        pip_install(["torch", "torchvision", "torchaudio",
                     "--index-url", "https://download.pytorch.org/whl/cpu"])
        return

    name = torch.cuda.get_device_name(0)
    cc_major, cc_minor = torch.cuda.get_device_capability(0)
    cc = float(f"{cc_major}.{cc_minor}")
    print(f"🔍 GPU detected: {name}, Compute Capability {cc}")

    # === Compatibility Map ===
    if cc < 5.0:
        print("❌ Compute Capability < 5.0 is too old for modern PyTorch")
        sys.exit(1)

    elif 5.0 <= cc < 7.0:
        print("➡️ Maxwell/Pascal GPU detected → using last supported PyTorch (1.13.1 + CUDA 11.7)")
        pip_install([
            "torch==1.13.1+cu117",
            "torchvision==0.14.1+cu117",
            "torchaudio==0.13.1",
            "--extra-index-url", "https://download.pytorch.org/whl/cu117"
        ])

    elif 7.0 <= cc < 8.0:
        print("➡️ Volta/Turing GPU detected → safe with CUDA 11.8 builds")
        pip_install([
            "torch==2.0.1+cu118",
            "torchvision==0.15.2+cu118",
            "torchaudio==2.0.2",
            "--extra-index-url", "https://download.pytorch.org/whl/cu118"
        ])

    elif 8.0 <= cc < 9.0:
        print("➡️ Ampere GPU detected → use CUDA 12.1+ wheels")
        pip_install([
            "torch==2.2.2+cu121",
            "torchvision==0.17.2+cu121",
            "torchaudio==2.2.2",
            "--extra-index-url", "https://download.pytorch.org/whl/cu121"
        ])

    elif 9.0 <= cc < 10.0:
        print("➡️ Hopper GPU detected → use CUDA 12.4+ wheels")
        pip_install([
            "torch==2.3.0+cu124",
            "torchvision==0.18.0+cu124",
            "torchaudio==2.3.0",
            "--extra-index-url", "https://download.pytorch.org/whl/cu124"
        ])

    else:
        print("➡️ Unknown future GPU, defaulting to latest CUDA 12.6 wheels")
        pip_install([
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu126"
        ])

    print("✅ PyTorch installed successfully")

if __name__ == "__main__":
    install_torch()