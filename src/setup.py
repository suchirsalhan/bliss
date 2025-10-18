import subprocess
import sys
from pathlib import Path

def check_python_version():
    if sys.version_info < (3, 7):
        print("❌ Python 3.7+ required")
        return False
    print(f"✅ Python {sys.version.split()[0]}")
    return True

def install_dependencies():
    packages = ['spacy', 'transformers', 'lemminflect']

    for package in packages:
        try:
            __import__(package)
            print(f"✅ {package} already installed")
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✅ {package} installed")

def download_spacy_model():
    try:
        import spacy
        spacy.load("en_core_web_sm")
        print("✅ spaCy English model already available")
    except OSError:
        print("Downloading spaCy English model...")
        subprocess.check_call([sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'])
        print("✅ spaCy English model downloaded")

def verify_toolkit():
    print("\nVerifying toolkit...")

    # check imports
    try:
        import convert_writeandimprove
        print("✅ convert_writeandimprove module")
    except ImportError as e:
        print(f"❌ convert_writeandimprove: {e}")
        return False

    try:
        import generate_artificial_errors
        print("✅ generate_artificial_errors module")
    except ImportError as e:
        print(f"❌ generate_artificial_errors: {e}")
        return False

    # test basic functionality
    try:
        from generate_artificial_errors import ErrorGenerator
        generator = ErrorGenerator(approach="precise")
        print("✅ ErrorGenerator initialization")
    except Exception as e:
        print(f"❌ ErrorGenerator: {e}")
        return False

    return True

def main():
    print("BLISS Dataset Reconstruction Toolkit Setup")
    print("=" * 45)

    if not check_python_version():
        sys.exit(1)

    print("\nInstalling dependencies...")
    install_dependencies()

    print("\nDownloading models...")
    download_spacy_model()

    if verify_toolkit():
        print("\n🎉 Setup complete! Toolkit is ready to use.")
        print("\nNext steps:")
        print("1. Obtain Write&Improve 2024 data")
        print("2. Run: python convert_writeandimprove.py /path/to/data /path/to/output")
        print("3. Run: python generate_artificial_errors.py --datasets writeandimprove")
    else:
        print("\n❌ Setup failed. Check error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()