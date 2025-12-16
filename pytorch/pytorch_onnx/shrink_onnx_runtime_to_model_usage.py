import os
import subprocess
import onnx
import shutil


def shrink_onnx_runtime(model_path: str, output_dir: str) -> None:
    """
    Shrinks the ONNX Runtime to include only the operators required by the given model.

    Args:
        model_path (str): Path to the ONNX model file.
        output_dir (str): Directory where the custom ONNX Runtime build will be saved.

    Raises:
        RuntimeError: If the ONNX Runtime custom build process fails.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get ONNX Runtime source directory
    ort_source = os.environ.get("ONNXRUNTIME_SOURCE", "/opt/onnxruntime")
    if not os.path.exists(ort_source):
        raise RuntimeError(f"ONNX Runtime source not found at: {ort_source}")

    # Step 1: Analyze the model to extract required operators
    print("Analyzing the ONNX model to extract required operators...")
    required_ops_file = os.path.join(output_dir, "model.required_operators.config")

    # Use the convert_onnx_models_to_ort tool to generate the config
    convert_cmd = [
        "python",
        os.path.join(ort_source, "tools/python/convert_onnx_models_to_ort.py"),
        model_path,
        "--output_dir",
        output_dir,
        "--optimization_style",
        "Fixed",
    ]
    proc = subprocess.Popen(
        convert_cmd,
        cwd=ort_source,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Model conversion failed (return code {proc.returncode})")
    print(f"Required operators configuration generated in: {output_dir}")

    # Step 2: Build a custom minimal ONNX Runtime
    print("Building a custom ONNX Runtime with only the required operators...")

    build_dir = os.path.join(output_dir, "build")
    os.makedirs(build_dir, exist_ok=True)

    # Build command for minimal runtime
    build_command = [
        os.path.join(ort_source, "build.sh"),
        "--config",
        "Release",
        "--build_dir",
        build_dir,
        "--parallel",
        "--skip_tests",
        "--build_shared_lib",
        "--minimal_build",
        "extended",
        "--include_ops_by_config",
        required_ops_file,
        "--cmake_extra_defines",
        "CMAKE_POLICY_VERSION_MINIMUM=3.5;" "--disable_contrib_ops",
        "--disable_ml_ops",
    ]

    # Execute build from ONNX Runtime source directory and stream logs immediately
    build_proc = subprocess.Popen(
        build_command,
        cwd=ort_source,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    for line in build_proc.stdout:
        print(line, end="", flush=True)
    build_proc.wait()
    if build_proc.returncode != 0:
        raise RuntimeError(
            f"ONNX Runtime custom build failed (return code {build_proc.returncode})"
        )

    print("Custom ONNX Runtime built successfully!")

    # Find the built library
    lib_path = os.path.join(build_dir, "Release")
    if os.path.exists(lib_path):
        print(f"Custom runtime libraries located at: {lib_path}")
        # List the built artifacts
        for item in os.listdir(lib_path):
            print(f"  - {item}")

    print(f"\nCustom runtime output directory: {output_dir}")
    print(f"To use this runtime, set LD_LIBRARY_PATH to: {lib_path}")

    # strip --strip-unneeded libonnxruntime.so

    for item in os.listdir(lib_path):
        if item.startswith("libonnxruntime") and item.endswith(".so"):
            lib_file = os.path.join(lib_path, item)
            print(f"Stripping unneeded symbols from {lib_file}...")
            subprocess.run(["strip", "--strip-unneeded", lib_file], check=True)
            print(f"Stripped {lib_file}")


def main():
    # Path to the ONNX model is in the folder where this script is located and is called model.onnx
    model_path = os.path.join(os.path.dirname(__file__), "model.onnx")

    # Directory to save the custom ONNX Runtime. Also where this script is located and called custom_onnx_runtime
    output_dir = os.path.join(os.path.dirname(__file__), "custom_onnx_runtime")

    # Shrink the ONNX Runtime
    shrink_onnx_runtime(model_path, output_dir)


if __name__ == "__main__":
    main()
