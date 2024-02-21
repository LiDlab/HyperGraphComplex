import subprocess
import torch


def run_python_script(script_path):
    print(f"Running script: {script_path}")
    subprocess.run(["python", script_path])


def main():
    # 按顺序运行其他Python脚本
    script_list = ["preprocessing.py", "HGVAE.py", "Classifier_module_DNN.py"]

    for script in script_list:
        run_python_script(script)

        # 清理GPU内存
        torch.cuda.empty_cache()
        print("GPU memory has been cleared.")


if __name__ == "__main__":
    main()