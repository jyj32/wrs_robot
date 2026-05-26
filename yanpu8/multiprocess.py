import multiprocessing
import subprocess
import time
# 双进程一键启动

# 解释器路径
interpreter = r"E:\anaconda\anaconda\envs\wrsrobot\python.exe"
# 文件路径
file1 = r"E:\py_project\wrsrobot\wrs_shu\yanpu_ur8\socket_send_image.py"
file2 = r"E:\py_project\wrsrobot\wrs_shu\yanpu_ur8\main.py"
def run_file1():
    """启动第一个Python文件"""
    subprocess.run([interpreter, file1])    # 解释器，文件名

def run_file2():
    """启动第二个Python文件"""
    subprocess.run([interpreter, file2])

if __name__ == "__main__":
    # 创建两个进程
    p1 = multiprocessing.Process(target=run_file1)
    p2 = multiprocessing.Process(target=run_file2)

    # 启动进程
    p1.start()
    time.sleep(5)
    p2.start()

    # 等待进程结束
    p1.join()
    time.sleep(5)
    p2.join()
