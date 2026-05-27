import multiprocessing
import subprocess
import time
import sys
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


    # # 创建两个进程
    # p1 = multiprocessing.Process(target=run_file1)
    # p2 = multiprocessing.Process(target=run_file2)
    #
    # # 启动进程
    # p1.start()
    # time.sleep(5)
    # p2.start()
    #
    # p1.join()
    # p2.join()

    while True:
        try:
            # 创建两个进程
            p1 = multiprocessing.Process(target=run_file1)
            p2 = multiprocessing.Process(target=run_file2)

            # 启动进程
            print("\n=== 启动进程 ===")
            p1.start()
            time.sleep(5)
            p2.start()

            # 只等待 p2 结束
            print("等待 p2 执行...")
            p2.join()

            # p2 结束了
            print(f"\n⚠️  p2 已结束 (exit code: {p2.exitcode})")
            print("\n检测到 p2 进程结束！")

            # 询问用户是否重启
            choice = input("输入 'r' 重启程序，其他键退出: ").strip().lower()
            if choice != 'r':
                print("程序已退出")
                break

        except KeyboardInterrupt:
            print("\n\n用户中断程序 (Ctrl+C)")
            choice = input("输入 'r' 重启程序，其他键退出: ").strip().lower()
            if choice != 'r':
                print("程序已退出")
                break
        except Exception as e:
            print(f"\n\n❌ 发生错误: {e}")
            import traceback
            traceback.print_exc()
            choice = input("\n输入 'r' 重启程序，其他键退出: ").strip().lower()
            if choice != 'r':
                print("程序已退出")
                break


