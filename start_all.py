"""
启动 FactorFlow 完整服务（后端API + 前端HTTP服务器）
"""
import subprocess
import sys
import webbrowser
import time
from pathlib import Path

def main():
    print("=" * 60)
    print("启动 FactorFlow 完整服务")
    print("=" * 60)

    # 项目根目录
    project_root = Path(__file__).parent
    frontend_dir = project_root / "frontend" / "web"

    # 检查前端目录
    if not frontend_dir.exists():
        print(f"错误: 前端目录不存在: {frontend_dir}")
        return

    try:
        # 启动后端 API 服务
        print("\n[1/2] 启动后端 API 服务...")
        api_process = subprocess.Popen(
            [sys.executable, "start_api.py"],
            cwd=str(project_root)
        )

        # 等待 API 服务启动
        print("等待 API 服务启动...")
        time.sleep(3)

        # 启动前端 HTTP 服务器
        print("\n[2/2] 启动前端 HTTP 服务器 (端口 8080)...")
        frontend_process = subprocess.Popen(
            [sys.executable, "-m", "http.server", "8080"],
            cwd=str(frontend_dir)
        )

        # 打开浏览器
        print("\n" + "=" * 60)
        print("服务启动完成!")
        print("=" * 60)
        print(f"前端地址: http://localhost:8080")
        print(f"API 地址: http://localhost:8000")
        print(f"API 文档: http://localhost:8000/docs")
        print("=" * 60)
        print("\n正在打开浏览器...")

        time.sleep(1)
        webbrowser.open("http://localhost:8080")

        print("\n按 Ctrl+C 停止所有服务")
        print("-" * 60)

        # 等待用户中断
        try:
            while True:
                time.sleep(1)
                # 检查进程是否还在运行
                if api_process.poll() is not None:
                    print("\nAPI 服务已停止")
                    break
                if frontend_process.poll() is not None:
                    print("\n前端服务已停止")
                    break
        except KeyboardInterrupt:
            print("\n\n正在停止服务...")
            api_process.terminate()
            frontend_process.terminate()
            api_process.wait()
            frontend_process.wait()
            print("所有服务已停止")

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
