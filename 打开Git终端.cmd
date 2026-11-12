@echo off
chcp 65001 >nul
title Git 终端 - dog_yolo
set "PATH=%PATH%;D:\ds工作文档2\tools\PortableGit\cmd"
cd /d "%~dp0"

echo.
echo   ============================================
echo     Git 终端已就绪（本工程专用）
echo   ============================================
echo.
git --version
echo.
echo   当前目录: %CD%
echo.
echo   常用命令:
echo     git status                     查看改了什么
echo     git add .                      暂存全部改动
echo     git commit -m "说明"           提交
echo     git push                       推到 GitHub
echo     git log --oneline -5           看最近 5 条提交
echo.
echo   要推送到 GitHub 的话，直接运行:
echo     tools\push_github.ps1 -Repo https://github.com/你的用户名/仓库名.git
echo.

cmd /k
