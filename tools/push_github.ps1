# 一键把 dog_yolo 推到 GitHub
#
#   .\tools\push_github.ps1 -Repo https://github.com/你的用户名/dog-detection-yolov8s.git
#   .\tools\push_github.ps1 -Repo https://github.com/你的用户名/dog-detection-yolov8s.git -NoDataset
#
# 第一次 push 会要求输入账号密码：密码位置填 Personal Access Token（不是登录密码）。

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Repo,
    [switch]$NoDataset,
    [string]$Branch = "main",
    [string]$Message = "YOLOv8s 狗检测：数据集 + 训练代码 + 结果"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Info($m) { Write-Host "[..] $m" -ForegroundColor Cyan }
function Ok($m) { Write-Host "[OK] $m" -ForegroundColor Green }
function Warn($m) { Write-Host "[!]  $m" -ForegroundColor Yellow }

# ---------- 1. 检查 git（优先用本工程自带的 PortableGit） ----------
$portable = Join-Path (Split-Path -Parent $root) "tools\PortableGit\cmd"
if ((-not (Get-Command git -ErrorAction SilentlyContinue)) -and (Test-Path (Join-Path $portable "git.exe"))) {
    $env:Path = "$env:Path;$portable"
    Info "使用随工程安装的 PortableGit：$portable"
}
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Warn "本机没有 git。先装一个，二选一："
    Write-Host "  a) python tools\install_git_portable.py    # 免安装，最快"
    Write-Host "  b) winget install --id Git.Git -e --source winget"
    Write-Host "装完关掉这个窗口重开，再跑一次本脚本。"
    exit 1
}
Ok ("git " + (git --version))

# ---------- 2. 初始化仓库 ----------
if (-not (Test-Path ".git")) {
    Info "初始化本地仓库"
    git init | Out-Null
}
git checkout -B $Branch 2>$null | Out-Null

# ---------- 3. 身份配置（没配过才提示） ----------
if (-not (git config user.name)) {
    $name = Read-Host "你的 GitHub 用户名（会写进提交记录）"
    git config user.name $name
}
if (-not (git config user.email)) {
    $mail = Read-Host "你的 GitHub 邮箱（没有就填 用户名@users.noreply.github.com）"
    git config user.email $mail
}

# ---------- 4. 要不要带数据集 ----------
if ($NoDataset) {
    $gi = ".gitignore"
    $mark = "# [push_github] 不传数据集"
    if (-not (Select-String -Path $gi -SimpleMatch $mark -Quiet)) {
        Add-Content $gi "`n$mark`ndataset/images/`ndataset/labels/"
        Info "已在 .gitignore 中排除 dataset/images 和 dataset/labels（只传 data.yaml 与统计）"
    }
}

# ---------- 5. 提交 ----------
Info "统计要提交的文件"
git add -A
$staged = (git diff --cached --name-only | Measure-Object).Count
$sizeMb = [math]::Round((git diff --cached --name-only | ForEach-Object {
            if (Test-Path $_) { (Get-Item $_).Length } else { 0 }
        } | Measure-Object -Sum).Sum / 1MB, 1)
Write-Host "     文件 $staged 个，约 $sizeMb MB"
if ($staged -eq 0) { Warn "没有改动，跳过提交" }
else {
    git commit -m $Message | Out-Null
    Ok "已提交"
}

# ---------- 6. 关联远端 ----------
$remotes = git remote
if ($remotes -contains "origin") { git remote set-url origin $Repo }
else { git remote add origin $Repo }
Ok "远端：$Repo"

# ---------- 7. 推送 ----------
if ($sizeMb -gt 80) {
    Warn "本次要传 $sizeMb MB，国内上传可能较慢；中断了就再跑一次本脚本（git 会续传已打包的对象）"
}
Info "开始 push（第一次会让你输入用户名 + Token）"
git push -u origin $Branch
if ($LASTEXITCODE -ne 0) {
    Warn "push 失败。常见两种情况："
    Write-Host "  1) 远端已有 README/文件 → 先执行：git pull --rebase origin $Branch  再重跑"
    Write-Host "  2) 认证失败 → 密码处要填 Personal Access Token，不是账号密码"
    exit $LASTEXITCODE
}
Ok "完成，仓库地址：$($Repo -replace '\.git$', '')"
