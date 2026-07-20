# 面向AI的Linux

> 大多数AI运行在Linux上。你需要掌握足够的知识，以免陷入困境。

**类型：** 学习
**语言：** --
**先修要求：** 阶段 0，第 01 课
**时间：** 约 30 分钟

## 学习目标

- 从命令行导航Linux文件系统并执行基本文件操作
- 使用`chmod`和`chown`管理文件权限以解决"Permission denied"错误
- 使用`chmod`安装系统包并为AI工作设置全新的GPU机器
- 识别macOS与Linux之间的差异，这些差异经常困扰在远程机器上工作的开发者

## 问题

你在macOS或Windows上开发。但当你SSH进入云GPU机器、租用Lambda实例或启动EC2机器时，你会进入Ubuntu。终端是你唯一的界面。没有Finder，没有Explorer，没有图形界面。如果你无法从命令行导航文件系统、安装包和管理进程，你只能为闲置的GPU时间付费，同时谷歌搜索"how to unzip a file in Linux"。

这是一本生存指南。它涵盖了在远程Linux机器上进行AI工作所需的一切。仅此而已。

## 文件系统布局

Linux将所有内容组织在单个根目录`/`下。没有`C:\`或`/Volumes`。你会实际用到的目录：

```mermaid
graph TD
    root["/"] --> home["home/your-username/<br/>Your files — clone repos, run training"]
    root --> tmp["tmp/<br/>Temporary files, cleared on reboot"]
    root --> usr["usr/<br/>System programs and libraries"]
    root --> etc["etc/<br/>Config files"]
    root --> varlog["var/log/<br/>Logs — check when something breaks"]
    root --> mnt["mnt/ or /media/<br/>External drives and volumes"]
    root --> proc["proc/ and /sys/<br/>Virtual files — kernel and hardware info"]
```

你的家目录是`~`或`/home/your-username`。你几乎做所有事情都在这里。

## 基本命令

以下15个命令涵盖了你在远程GPU机器上95%的操作。

### 导航

```bash
pwd                         # Where am I?
ls                          # What's here?
ls -la                      # What's here, including hidden files with details?
cd /path/to/dir             # Go there
cd ~                        # Go home
cd ..                       # Go up one level
```

### 文件和目录

```bash
mkdir my-project            # Create a directory
mkdir -p a/b/c              # Create nested directories in one shot

cp file.txt backup.txt      # Copy a file
cp -r src/ src-backup/      # Copy a directory (recursive)

mv old.txt new.txt          # Rename a file
mv file.txt /tmp/           # Move a file

rm file.txt                 # Delete a file (no trash, it's gone)
rm -rf my-dir/              # Delete a directory and everything inside
```

`rm -rf`是永久性的。没有撤销操作。在按回车前请仔细检查路径。

### 读取文件

```bash
cat file.txt                # Print entire file
head -20 file.txt           # First 20 lines
tail -20 file.txt           # Last 20 lines
tail -f log.txt             # Follow a log file in real time (Ctrl+C to stop)
less file.txt               # Scroll through a file (q to quit)
```

### 搜索

```bash
grep "error" training.log           # Find lines containing "error"
grep -r "learning_rate" .           # Search all files in current directory
grep -i "cuda" config.yaml          # Case-insensitive search

find . -name "*.py"                 # Find all Python files under current dir
find . -name "*.ckpt" -size +1G     # Find checkpoint files larger than 1GB
```

## 权限

Linux中的每个文件都有所有者(owner)和权限位(permission bits)。当脚本无法执行或无法写入目录时，你会遇到这个问题。

```bash
ls -l train.py
# -rwxr-xr-- 1 user group 2048 Mar 19 10:00 train.py
#  ^^^             owner permissions: read, write, execute
#     ^^^          group permissions: read, execute
#        ^^        everyone else: read only
```

常见修复方法：

```bash
chmod +x train.sh           # Make a script executable
chmod 755 deploy.sh         # Owner: full, others: read+execute
chmod 644 config.yaml       # Owner: read+write, others: read only

chown user:group file.txt   # Change who owns a file (needs sudo)
```

当提示"权限被拒绝"时，几乎总是权限问题。`chmod +x`或`sudo`可以解决大多数情况。

## 包管理 (apt)

Ubuntu使用`apt`。这是安装系统级软件的方式。

```bash
sudo apt update             # Refresh the package list (always do this first)
sudo apt install -y htop    # Install a package (-y skips confirmation)
sudo apt install -y build-essential  # C compiler, make, etc. Needed by many Python packages
sudo apt install -y tmux    # Terminal multiplexer (keep sessions alive after disconnect)

apt list --installed        # What's installed?
sudo apt remove htop        # Uninstall
```

在新GPU机上常安装的软件包：

```bash
sudo apt update && sudo apt install -y \
    build-essential \
    git \
    curl \
    wget \
    tmux \
    htop \
    unzip \
    python3-venv
```

## 用户和sudo

你通常以普通用户身份登录。某些操作需要root（管理员）权限。

```bash
whoami                      # What user am I?
sudo command                # Run a single command as root
sudo su                     # Become root (exit to go back, use sparingly)
```

在云GPU实例上，你通常是唯一用户，并且已有sudo权限。不要以root身份运行所有命令，仅在需要时使用sudo。

## 进程和systemd

当训练挂起或需要检查运行内容时：

```bash
htop                        # Interactive process viewer (q to quit)
ps aux | grep python        # Find running Python processes
kill 12345                  # Gracefully stop process with PID 12345
kill -9 12345               # Force kill (use when graceful doesn't work)
nvidia-smi                  # GPU processes and memory usage
```

systemd管理服务（后台守护进程）。如果你运行推理服务器(inference servers)时会用到它：

```bash
sudo systemctl start nginx          # Start a service
sudo systemctl stop nginx           # Stop it
sudo systemctl restart nginx        # Restart it
sudo systemctl status nginx         # Check if it's running
sudo systemctl enable nginx         # Start automatically on boot
```

## 磁盘空间

GPU机通常磁盘空间有限。模型和数据集会很快填满它。

```bash
df -h                       # Disk usage for all mounted drives
df -h /home                 # Disk usage for /home specifically

du -sh *                    # Size of each item in current directory
du -sh ~/.cache             # Size of your cache (pip, huggingface models land here)
du -sh /data/checkpoints/   # Check how big your checkpoints are

# Find the biggest space hogs
du -h --max-depth=1 / 2>/dev/null | sort -hr | head -20
```

常见节省空间的方法：

```bash
# Clear pip cache
pip cache purge

# Clear apt cache
sudo apt clean

# Remove old checkpoints you don't need
rm -rf checkpoints/epoch_01/ checkpoints/epoch_02/
```

## 网络

你将通过命令行下载模型、传输文件以及调用API。

```bash
# Download files
wget https://example.com/model.bin                   # Download a file
curl -O https://example.com/data.tar.gz              # Same thing with curl
curl -s https://api.example.com/health | python3 -m json.tool  # Hit an API, pretty-print JSON

# Transfer files between machines
scp model.bin user@remote:/data/                     # Copy file to remote machine
scp user@remote:/data/results.csv .                  # Copy file from remote to local
scp -r user@remote:/data/checkpoints/ ./local-dir/   # Copy directory

# Sync directories (faster than scp for large transfers, resumes on failure)
rsync -avz --progress ./data/ user@remote:/data/
rsync -avz --progress user@remote:/results/ ./results/
```

对于大文件，使用`rsync`而不是`scp`。它只传输发生变化的字节，并能处理断开的连接。

## tmux：保持会话活跃

当你通过SSH连接到远程机器时，关闭笔记本电脑会导致训练进程中断。tmux可以避免这种情况。

```bash
tmux new -s train           # Start a new session named "train"
# ... start your training, then:
# Ctrl+B, then D            # Detach (training keeps running)

tmux ls                     # List sessions
tmux attach -t train        # Reattach to session

# Inside tmux:
# Ctrl+B, then %            # Split pane vertically
# Ctrl+B, then "            # Split pane horizontally
# Ctrl+B, then arrow keys   # Switch between panes
```

始终在tmux中运行长时间的训练任务。始终如此。

## Windows用户的WSL2

如果你使用Windows，WSL2能为你提供真实的Linux环境，无需双系统。

```bash
# In PowerShell (admin)
wsl --install -d Ubuntu-24.04

# After restart, open Ubuntu from Start menu
sudo apt update && sudo apt upgrade -y
```

WSL2运行真正的Linux内核。本课中的所有内容都在其中工作。你的Windows文件在WSL中位于`/mnt/c/Users/YourName/`。

GPU透传在Windows端安装NVIDIA驱动的情况下工作。安装Windows版NVIDIA驱动（不要装Linux版），CUDA即可在WSL2中使用。

## macOS到Linux的陷阱：

如果你从macOS迁移过来，以下几件事可能会让你遇到问题：

|  macOS  |  Linux  |  备注  |
|-------|-------|-------|
|  `brew install`  |  `sudo apt install`  |  有时包名不同。`brew install htop`和`sudo apt install htop`工作方式相同，但`brew install readline`和`sudo apt install libreadline-dev`则不同。  |
|  `open file.txt`  |  `xdg-open file.txt`  |  但在远程机器上没有图形用户界面。使用`cat`或`less`。 |
|  `pbcopy` / `pbpaste`  |  不可用  |  通过SSH，剪贴板的管道不存在。 |
|  `~/.zshrc`  |  `~/.bashrc`  |  macOS默认使用zsh。大多数Linux服务器使用bash。 |
|  `/opt/homebrew/`  |  `/usr/bin/`, `/usr/local/bin/`  |  二进制文件位于不同位置。 |
|  `sed -i '' 's/a/b/' file`  |  `sed -i 's/a/b/' file`  |  macOS的sed需要在`-i`后跟一个空字符串。Linux则不需要。 |
|  不区分大小写的文件系统  |  区分大小写的文件系统  |  在Linux上，`Model.py`和`model.py`是两个不同的文件。 |
|  行结束符`\n`  |  行结束符`\n`  |  相同。但Windows使用`\r\n`，这会破坏bash脚本。运行`dos2unix`修复。 |

## 快速参考卡

```
Navigation:     pwd, ls, cd, find
Files:          cp, mv, rm, mkdir, cat, head, tail, less
Search:         grep, find
Permissions:    chmod, chown, sudo
Packages:       apt update, apt install
Processes:      htop, ps, kill, nvidia-smi
Services:       systemctl start/stop/restart/status
Disk:           df -h, du -sh
Network:        curl, wget, scp, rsync
Sessions:       tmux new/attach/detach
```

## 练习

1. SSH进入任何Linux机器（或打开WSL2）并导航到你的主目录。创建一个项目文件夹，在其中用`touch`创建三个空文件，然后用`ls -la`列出它们。
2. 用apt安装`touch`，运行它，并识别哪个进程占用内存最多。
3. 启动一个tmux会话，在里面运行`touch`，分离，列出会话，然后重新连接。
4. 使用`touch`检查可用磁盘空间，然后使用`ls -la`查找缓存中占用空间的内容。
5. 使用`touch`将文件从本地机器传输到远程机器，然后使用`ls -la`进行同样的传输，并比较体验。
