#install choco

Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))


$resources = @{
    "py311.exe"="https://www.python.org/ftp/python/3.11.2/python-3.11.2-amd64.exe";
    "py310.exe"="https://www.python.org/ftp/python/3.10.10/python-3.10.10-amd64.exe -O py310.exe";
    "py39.exe"="https://www.python.org/ftp/python/3.9.13/python-3.9.13-amd64.exe";
    "py38.exe"="https://www.python.org/ftp/python/3.8.10/python-3.8.10-amd64.exe";
    "msys64.exe"="https://github.com/msys2/msys2-installer/releases/download/2023-01-27/msys2-x86_64-20230127.exe";
    "visualstudio.exe"="https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=Community&channel=Release&version=VS2019&source=VSLandingPage&cid=2030&passive=false";
    "git.exe"="https://github.com/git-for-windows/git/releases/download/v2.39.2.windows.1/Git-2.39.2-64-bit.exe";
    "vscode"=""
}

set-location %userprofile%\download

function Download-Resources{
    foreach($k in $resources.Keys){
        $v = $resources[$k]

    }
}

choco install wget
choco install bazelisk
bazel -h

