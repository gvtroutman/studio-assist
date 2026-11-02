param(
    [string]$Python = 'python',
    [ValidateSet('voxcpm','qwen','omnivoice','chatterbox','kokoro','pocket')]
    [string[]]$Engines = @('voxcpm','qwen','omnivoice'),
    [switch]$CpuOnly
)
# Run this file on the LLM PC. Studio Assist never invokes it locally.
$ErrorActionPreference = 'Stop'
$taskRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$taskPython = (Get-Command $Python -ErrorAction Stop).Source
$taskVersion = & $taskPython -c 'import sys; print("%d.%d" % sys.version_info[:2])'
if ($LASTEXITCODE -ne 0) { throw 'Python did not start.' }
if ($taskVersion.Trim() -ne '3.11') { throw 'Use a separate Python 3.11 installation for these engine environments.' }
$taskPackages = @{
    voxcpm = 'voxcpm==2.0.3'; qwen = 'qwen-tts==0.1.1'; omnivoice = 'omnivoice==0.2.1'
    chatterbox = 'chatterbox-tts==0.1.7'; kokoro = 'kokoro==0.9.4'; pocket = 'pocket-tts==3.3.0'
}
$taskConfigPath = Join-Path $taskRoot 'voice-environments.json'
$taskConfig = @{}
if (Test-Path -LiteralPath $taskConfigPath) {
    $taskOld = Get-Content -LiteralPath $taskConfigPath -Raw | ConvertFrom-Json
    foreach ($taskProperty in $taskOld.PSObject.Properties) { $taskConfig[$taskProperty.Name] = $taskProperty.Value }
}
foreach ($taskEngine in $Engines) {
    $taskEnv = Join-Path $taskRoot "runtime/$taskEngine"
    $taskExe = Join-Path $taskEnv 'Scripts/python.exe'
    if (-not (Test-Path -LiteralPath $taskExe)) {
        & $taskPython -m venv $taskEnv
        if ($LASTEXITCODE -ne 0) { throw "Could not create the $taskEngine environment." }
    }
    & $taskExe -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw 'Could not update the isolated environment installer.' }
    $taskCpu = $CpuOnly -or $taskEngine -in @('kokoro','pocket')
    $taskTorchIndex = if ($taskCpu) { 'https://download.pytorch.org/whl/cpu' } elseif ($taskEngine -eq 'chatterbox') { 'https://download.pytorch.org/whl/cu124' } else { 'https://download.pytorch.org/whl/cu128' }
    $taskTorchVersion = if ($taskEngine -eq 'chatterbox') { '2.6.0' } else { '2.8.0' }
    & $taskExe -m pip install "torch==$taskTorchVersion" "torchaudio==$taskTorchVersion" --index-url $taskTorchIndex
    if ($LASTEXITCODE -ne 0) { throw "Could not install PyTorch for $taskEngine." }
    $taskConstraints = Join-Path $taskEnv 'voice-constraints.txt'
    $taskConstraintText = "torch==$taskTorchVersion`ntorchaudio==$taskTorchVersion`n"
    if ($taskEngine -eq 'voxcpm') {
        # TorchCodec releases track PyTorch; an unbounded dependency can
        # install incompatible native DLLs beside the pinned GPU build.
        $taskConstraintText += "torchcodec==0.7.0`ntransformers==4.57.3`n"
    }
    [IO.File]::WriteAllText($taskConstraints, $taskConstraintText, (New-Object Text.UTF8Encoding($false)))
    & $taskExe -m pip install --constraint $taskConstraints $taskPackages[$taskEngine] 'soundfile==0.14.0'
    if ($LASTEXITCODE -ne 0) { throw "Could not install $taskEngine. The other environments are unchanged." }
    if ($taskEngine -eq 'kokoro') {
        & $taskExe -m pip install 'https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl'
        if ($LASTEXITCODE -ne 0) { throw 'Could not install the English pronunciation model.' }
    }
    & $taskExe -m pip check
    if ($LASTEXITCODE -ne 0) { throw "The $taskEngine environment has incompatible dependencies." }
    $taskModules = @{ voxcpm='voxcpm'; qwen='qwen_tts'; omnivoice='omnivoice'; chatterbox='chatterbox.tts_turbo'; kokoro='kokoro'; pocket='pocket_tts' }
    & $taskExe -c 'import importlib, sys, torch, torchaudio; importlib.import_module(sys.argv[1]); print("torch", torch.__version__, "CUDA available:", torch.cuda.is_available()); assert sys.argv[2] == "True" or torch.cuda.is_available(), "CUDA is unavailable; check the remote NVIDIA driver or use -CpuOnly"' $taskModules[$taskEngine] "$taskCpu"
    if ($LASTEXITCODE -ne 0) { throw "The $taskEngine import/GPU check failed. Check its output; VoxCPM may need shared FFmpeg DLLs on PATH." }
    $taskDevice = if ($taskCpu) { 'cpu' } else { 'cuda:0' }
    $taskConfig[$taskEngine] = @{ python=$taskExe; device=$taskDevice }
    Write-Output "Installed $taskEngine in its own environment. Model weights download only when allowed in Voice Studio."
}
$taskConfigJson = $taskConfig | ConvertTo-Json -Depth 8
[IO.File]::WriteAllText($taskConfigPath, $taskConfigJson, (New-Object Text.UTF8Encoding($false)))
$taskTokenPath = Join-Path $taskRoot 'voice-token.txt'
if (-not (Test-Path -LiteralPath $taskTokenPath)) {
    [IO.File]::WriteAllText($taskTokenPath, ([guid]::NewGuid().ToString('N') + [guid]::NewGuid().ToString('N')))
}
$taskLaunch = @{ python=$taskPython } | ConvertTo-Json
[IO.File]::WriteAllText((Join-Path $taskRoot 'voice-launch.json'), $taskLaunch, (New-Object Text.UTF8Encoding($false)))
Write-Output 'Setup complete. Run apps/voice_studio/start_remote.ps1 on this PC, then enter its URL and the token from voice-token.txt in Studio Assist.'
