param([string]$BindAddress = 'llm-pc', [int]$Port = 17494)
$ErrorActionPreference = 'Stop'
$taskRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$taskLaunch = Get-Content -LiteralPath (Join-Path $taskRoot 'voice-launch.json') -Raw | ConvertFrom-Json
$env:STUDIO_VOICE_TOKEN = (Get-Content -LiteralPath (Join-Path $taskRoot 'voice-token.txt') -Raw).Trim()
$taskArguments = @('-m','apps.voice_studio.remote','--host',$BindAddress,'--port',"$Port")
$taskProcess = Start-Process -FilePath $taskLaunch.python -ArgumentList $taskArguments -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskRoot 'voice-server-output.log') -RedirectStandardError (Join-Path $taskRoot 'voice-server-error.log')
Write-Output "Started the voice server as PID $($taskProcess.Id) on http://${BindAddress}:$Port. Check its log for startup errors."
Write-Output 'Enter the token from voice-token.txt in the Voice Studio tab. Only the URL is remembered by the workstation.'
