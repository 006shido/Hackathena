param(
    [string]$PythonExe = $(if (Test-Path (Join-Path $PSScriptRoot '../../.venv/Scripts/python.exe')) { Join-Path $PSScriptRoot '../../.venv/Scripts/python.exe' } else { Join-Path $env:USERPROFILE '.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' }),
    [switch]$EnableTracking,
    [switch]$EnableMediaDetection,
    [switch]$EnableWebRTCVideo,
    [switch]$ForwardedDemo,
    [switch]$SimpleDemoPasswords,
    [ValidateRange(1024,65535)][int]$AppPort = 5001,
    [ValidateRange(1024,65535)][int]$MlPort = 8001,
    [switch]$CheckOnly,
    [int]$SmokeSeconds = 0
)
$ErrorActionPreference = 'Stop'
$researchRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$pythonPath = (Resolve-Path -LiteralPath $PythonExe).Path
$nodePath = (Get-Command node.exe -ErrorAction Stop).Source
$assets = @{
    'ml/experiments/reference_backend/assets/inswapper_128.onnx' = 'E4A3F08C753CB72D04E10AA0F7DBE3DEEBBF39567D4EAD6DCE08E98AA49E16AF'
    'ml/experiments/reference_backend/assets/w600k_r50.onnx' = '4C06341C33C2CA1F86781DAB0E829F88AD5B64BE9FBA56E56BC9EBDEFC619E43'
    'ml/models/weights/ms1mv2_iresnet50.pth' = '2B75B93C48B01C78A4263F7295AB2DBF84F85190C51BE45B92C4C0B0AEDCEAA3'
}
foreach ($entry in $assets.GetEnumerator()) {
    $assetPath = Join-Path $researchRoot $entry.Key
    if ((Get-FileHash -LiteralPath $assetPath -Algorithm SHA256).Hash -ne $entry.Value) { throw "Asset verification failed: $($entry.Key)" }
}
foreach ($relative in @('ml/experiments/phase6g_3_full_dataset/run/best_model.pt','client/public/models/face_landmarker.task','server/dist/index.js','client/node_modules/vite/bin/vite.js')) {
    if (-not (Test-Path -LiteralPath (Join-Path $researchRoot $relative))) { throw "Missing required file: $relative" }
}
if ($SmokeSeconds -lt 0) { throw 'SmokeSeconds must be nonnegative.' }
if (-not $ForwardedDemo -and $AppPort -ne 5001) { throw 'Custom AppPort requires -ForwardedDemo; the development proxy uses port 5001.' }
if ($AppPort -eq $MlPort -or $AppPort -eq 5173 -or $MlPort -eq 5173) { throw 'Application, ML and development client ports must differ.' }
Write-Host 'Verified research assets. This preview uses research-restricted pretrained weights; general rollout is not approved.'
if ($CheckOnly) { return }
foreach ($port in $(if ($ForwardedDemo) { @($AppPort,$MlPort) } else { @($AppPort,5173,$MlPort) })) {
    if (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) { throw "Port $port is already in use. Existing services were left running." }
}
$logDirectory = Join-Path $researchRoot ('ml/experiments/reference_backend/local_preview/' + [DateTime]::UtcNow.ToString('yyyyMMdd-HHmmss-fff'))
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
$configuration = @{
    PYTHONPATH = $(if ($EnableWebRTCVideo) { (Join-Path $researchRoot 'ml/runtime_webrtc') + ';' + (Join-Path $researchRoot '.venv/Lib/site-packages') } else { Join-Path $researchRoot '.venv/Lib/site-packages' })
    ML_ENABLE_WEBRTC_VIDEO = $(if ($EnableWebRTCVideo) { '1' } else { '0' })
    VITE_ENABLE_WEBRTC_VIDEO = $(if ($EnableWebRTCVideo) { 'true' } else { 'false' })
    ML_ENABLE_VIDEO_PREVIEW = '1'; ML_VIDEO_BACKEND = 'research-reference'
    ML_ENABLE_MEDIA_DETECTION = $(if ($EnableMediaDetection) { '1' } else { '0' })
    VITE_ENABLE_MEDIA_DETECTION = $(if ($EnableMediaDetection) { 'true' } else { 'false' })
    ML_VIDEO_TRACKING = $(if ($EnableTracking) { '1' } else { '0' })
    ML_MODEL_VARIANT = 'fullres'
    ML_CHECKPOINT_PATH = (Join-Path $researchRoot 'ml/experiments/phase6g_3_full_dataset/run/best_model.pt')
    VITE_ENABLE_NEURAL_VIDEO_PREVIEW = 'true'; ML_SERVICE_PORT = [string]$MlPort; PORT = [string]$AppPort; HOST = '127.0.0.1'
}
if ($ForwardedDemo) {
    function New-DemoSecret {
        $bytes = New-Object byte[] 32
        $random = [Security.Cryptography.RandomNumberGenerator]::Create()
        try { $random.GetBytes($bytes) } finally { $random.Dispose() }
        return [Convert]::ToBase64String($bytes)
    }
    $configuration.PUBLIC_RESEARCH_DEMO = '1'
    $configuration.LOCAL_DEMO_LOGIN = '1'
    $configuration.VITE_DEMO_PASSWORD_REQUIRED = $(if ($SimpleDemoPasswords) { 'false' } else { 'true' })
    $configuration.DEMO_DEFAULT_PASSWORDS = $(if ($SimpleDemoPasswords) { '1' } else { '0' })
    $configuration.JWT_SECRET = New-DemoSecret
    $configuration.DEMO_USER_PASSWORD = $(if ($SimpleDemoPasswords) { 'user123' } else { New-DemoSecret })
    $configuration.DEMO_TESTER_PASSWORD = $(if ($SimpleDemoPasswords) { 'tester123' } else { New-DemoSecret })
    $configuration.CLIENT_DIST_PATH = Join-Path $logDirectory 'client-dist'
    $accessPath = Join-Path $logDirectory 'demo-access.local.txt'
    @("user: $($configuration.DEMO_USER_PASSWORD)", "tester: $($configuration.DEMO_TESTER_PASSWORD)") | Set-Content -LiteralPath $accessPath
    Write-Host "Private demo credentials: $accessPath (do not commit or share publicly)."
    if (-not $env:VITE_TURN_URLS) {
        Write-Warning 'No TURN relay configured. Different-network media connectivity is not guaranteed; test both devices before the demo.'
    }
}
$savedEnvironment = @{}
$ownedProcesses = [Collections.Generic.List[Diagnostics.Process]]::new()
function Start-OwnedProcess([string]$Executable,[string[]]$Arguments,[string]$Directory,[string]$Name) {
    $quotedArguments = $Arguments | ForEach-Object { '"' + $_.Replace('"','\"') + '"' }
    $child = Start-Process -FilePath $Executable -ArgumentList $quotedArguments -WorkingDirectory $Directory -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logDirectory "$Name.log") -RedirectStandardError (Join-Path $logDirectory "$Name.error.log")
    $ownedProcesses.Add($child)
    return $child
}
try {
    foreach ($key in $configuration.Keys) {
        $savedEnvironment[$key] = [Environment]::GetEnvironmentVariable($key,'Process')
        [Environment]::SetEnvironmentVariable($key,$configuration[$key],'Process')
    }
    if ($ForwardedDemo) {
        Push-Location (Join-Path $researchRoot 'client')
        try {
            & $nodePath (Join-Path $researchRoot 'client/node_modules/vite/bin/vite.js') build --outDir $configuration.CLIENT_DIST_PATH
            if ($LASTEXITCODE -ne 0) { throw 'Forwarded demo frontend build failed.' }
        } finally { Pop-Location }
    }
    $mlProcess = Start-OwnedProcess $pythonPath @('-m','uvicorn','ml.api.app:app','--host','127.0.0.1','--port',[string]$MlPort,'--no-access-log') $researchRoot 'ml'
    $serverProcess = Start-OwnedProcess $nodePath @((Join-Path $researchRoot 'server/dist/index.js')) (Join-Path $researchRoot 'server') 'server'
    if (-not $ForwardedDemo) {
        $clientProcess = Start-OwnedProcess $nodePath @((Join-Path $researchRoot 'client/node_modules/vite/bin/vite.js'),'--host','127.0.0.1','--port','5173','--strictPort') (Join-Path $researchRoot 'client') 'client'
    }
    foreach ($key in $savedEnvironment.Keys) { [Environment]::SetEnvironmentVariable($key,$savedEnvironment[$key],'Process') }
    $deadline = [DateTime]::UtcNow.AddSeconds(90)
    $previewUrl = if ($ForwardedDemo) { "http://127.0.0.1:$AppPort" } else { 'http://127.0.0.1:5173' }
    $ready = $false
    while ([DateTime]::UtcNow -lt $deadline) {
        foreach ($child in $ownedProcesses) { if ($child.HasExited) { throw "Preview process exited. Inspect $logDirectory" } }
        try {
            $health = Invoke-RestMethod "http://127.0.0.1:$MlPort/health" -TimeoutSec 2
            $client = Invoke-WebRequest $previewUrl -TimeoutSec 2
            if ($health.ready -and $health.video_backend -eq 'research-reference' -and $health.video_preview_enabled -and $client.StatusCode -eq 200) { $ready = $true; break }
        } catch { }
        Start-Sleep -Milliseconds 500
    }
    if (-not $ready) { throw "Preview did not become ready. Inspect $logDirectory" }
    @{ ready=$ready; health=$health; tracking=[bool]$EnableTracking; forwarded_demo=[bool]$ForwardedDemo; url=$previewUrl; logs=$logDirectory } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $logDirectory 'startup.json')
    Write-Host "Research preview ready: $previewUrl  Logs: $logDirectory"
    if ($ForwardedDemo) {
        Write-Host "Forward only port $AppPort in VS Code. Use its HTTPS link on both devices and the generated demo credentials. Do not forward ML port $MlPort."
    } else {
        Write-Host 'Use your normal tester/user accounts in separate browser profiles.'
    }
    Write-Host 'Ctrl+C stops only these preview processes.'
    $end = if ($SmokeSeconds -gt 0) { [DateTime]::UtcNow.AddSeconds($SmokeSeconds) } else { [DateTime]::MaxValue }
    while ([DateTime]::UtcNow -lt $end) {
        foreach ($child in $ownedProcesses) { if ($child.HasExited) { throw "Preview process exited. Inspect $logDirectory" } }
        Start-Sleep -Milliseconds 500
    }
} finally {
    foreach ($key in $savedEnvironment.Keys) { [Environment]::SetEnvironmentVariable($key,$savedEnvironment[$key],'Process') }
    foreach ($child in $ownedProcesses) { if (-not $child.HasExited) { Stop-Process -Id $child.Id -Force -ErrorAction SilentlyContinue } }
}
