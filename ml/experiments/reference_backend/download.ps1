$ErrorActionPreference = 'Stop'
$assetRoot = Join-Path $PSScriptRoot 'assets'
New-Item -ItemType Directory -Path $assetRoot -Force | Out-Null
# Release SHA-256 digests independently verified through the browser tool:
# https://github.com/deepinsight/insightface/releases/expanded_assets/model-zoo
# The execution host's HTTPS certificate chain is unavailable. These public
# downloads are accepted ONLY after matching the pinned official digest.
$assets = @(
    @{ Name = 'inswapper_128.onnx'; Sha = 'e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af' },
    @{ Name = 'buffalo_l.zip'; Sha = '80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f' }
)
foreach ($asset in $assets) {
    $destination = Join-Path $assetRoot $asset.Name
    if (Test-Path -LiteralPath $destination) {
        if ((Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLower() -eq $asset.Sha) {
            Write-Output "Already verified: $($asset.Name)"
            continue
        }
        throw "Existing asset digest mismatch: $destination"
    }
    $temporary = "$destination.download"
    $url = 'https://github.com/deepinsight/insightface/releases/download/model-zoo/' + $asset.Name
    & curl.exe --insecure --location --fail --silent --show-error --max-time 900 --output $temporary $url
    if ($LASTEXITCODE -ne 0) { throw "Official download failed: $url" }
    if ((Get-FileHash -LiteralPath $temporary -Algorithm SHA256).Hash.ToLower() -ne $asset.Sha) {
        throw "Downloaded asset digest mismatch; unverified file remains quarantined: $temporary"
    }
    Move-Item -LiteralPath $temporary -Destination $destination
    Write-Output "SHA-256 verified: $($asset.Name)"
}
