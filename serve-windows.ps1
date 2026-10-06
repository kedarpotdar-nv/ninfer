# Serve the RTX 5090 recipe natively on Windows with the same flags as recipes/rtx5090-qwen38-27b/serve.sh.
# Usage: .\serve-windows.ps1 -Artifact C:\path\model.ninfer [-Port 9932] [-Concurrency 1] [-ExtraArgs @('--cors')]
param(
    [Parameter(Mandatory = $true)] [string] $Artifact,
    [int] $Port = 9932,
    [int] $Concurrency = 1,
    [string[]] $ExtraArgs = @()
)
$exe = Join-Path $PSScriptRoot "build-win\apps\ninfer-serve.exe"
& $exe $Artifact --host 127.0.0.1 --port $Port --model-id qwen3.8-27b `
    --max-context 65536 --kv-capacity 65536 --max-concurrency $Concurrency `
    --kv-dtype bf16 --prefill-chunk 2048 `
    --device-state-slots 4 --host-context-mib 8192 --preserve-thinking `
    --agent-prompt-cache `
    --spec dflash2 --draft-tokens 7 --lm-head-draft @ExtraArgs
