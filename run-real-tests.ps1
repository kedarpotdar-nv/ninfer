# Run the real-artifact suites natively. Usage: .\run-real-tests.ps1 -Artifact C:\models\model.ninfer
param([Parameter(Mandatory = $true)] [string] $Artifact)
$env:NINFER_TEST_ARTIFACT = $Artifact
Set-Location (Join-Path $PSScriptRoot "build-win")
# moe_real needs the 35B-A3B artifact and dflash_real a DFlash-v1 draft; prefix_real's default "all" scenario
# carries a Qwen3.5 chat-template golden, so its scenarios run individually below.
ctest -j1 --timeout 1800 --output-on-failure -R 'ninfer_qwen3_5_(loading_real|native_transactions|preemption_real|score_real|vision_workspace|dflash2_real|dflash_prefill_real|agent_continuation_real)_test'
$prefix = Join-Path (Get-Location) "tests\ninfer_qwen3_5_prefix_real_test.exe"
foreach ($scenario in "concurrent", "anthropic-prefix-regression", "nested-tool-markers", "shared-rewrite-materialization", "late-instructions", "agent-continuation", "explicit-prefix", "explicit-anchor", "rewrite-checkpoint", "stream-observations", "attention") {
    $env:NINFER_PREFIX_REAL_SCENARIO = $scenario
    & $prefix 2>&1 | Select-Object -Last 1 | Out-Null
    "prefix_real $scenario rc=$LASTEXITCODE"
}
Remove-Item Env:NINFER_PREFIX_REAL_SCENARIO -ErrorAction SilentlyContinue
