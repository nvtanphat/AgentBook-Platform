param(
    [string]$Mode = "e2e-gold-v2",
    [string]$Input = "D:\GenAI\DoAn01\evaluation\datasets\gold\meta_dataset.jsonl",
    [string]$Output = "D:\GenAI\DoAn01\evaluation\datasets\gold\e2e_gold_v2.jsonl"
)

Set-Location D:\GenAI\DoAn01

# Load .env
Get-Content "D:\GenAI\DoAn01\backend\.env" | Where-Object { $_ -match "^[A-Z]" -and $_ -notmatch "^#" } | ForEach-Object {
    $parts = $_ -split "=", 2
    if ($parts.Count -eq 2) { [Environment]::SetEnvironmentVariable($parts[0].Trim(), $parts[1].Trim(), "Process") }
}

$env:PYTHONIOENCODING = "utf-8"

python evaluation/cli/generate_dataset.py `
    --owner-id nvtanphat69_gmail_com `
    --collection-id 6a3569119a31a28f07578964 `
    --mode $Mode `
    --input $Input `
    --output $Output `
    --api-url http://localhost:8000 `
    --provider openai `
    --model gpt-4.1-mini `
    --api-base https://luongchidung.online/v1 `
    --temperature 0 `
    --seed 42
