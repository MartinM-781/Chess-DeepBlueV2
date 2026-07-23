# Garde-éveil : empêche la mise en veille SYSTÈME tant que ce script tourne
# (l'écran peut s'éteindre, lui). Aucune modification de réglage Windows —
# c'est l'API standard des applications (SetThreadExecutionState), tout
# redevient normal dès que le processus s'arrête.
# Arrêt : Get-Process powershell | Where-Object { $_.CommandLine -match 'garde_eveil' } | Stop-Process
$signature = '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
$api = Add-Type -MemberDefinition $signature -Name Eveil -Namespace Util -PassThru
# 0x80000001 = ES_CONTINUOUS | ES_SYSTEM_REQUIRED
while ($true) {
    [Util.Eveil]::SetThreadExecutionState([uint32]"0x80000001") | Out-Null
    Start-Sleep -Seconds 240
}
