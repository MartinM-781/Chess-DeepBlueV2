# Chien de garde de l'entraînement : relance train.exe s'il est mort.
# Appelé toutes les 5 min par la tâche planifiée « EchecTrainWatchdog ».
# SOURCE DE VÉRITÉ des arguments du régime courant — à mettre à jour à
# chaque changement de régime.
if (-not (Get-Process train -ErrorAction SilentlyContinue)) {
    Start-Process -WindowStyle Hidden -FilePath "C:\dev\Echec\target\release\train.exe" `
        -ArgumentList "--out","models","--threads","18","--search-nodes","24000",`
        "--lr","0.0001","--td-lambda","0.2",`
        "--oracle","engines/stockfish/stockfish-windows-x86-64-avx2.exe",`
        "--oracle-movetime","40","--mentor-poids","1.0",`
        "--replay","2400000","--games-per-cycle","60","--eval-games","64",`
        "--gate-every","5","--elo-every","8",`
        "--departs-ouvertures","0.6","--departs-finales","0.2","--int8" `
        -WorkingDirectory "C:\dev\Echec" `
        -RedirectStandardOutput "C:\dev\Echec\train.log" `
        -RedirectStandardError "C:\dev\Echec\train.err.log"
    Add-Content -Path "C:\dev\Echec\watchdog.log" -Value ("relance " + (Get-Date -Format "dd/MM HH:mm:ss"))
}
# Le serveur web aussi (léger, autant le garantir).
if (-not (Get-Process serve -ErrorAction SilentlyContinue)) {
    Start-Process -WindowStyle Hidden -FilePath "C:\dev\Echec\target\release\serve.exe" `
        -ArgumentList "--int8" `
        -WorkingDirectory "C:\dev\Echec" `
        -RedirectStandardOutput "C:\dev\Echec\serve.log" `
        -RedirectStandardError "C:\dev\Echec\serve.err.log"
    Add-Content -Path "C:\dev\Echec\watchdog.log" -Value ("relance serve " + (Get-Date -Format "dd/MM HH:mm:ss"))
}
