# Chien de garde de l'entraînement : relance train.exe s'il est mort.
# Appelé toutes les 5 min par la tâche planifiée « EchecTrainWatchdog ».
# SOURCE DE VÉRITÉ des arguments du régime courant — à mettre à jour à
# chaque changement de régime.
#
# COUPERET (gating) — 512 parties tous les 40 cycles, seuil 52,5 % :
#   Le budget par cycle est IDENTIQUE au régime précédent (512/40 = 64/5 =
#   12,8 parties par cycle) : le changement ne coûte pas une partie de plus.
#   Ce qu'il achète, c'est la fiabilité du verdict. Un duel de 64 parties
#   bruite l'estimation de ±36,4 Elo alors que la dérive vraie entre deux
#   couperets vaut ~+0,085 Elo : 91 % de ce qu'on mesurait était du bruit, et
#   il fallait +66 Elo de progrès réel pour être promu à 80 % de puissance —
#   autant dire jamais, sinon par tirage favorable. À 512 parties le bruit
#   tombe à ±12,9 Elo, ce qui autorise à abaisser le seuil de 55 % à 52,5 %
#   (SEUIL_PROMOTION dans src/bin/train.rs).
#   CONSÉQUENCE VOULUE : les promotions deviennent BEAUCOUP plus rares —
#   c'est le signe que le couperet fonctionne, pas qu'il est cassé.
#
# ÉCHELLE ELO — --elo-games est désormais un budget TOTAL (et non par ancre) :
#   168 = le coût historique d'une mesure dans le régime RECHERCHE + ORACLE
#   (7 ancres x 24 parties), mais réparti par l'échelle ADAPTATIVE sur les
#   seules ancres encore informatives (entrée si le dernier score tombe dans
#   [15 %, 85 %], maintien tant qu'il reste dans [10 %, 90 %] — l'hystérésis
#   évite qu'une demi-partie sur 24 fasse basculer l'ensemble d'ancres) ;
#   3 ancres actives -> 56 parties chacune. Passé explicitement ici pour que
#   le changement de sémantique soit lisible au déploiement.
#   DEUX AVERTISSEMENTS DE COÛT, à vérifier au chronomètre après déploiement :
#   1. « budget constant » vaut en PARTIES, pas en temps mural. Les ancres
#      maison étant saturées, les 168 parties vont désormais aux ancres
#      Stockfish (60 ms/coup, un processus moteur par partie) là où l'ancien
#      régime n'en jouait que 48 : ~3,5x plus de parties moteur par mesure.
#   2. en régime 1 pli ou sans --oracle, l'ancien coût n'était que de
#      120 parties (5 ancres maison x 24) : le budget total de 168 y coûte
#      40 % de plus.
#   Si le temps de mesure dérape, BAISSER --elo-games — surtout pas rétablir
#   les ancres saturées, qui ne rapportent aucune information.
#
# DÉPARTS — transition 30 % depuis le 09/08 (20 % avant) :
#   verdict du match contre le Fantôme 2800 (arbitre, 327 plis) : transition
#   51,9 cp de perte moyenne contre 6,5 à l'adversaire — la seule phase où le
#   champion décroche. Le banc (src/bin/banc.rs) situe la faute : défense des
#   pièces échouées au bord (cavalier a2/b2) et passivité dans les positions
#   générées de milieu tardif. UNE variable changée : ouvertures 0,5 -> 0,4,
#   transition 0,2 -> 0,3. Référence du banc AVANT (graine 20260809, 200k
#   nœuds, arbitre 1,5 s) : transition 28,7 cp / 6 fautes distinctes.
#   Juge : re-banc même graine après ~24-48 h + pente SF1700 sur >= 30 h
#   (jamais moins — leçon du faux plateau). Retour à 0,5/0,2 si dégradation.
#
# DRAPEAU D'ARRÊT — C:\dev\Echec\PAUSE_COUPERETS :
#   tant que ce fichier existe, l'ENTRAÎNEMENT n'est pas relancé. C'est le
#   seul mécanisme qui lit ce drapeau : il n'avait jusqu'ici aucun effet, et
#   toute exécution du script relançait train.exe malgré lui.
#   Le drapeau ne concerne QUE l'entraînement : le serveur du plateau, lui,
#   reste sous chien de garde (c'est le livrable du soir).
$pause = Test-Path "C:\dev\Echec\PAUSE_COUPERETS"

# GARDE-FOU BINAIRE PÉRIMÉ — le sens des drapeaux a changé avec ce chantier :
#   --elo-games est passé de « parties PAR ANCRE » à « budget TOTAL », et
#   SEUIL_PROMOTION de 55 % à 52,5 %. Un exe compilé AVANT lit donc les mêmes
#   arguments à l'envers : 168 par ancre = 1176 parties par mesure, et un
#   couperet de 512 parties jugé au seuil 55 % devient plus SÉVÈRE qu'avant —
#   exactement l'inverse de l'intention. Un serve.exe périmé, lui, ignore
#   --syzygy en silence (ses arguments inconnus sont jetés sans message) : le
#   plateau jouerait sans tables et rien dans serve.log ne le dirait.
#   La dégradation étant silencieuse dans les trois cas, on refuse de démarrer.
function Test-BinairePerime([string]$exe, [string[]]$sources) {
    if (-not (Test-Path $exe)) { return $true }
    $tExe = (Get-Item $exe).LastWriteTime
    foreach ($s in $sources) {
        if ((Test-Path $s) -and ((Get-Item $s).LastWriteTime -gt $tExe)) { return $true }
    }
    return $false
}
$srcCommunes = @("C:\dev\Echec\src\elo.rs", "C:\dev\Echec\src\bots.rs")
$trainPerime = Test-BinairePerime "C:\dev\Echec\target\release\train.exe" `
    (@("C:\dev\Echec\src\bin\train.rs") + $srcCommunes)
$servePerime = Test-BinairePerime "C:\dev\Echec\target\release\serve.exe" `
    (@("C:\dev\Echec\src\bin\serve.rs") + $srcCommunes)
#   Le refus est PAR BINAIRE : un train.exe périmé ne doit pas empêcher le
#   chien de garde de remettre le plateau debout, et réciproquement.
if ($trainPerime -or $servePerime) {
    # ASCII pur dans les CHAINES : ce fichier est en UTF-8 sans BOM, que
    # Windows PowerShell 5.1 relit en ANSI — un tiret cadratin y devient un
    # guillemet fermant, qui termine la chaine et casse le script.
    $msg = "REFUS : binaire release plus ancien que ses sources. "
    $msg += "Lancer 'cargo build --release --bin train --bin serve' "
    $msg += "(arreter serve.exe d'abord : il verrouille son propre exe), puis verifier "
    $msg += "que serve.log annonce les tables Syzygy et train.log l'echelle Elo adaptative."
    Add-Content -Path "C:\dev\Echec\watchdog.log" -Value ($msg + " " + (Get-Date -Format "dd/MM HH:mm:ss"))
    Write-Error $msg
}

if ((-not $pause) -and (-not $trainPerime) -and (-not (Get-Process train -ErrorAction SilentlyContinue))) {
    Start-Process -WindowStyle Hidden -FilePath "C:\dev\Echec\target\release\train.exe" `
        -ArgumentList "--out","models","--threads","18","--search-nodes","8000",`
        "--lr","0.0001","--td-lambda","0.2",`
        "--oracle","engines/stockfish/stockfish-windows-x86-64-avx2.exe",`
        "--oracle-movetime","40","--mentor-poids","1.0",`
        "--replay","2400000","--games-per-cycle","60","--eval-games","64",`
        "--gate-every","40","--gate-games","512","--elo-every","8","--elo-games","168",`
        "--departs-ouvertures","0.4","--departs-finales","0.2",`
        "--departs-transition","0.3","--syzygy","engines/syzygy","--int8" `
        -WorkingDirectory "C:\dev\Echec" `
        -RedirectStandardOutput "C:\dev\Echec\train.log" `
        -RedirectStandardError "C:\dev\Echec\train.err.log"
    Add-Content -Path "C:\dev\Echec\watchdog.log" -Value ("relance " + (Get-Date -Format "dd/MM HH:mm:ss"))
}
# Le serveur web aussi (léger, autant le garantir).
# --syzygy : le plateau joue les finales <= 5 pièces par DTZ, comme le
# self-play qui a entraîné le réseau et comme le couperet qui l'a promu.
# Sans ce drapeau, l'examinateur réclamait au réseau, seul, ce que les tables
# faisaient à sa place pendant tout son entraînement.
if ((-not $servePerime) -and (-not (Get-Process serve -ErrorAction SilentlyContinue))) {
    Start-Process -WindowStyle Hidden -FilePath "C:\dev\Echec\target\release\serve.exe" `
        -ArgumentList "--int8","--syzygy","engines/syzygy" `
        -WorkingDirectory "C:\dev\Echec" `
        -RedirectStandardOutput "C:\dev\Echec\serve.log" `
        -RedirectStandardError "C:\dev\Echec\serve.err.log"
    Add-Content -Path "C:\dev\Echec\watchdog.log" -Value ("relance serve " + (Get-Date -Format "dd/MM HH:mm:ss"))
}
