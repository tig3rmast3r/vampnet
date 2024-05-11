# Imposta il percorso di destinazione sulla directory in cui risiede lo script, o sulla directory corrente se lo script non è salvato
$destinationPath = if ($PSScriptRoot) { $PSScriptRoot } else { Get-Location }
$demucsMode = "htdemucs_ft"

# Ottieni tutti i file WAV nella cartella corrente
$wavFiles = Get-ChildItem -Path . -Filter *.wav

foreach ($file in $wavFiles) {
    $filePath = $file.FullName

    # Costruisci il comando
    $command = "python -m demucs.separate -n $($demucsMode) -o `"$($destinationPath)`" --filename `"{track}-{stem}.{ext}`" --shifts 10 -j 4 `"$($filePath)`""

    # Esegui il comando
    Invoke-Expression $command
    Write-Host "Conversione completata per: $($filePath)"
}
