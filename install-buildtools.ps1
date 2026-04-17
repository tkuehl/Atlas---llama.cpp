$ErrorActionPreference = 'Continue'
$logPath = 'C:\Users\tk199\AppData\Local\Temp\atlas-vsinstall.log'
"=== Start: $(Get-Date -Format o) ===" | Out-File $logPath -Encoding ascii

Stop-Service VSStandardCollectorService150 -Force -ErrorAction SilentlyContinue
"Stopped VSStandardCollectorService150" | Out-File $logPath -Append -Encoding ascii

$installer = "$env:TEMP\vs_BuildTools.exe"
Invoke-WebRequest -Uri 'https://aka.ms/vs/17/release/vs_BuildTools.exe' -OutFile $installer
"Downloaded: $installer ($((Get-Item $installer).Length) bytes)" | Out-File $logPath -Append -Encoding ascii

$vsArgs = @(
    '--wait', '--quiet', '--norestart', '--nocache', '--force',
    '--add', 'Microsoft.VisualStudio.Workload.VCTools',
    '--includeRecommended',
    '--add', 'Microsoft.VisualStudio.Component.Windows11SDK.22621'
)
"Launching: $installer $($vsArgs -join ' ')" | Out-File $logPath -Append -Encoding ascii

$p = Start-Process -FilePath $installer -ArgumentList $vsArgs -Wait -PassThru -NoNewWindow
"vs_BuildTools.exe exit: $($p.ExitCode)" | Out-File $logPath -Append -Encoding ascii
"=== End: $(Get-Date -Format o) ===" | Out-File $logPath -Append -Encoding ascii

exit $p.ExitCode
