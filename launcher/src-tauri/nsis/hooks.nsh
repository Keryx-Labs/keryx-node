; Portable install: Users can write keryx-launcher.json / bin / data under $INSTDIR.
; SID S-1-5-32-545 = BUILTIN\Users (locale-independent).

!macro NSIS_HOOK_POSTINSTALL
  nsExec::ExecToLog 'icacls "$INSTDIR" /grant *S-1-5-32-545:(OI)(CI)M /T /C /Q'
!macroend

!macro NSIS_HOOK_PREUNINSTALL
  Delete "$INSTDIR\keryx-launcher.json"
!macroend
