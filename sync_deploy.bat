@echo off
REM sync_deploy.bat
REM Sincroniza los cambios del desarrollo local a la carpeta linux_deploy/
REM NO sobreescribe linux_deploy/.env (credenciales de la MV)
REM
REM Uso: doble clic o ejecutar desde la raiz del proyecto

echo Sincronizando linux_deploy/...

xcopy /s /y src\*.py linux_deploy\src\
xcopy /s /y docs\*.md linux_deploy\docs\
copy /y requirements.txt linux_deploy\requirements.txt
copy /y run_pipeline.sh linux_deploy\run_pipeline.sh

echo.
echo Sincronizacion completada.
echo RECUERDA: linux_deploy/.env NO se ha modificado (credenciales MV).
echo Sube linux_deploy/ completo a la MV via FileZilla cuando sea necesario.
