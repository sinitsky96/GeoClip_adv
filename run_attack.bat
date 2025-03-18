@echo off
echo Running PGDTrim untargeted attacks on GeoCLIP...

REM Set Python path - adjust if needed
set PYTHON=python

REM Run the attack script with different presets (all untargeted)
echo.
echo Running untargeted attack with default preset (using geodesic distance)...
%PYTHON% run_geoclip_attack.py --preset default --save_results --cuda --use_geodesic
if %ERRORLEVEL% NEQ 0 echo Attack failed with error code %ERRORLEVEL%, continuing with next attack...

@REM echo.
@REM echo Running untargeted attack with sparse preset (using geodesic distance)...
@REM %PYTHON% run_geoclip_attack.py --preset sparse --save_results --cuda --use_geodesic
@REM if %ERRORLEVEL% NEQ 0 echo Attack failed with error code %ERRORLEVEL%, continuing with next attack...

@REM echo.
@REM echo Running untargeted attack with aggressive preset (using geodesic distance)...
@REM %PYTHON% run_geoclip_attack.py --preset aggressive --save_results --cuda --use_geodesic
@REM if %ERRORLEVEL% NEQ 0 echo Attack failed with error code %ERRORLEVEL%, continuing with next attack...

@REM echo.
@REM echo Running untargeted attack with default preset (without geodesic distance)...
@REM %PYTHON% run_geoclip_attack.py --preset default --save_results --cuda --use_geodesic=False
@REM if %ERRORLEVEL% NEQ 0 echo Attack failed with error code %ERRORLEVEL%, continuing with next attack...

@REM echo.
@REM echo All untargeted attacks completed!
@REM pause 