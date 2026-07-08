# Source this script to set up the HeAnalysis build that this script is part of.

if [ "x${BASH_ARGV[0]}" = "x" ]; then
    if [ ! -f Scripts/thisanalysis.sh ]; then
        echo ERROR: must "'cd where/analysis/is'" before calling "'source Scripts/thisanalysis.sh'" for this version of bash!
        PHOTONFRAMEWORK=; export PHOTONFRAMEWORK
        return 1
    fi
    PHOTONFRAMEWORK="$PWD"; export PHOTONFRAMEWORK
else
    # get param to "."
    thisanalysis=$(dirname ${BASH_ARGV[0]})
    PHOTONFRAMEWORK=$(cd ${thisanalysis}/..;pwd); export PHOTONFRAMEWORK
fi


if [ -z "${PATH}" ]; then
   PATH=$PHOTONFRAMEWORK/bin:$PHOTONFRAMEWORK/Scripts; export PATH
else
   PATH=$PHOTONFRAMEWORK/bin:$PHOTONFRAMEWORK/Scripts:$PATH; export PATH
fi

if [ -z "${LD_LIBRARY_PATH}" ]; then
   LD_LIBRARY_PATH=$PHOTONFRAMEWORK/lib; export LD_LIBRARY_PATH       # Linux, ELF HP-UX
else
   LD_LIBRARY_PATH=$PHOTONFRAMEWORK/lib:$LD_LIBRARY_PATH; export LD_LIBRARY_PATH
fi

if [ -z "${DYLD_LIBRARY_PATH}" ]; then
   DYLD_LIBRARY_PATH=$PHOTONFRAMEWORK/lib; export DYLD_LIBRARY_PATH   # Mac OS X
else
   DYLD_LIBRARY_PATH=$PHOTONFRAMEWORK/lib:$DYLD_LIBRARY_PATH; export DYLD_LIBRARY_PATH
fi

if [ -z "${SHLIB_PATH}" ]; then
   SHLIB_PATH=$PHOTONFRAMEWORK/lib; export SHLIB_PATH                 # legacy HP-UX
else
   SHLIB_PATH=$PHOTONFRAMEWORK/lib:$SHLIB_PATH; export SHLIB_PATH
fi

if [ -z "${LIBPATH}" ]; then
   LIBPATH=$PHOTONFRAMEWORK/lib; export LIBPATH                       # AIX
else
   LIBPATH=$PHOTONFRAMEWORK/lib:$LIBPATH; export LIBPATH
fi

if [ -z "${CLUSTERS_JOB_SETTINGS}" ]; then
   CLUSTERS_JOB_SETTINGS=$PHOTONFRAMEWORK/Configuration/analysis-job-settings.cfg; export CLUSTERS_JOB_SETTINGS
else
   CLUSTERS_JOB_SETTINGS=$CLUSTERS_JOB_SETTINGS:$PHOTONFRAMEWORK/Configuration/analysis-job-settings.cfg; export CLUSTERS_JOB_SETTINGS
fi

if [ -z "${AC_COOKBOOK_MACRO_PATH}" ]; then
   AC_COOKBOOK_MACRO_PATH=$PHOTONFRAMEWORK/Macros; export AC_COOKBOOK_MACRO_PATH
else
   AC_COOKBOOK_MACRO_PATH=$AC_COOKBOOK_MACRO_PATH:$PHOTONFRAMEWORK/Macros; export AC_COOKBOOK_MACRO_PATH
fi

if [ -z "${ACSOFT_ADDITIONAL_LOGONS}" ]; then
   ACSOFT_ADDITIONAL_LOGONS=$PHOTONFRAMEWORK/rootlogon.C; export ACSOFT_ADDITIONAL_LOGONS
else
   ACSOFT_ADDITIONAL_LOGONS=$ACSOFT_ADDITIONAL_LOGONS:$PHOTONFRAMEWORK/rootlogon.C; export ACSOFT_ADDITIONAL_LOGONS
fi

if [ -z "${AC_CHOOSE_UNLOAD_SCRIPTS}" ]; then
    AC_CHOOSE_UNLOAD_SCRIPTS=${PHOTONFRAMEWORK}/Scripts/unload_thisanalysis.sh; export AC_CHOOSE_UNLOAD_SCRIPTS
else
    AC_CHOOSE_UNLOAD_SCRIPTS=${AC_CHOOSE_UNLOAD_SCRIPTS}:${PHOTONFRAMEWORK}/Scripts/unload_thisanalysis.sh; export AC_CHOOSE_UNLOAD_SCRIPTS
fi

unset thisanalysis

unset ACSOFT_DO_NOT_LOAD_ADDITIONAL_LOGONS
